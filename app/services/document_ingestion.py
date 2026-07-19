import asyncio
import base64
import io
import json
from pathlib import Path

from huggingface_hub.utils import HfHubHTTPError
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import ValidationError
from pypdf import PdfReader

from app.core.ai_trace import trace_config
from app.core.config import Settings
from app.schemas.document import ClinicalExtraction, DocumentPlanResponse
from app.schemas.ndis import NavigationPlanRequest
from app.services.ndis_navigation import NdisNavigationError, NdisNavigationService

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png"}
SUPPORTED_PDF_TYPES = {"application/pdf"}

EXTRACTION_PROMPT = """You are an Australian hospital-to-home clinical information
extractor. Read the supplied discharge-summary document and return only the structured
JSON requested below. Extract only information explicitly stated in the document. Do not
infer diagnoses, equipment, NDIS eligibility, or support needs not documented. Preserve
uncertainty by using null for missing fields. This task is for a fictional software
demonstration and is not a clinical decision.

{format_instructions}
"""


class DocumentIngestionError(Exception):
    """A user-safe error raised when a clinical document cannot be processed."""


class DocumentInputError(DocumentIngestionError):
    """A file fails validation before it is sent to the model."""


class DocumentIngestionService:
    def __init__(
        self, settings: Settings, navigation_service: NdisNavigationService
    ) -> None:
        self.settings = settings
        self.navigation_service = navigation_service
        self.parser = JsonOutputParser(pydantic_object=ClinicalExtraction)

    def _chat_model(self) -> ChatHuggingFace:
        if not self.settings.huggingfacehub_api_token:
            raise DocumentIngestionError(
                "Document processing is unavailable because its AI provider is not configured."
            )

        model, provider = self.settings.huggingface_model_and_provider
        options = {
            "repo_id": model,
            "task": "text-generation",
            "huggingfacehub_api_token": self.settings.huggingfacehub_api_token,
            "temperature": 0.0,
            "max_new_tokens": 2_000,
        }
        if provider:
            options["provider"] = provider
        return ChatHuggingFace(
            llm=HuggingFaceEndpoint(**options),
            model_id=model,
        )

    def _chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", EXTRACTION_PROMPT),
                MessagesPlaceholder(variable_name="document_message"),
            ]
        )
        return prompt | self._chat_model() | self.parser

    @staticmethod
    def _file_type(filename: str, content_type: str | None) -> str:
        if content_type in SUPPORTED_PDF_TYPES | SUPPORTED_IMAGE_TYPES:
            return content_type
        extension = Path(filename).suffix.lower()
        return {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
        }.get(extension, "")

    @staticmethod
    def _extract_pdf_text(content: bytes) -> str:
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    def _validate_file(
        self, filename: str, content_type: str | None, content: bytes
    ) -> str:
        if not filename:
            raise DocumentInputError("A filename is required.")
        if not content:
            raise DocumentInputError("The uploaded document is empty.")
        if len(content) > self.settings.document_max_upload_bytes:
            raise DocumentInputError("The uploaded document exceeds the size limit.")

        file_type = self._file_type(filename, content_type)
        if file_type not in SUPPORTED_PDF_TYPES | SUPPORTED_IMAGE_TYPES:
            raise DocumentInputError("Only PDF, PNG, and JPEG documents are supported.")
        return file_type

    async def create_plan_from_document(
        self,
        filename: str,
        content_type: str | None,
        content: bytes,
        ndis_context: dict,
    ) -> DocumentPlanResponse:
        file_type = self._validate_file(filename, content_type, content)
        if file_type in SUPPORTED_PDF_TYPES:
            try:
                source_text = await asyncio.to_thread(self._extract_pdf_text, content)
            except Exception as error:
                raise DocumentInputError("The PDF could not be read.") from error
            if not source_text:
                raise DocumentInputError(
                    "The PDF contains no selectable text. Upload clear page images instead."
                )
            document_content: str | list[dict] = source_text
            preview = source_text[:2_000]
        else:
            encoded_image = base64.b64encode(content).decode("ascii")
            document_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{file_type};base64,{encoded_image}",
                    },
                },
                {
                    "type": "text",
                    "text": "Extract the clinical and functional information from this image.",
                },
            ]
            preview = "Clinical information was extracted from an uploaded image."

        try:
            extracted = await asyncio.wait_for(
                self._chain().ainvoke(
                    {
                        "document_message": [HumanMessage(content=document_content)],
                        "format_instructions": self.parser.get_format_instructions(),
                    },
                    config=trace_config("document_extraction"),
                ),
                timeout=self.settings.document_request_timeout_seconds,
            )
            clinical_information = ClinicalExtraction.model_validate(extracted)
            plan = await self.navigation_service.create_plan(
                NavigationPlanRequest(
                    clinical_extraction=clinical_information.model_dump(
                        exclude_none=True
                    ),
                    ndis_context=ndis_context or {"context": "Not provided"},
                )
            )
            return DocumentPlanResponse(
                source_filename=filename,
                extracted_clinical_information=clinical_information,
                plan=plan,
                source_text_preview=preview,
            )
        except asyncio.TimeoutError as error:
            raise DocumentIngestionError("Document processing timed out.") from error
        except (OutputParserException, ValidationError) as error:
            raise DocumentIngestionError(
                "The document could not be converted into a reliable clinical summary."
            ) from error
        except (HfHubHTTPError, NdisNavigationError) as error:
            raise DocumentIngestionError(
                "The document processing service is temporarily unavailable."
            ) from error

    async def extract_clinical_information(
        self, filename: str, content_type: str | None, content: bytes
    ) -> tuple[ClinicalExtraction, str]:
        file_type = self._validate_file(filename, content_type, content)
        if file_type in SUPPORTED_PDF_TYPES:
            source_text = await asyncio.to_thread(self._extract_pdf_text, content)
            if not source_text:
                raise DocumentInputError("The PDF contains no selectable text. Upload clear page images instead.")
            document_content: str | list[dict] = source_text
            preview = source_text[:2_000]
        else:
            encoded_image = base64.b64encode(content).decode("ascii")
            document_content = [{"type": "image_url", "image_url": {"url": f"data:{file_type};base64,{encoded_image}"}}, {"type": "text", "text": "Extract clinical and functional information."}]
            preview = "Clinical information was extracted from an uploaded image."
        try:
            extracted = await asyncio.wait_for(self._chain().ainvoke({"document_message": [HumanMessage(content=document_content)], "format_instructions": self.parser.get_format_instructions()}, config=trace_config("document_extraction")), timeout=self.settings.document_request_timeout_seconds)
            return ClinicalExtraction.model_validate(extracted), preview
        except Exception as error:
            raise DocumentIngestionError("The document could not be converted into a reliable clinical summary.") from error
