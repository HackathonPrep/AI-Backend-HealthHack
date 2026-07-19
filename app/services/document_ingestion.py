import asyncio
import base64
import io
import json
import logging
from pathlib import Path

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace
from pydantic import ValidationError
from pypdf import PdfReader

from app.core.ai_trace import traced
from app.core.config import Settings
from app.core.llm import (
    LlmNotConfiguredError,
    build_chat_model,
    is_provider_rate_limited,
    retry_transient_provider_errors,
)
from app.schemas.document import ClinicalExtraction, DocumentPlanResponse
from app.schemas.ndis import NavigationPlanRequest
from app.services.ndis_navigation import NdisNavigationError, NdisNavigationService

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png"}
SUPPORTED_PDF_TYPES = {"application/pdf"}

EXTRACTION_PROMPT = """You are an Australian hospital-to-home clinical information
extractor specialising in discharge summaries written for people with a new or changed
disability who may need NDIS access or plan review (for example spinal cord injury,
stroke, acquired brain injury, or other inpatient rehabilitation discharges).

Read the supplied document and return only the structured JSON requested below.
Extract only information explicitly stated. Do not invent diagnoses, equipment, NDIS
eligibility, or supports. Use null for missing fields. This task is for a fictional
software demonstration and is not a clinical decision.

Document structure cues commonly present in these summaries:
- Patient demographics, admission/discharge dates, discharging unit
- Reason for admission / principal diagnosis / procedures and hospital course
- Functional status at discharge (mobility, transfers, personal care, bladder/bowel,
  skin, cognition/mental health, living situation, informal supports)
- Discharge supports required (support coordination, support workers, OT, physiotherapy,
  assistive technology, home modifications, community nursing, transport, psychology)
- NDIS status and action required (new access request vs existing participant review)
- Medications on discharge, follow-up, red flags

Mapping guidance:
- Put AT/equipment lists into equipment_needs and bathroom/access work into home_modifications.
- Put the full “Discharge Supports Required” narrative into discharge_supports.
- Put NDIS participation and access/review advice into ndis_status verbatim in substance.
- Combine bladder and bowel into bladder_bowel when both are present.

{format_instructions}
"""


class DocumentIngestionError(Exception):
    """A user-safe error raised when a clinical document cannot be processed."""


class DocumentInputError(DocumentIngestionError):
    """A file fails validation before it is sent to the model."""


def resolve_ndis_pathway(
    extraction: ClinicalExtraction, ndis_context: dict | None
) -> tuple[str, dict]:
    """Derive access-request vs plan-review pathway and enrich ndis_context."""
    context = dict(ndis_context or {})
    status = (extraction.ndis_status or "").lower()
    has_plan_hint = context.get("has_active_plan")

    access_markers = (
        "not an ndis",
        "not a participant",
        "non-participant",
        "no active plan",
        "access request",
        "commence an ndis",
        "commence ndis",
        "apply for ndis",
        "ndis access",
    )
    review_markers = (
        "active plan",
        "existing participant",
        "current participant",
        "s48",
        "plan review",
        "change of circumstances",
    )

    if any(marker in status for marker in access_markers) or has_plan_hint is False:
        pathway = "ndis_access_request"
        context["has_active_plan"] = False
    elif any(marker in status for marker in review_markers) or has_plan_hint is True:
        pathway = "plan_review"
        context["has_active_plan"] = True
    else:
        pathway = "unknown"

    context["pathway"] = pathway
    if extraction.discharge_supports:
        context.setdefault("documented_discharge_supports", extraction.discharge_supports)
    return pathway, context


class DocumentIngestionService:
    def __init__(
        self, settings: Settings, navigation_service: NdisNavigationService
    ) -> None:
        self.settings = settings
        self.navigation_service = navigation_service
        self.parser = JsonOutputParser(pydantic_object=ClinicalExtraction)

    def _chat_model(self) -> ChatHuggingFace:
        try:
            return build_chat_model(
                self.settings, temperature=0.0, max_output_tokens=2_000
            )
        except LlmNotConfiguredError as error:
            raise DocumentIngestionError(str(error)) from error

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
                traced(retry_transient_provider_errors(self._chain()), "document_extraction").ainvoke(
                    {
                        "document_message": [HumanMessage(content=document_content)],
                        "format_instructions": self.parser.get_format_instructions(),
                    },
                ),
                timeout=self.settings.document_request_timeout_seconds,
            )
            clinical_information = ClinicalExtraction.model_validate(extracted)
            pathway, plan_context = resolve_ndis_pathway(
                clinical_information, ndis_context
            )
            plan = await self.navigation_service.create_plan(
                NavigationPlanRequest(
                    clinical_extraction=clinical_information.model_dump(
                        exclude_none=True
                    ),
                    ndis_context=plan_context or {"context": "Not provided"},
                )
            )
            return DocumentPlanResponse(
                source_filename=filename,
                extracted_clinical_information=clinical_information,
                plan=plan,
                source_text_preview=preview,
                pathway=pathway,
            )
        except asyncio.TimeoutError as error:
            raise DocumentIngestionError("Document processing timed out.") from error
        except (OutputParserException, ValidationError) as error:
            raise DocumentIngestionError(
                "The document could not be converted into a reliable clinical summary."
            ) from error
        except NdisNavigationError as error:
            # This used to hide every planning failure behind a single generic
            # message. Keep the safe, user-facing error produced by the
            # navigation service so an invalid model response, timeout, or
            # unavailable provider can be acted on correctly.
            raise DocumentIngestionError(str(error)) from error
        except Exception as error:
            if is_provider_rate_limited(error):
                raise DocumentIngestionError(
                    "The Gemma service is temporarily rate limited. Please wait a minute and try again."
                ) from error
            logger.exception("Unexpected document-processing failure")
            raise DocumentIngestionError(
                "The document could not be processed. Please try again with a smaller text-based PDF or a clear page image."
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
            extracted = await asyncio.wait_for(traced(retry_transient_provider_errors(self._chain()), "document_extraction").ainvoke({"document_message": [HumanMessage(content=document_content)], "format_instructions": self.parser.get_format_instructions()}), timeout=self.settings.document_request_timeout_seconds)
            return ClinicalExtraction.model_validate(extracted), preview
        except Exception as error:
            if is_provider_rate_limited(error):
                raise DocumentIngestionError(
                    "The Gemma service is temporarily rate limited. Please wait a minute and try again."
                ) from error
            raise DocumentIngestionError("The document could not be converted into a reliable clinical summary.") from error
