import asyncio
import json

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.action_pack import NdisActionPackResponse

PROMPT = """You create an Australian NDIS discharge action pack. Use only documented
clinical information. Apply the NDIS health/disability interface and s34. Do not promise
eligibility, funding, provider availability, or invent evidence. Recommend an access request
for non-participants with significant permanent impairment; recommend s48 review only for
active plans made insufficient by documented change. Provider categories are service types,
not named local providers. Return only JSON matching:
{format_instructions}
Clinical information: {clinical_extraction}
NDIS context: {ndis_context}"""


class NdisActionPackError(Exception):
    pass


class NdisActionPackService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.parser = JsonOutputParser(pydantic_object=NdisActionPackResponse)

    def _chain(self):
        if not self.settings.huggingfacehub_api_token:
            raise NdisActionPackError("Action-pack AI provider is not configured.")
        model, provider = self.settings.huggingface_model_and_provider
        options = {"repo_id": model, "task": "text-generation", "huggingfacehub_api_token": self.settings.huggingfacehub_api_token, "temperature": 0.1, "max_new_tokens": 2500}
        if provider:
            options["provider"] = provider
        return ChatPromptTemplate.from_messages([("system", PROMPT)]) | ChatHuggingFace(llm=HuggingFaceEndpoint(**options), model_id=model) | self.parser

    async def create(self, clinical_extraction: dict, ndis_context: dict) -> NdisActionPackResponse:
        try:
            result = await asyncio.wait_for(self._chain().ainvoke({"clinical_extraction": json.dumps(clinical_extraction), "ndis_context": json.dumps(ndis_context), "format_instructions": self.parser.get_format_instructions()}), timeout=self.settings.action_pack_request_timeout_seconds)
            return NdisActionPackResponse.model_validate(result)
        except asyncio.TimeoutError as error:
            raise NdisActionPackError("Action-pack generation timed out.") from error
        except (ValidationError, Exception) as error:
            if isinstance(error, NdisActionPackError):
                raise
            raise NdisActionPackError("Action-pack generation returned an invalid response.") from error
