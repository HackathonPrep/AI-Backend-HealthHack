import asyncio
import json
import logging

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from app.core.ai_trace import traced
from app.core.config import Settings
from app.core.llm import (
    LlmNotConfiguredError,
    build_chat_model,
    is_provider_rate_limited,
    retry_transient_provider_errors,
)
from app.schemas.action_pack import NdisActionPackResponse

logger = logging.getLogger(__name__)

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
        try:
            chat_model = build_chat_model(
                self.settings, temperature=0.1, max_output_tokens=4_000
            )
        except LlmNotConfiguredError as error:
            raise NdisActionPackError(str(error)) from error
        return (
            ChatPromptTemplate.from_messages(
                [
                    ("system", PROMPT),
                    (
                        "human",
                        "Generate the NDIS discharge action pack from the supplied clinical information and context.",
                    ),
                ]
            )
            | chat_model
            | self.parser
        )

    async def create(
        self, clinical_extraction: dict, ndis_context: dict
    ) -> NdisActionPackResponse:
        try:
            result = await asyncio.wait_for(
                traced(retry_transient_provider_errors(self._chain()), "ndis_action_pack").ainvoke(
                    {
                        "clinical_extraction": json.dumps(clinical_extraction),
                        "ndis_context": json.dumps(ndis_context),
                        "format_instructions": self.parser.get_format_instructions(),
                    }
                ),
                timeout=self.settings.action_pack_request_timeout_seconds,
            )
            return NdisActionPackResponse.model_validate(result)
        except asyncio.TimeoutError as error:
            raise NdisActionPackError("Action-pack generation timed out.") from error
        except NdisActionPackError:
            raise
        except (ValidationError, Exception) as error:
            if is_provider_rate_limited(error):
                raise NdisActionPackError(
                    "The Gemma service is temporarily rate limited. Please wait a minute and try again."
                ) from error
            logger.exception("Action-pack provider request failed")
            raise NdisActionPackError(
                "Action-pack generation returned an invalid response."
            ) from error
