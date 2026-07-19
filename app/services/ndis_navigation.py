import asyncio
import json

from huggingface_hub.utils import HfHubHTTPError
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.ndis import (
    ALLOWED_SUPPORT_CATEGORIES,
    NavigationPlanRequest,
    NavigationPlanResponse,
)


class NdisNavigationError(Exception):
    """A user-safe error raised when a navigation plan cannot be generated."""


SYSTEM_PROMPT = """You are an elite Australian NDIS Specialist Support Coordinator
(Level 3), Hospital-to-Home Clinical Liaison Officer, and expert in the NDIS Act
2013. Transform the supplied hospital discharge information and participant context
into an actionable NDIS navigation plan.

Apply s34 reasonable-and-necessary criteria and the Health vs Disability interface.
Do not recommend medical treatment, sub-acute rehabilitation, or hospital-in-the-home.
Do recommend disability-related functional supports, capacity building, personal care,
and disability-related health supports where justified. Every recommendation must
address an identified functional deficit, be evidence-based and value for money, and
sustain rather than replace informal supports. When evidence is missing, recommend the
appropriate assessment instead of inventing equipment or needs.

Only use these exact support categories:
{allowed_categories}

Writing requirements:
- practical_needs_summary: 3-5 empathetic, accessible sentences for family.
- provider_referral_summary: one concise clinical B2B paragraph.
- call_script: 3-4 assertive spoken sentences requesting urgent s48 review.
- next_steps_checklist: chronological array with 5-7 concrete items.
- recommended support justifications: 1-2 sentences.

Return only the JSON object required by these parser instructions:
{format_instructions}

Clinical extraction:
{clinical_extraction}

NDIS context:
{ndis_context}
"""


class NdisNavigationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.parser = JsonOutputParser(pydantic_object=NavigationPlanResponse)

    def _chain(self):
        if not self.settings.huggingfacehub_api_token:
            raise NdisNavigationError(
                "NDIS planning is unavailable because HUGGINGFACEHUB_API_TOKEN is not configured."
            )

        model, provider = self.settings.huggingface_model_and_provider
        endpoint_options = {
            "repo_id": model,
            "task": "text-generation",
            "huggingfacehub_api_token": self.settings.huggingfacehub_api_token,
            "temperature": 0.1,
            "max_new_tokens": 2_000,
        }
        if provider:
            endpoint_options["provider"] = provider

        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT)])
        chat_model = ChatHuggingFace(
            llm=HuggingFaceEndpoint(**endpoint_options),
            model_id=model,
        )
        return prompt | chat_model | self.parser

    async def create_plan(
        self, request: NavigationPlanRequest
    ) -> NavigationPlanResponse:
        try:
            result = await asyncio.wait_for(
                self._chain().ainvoke(
                    {
                        "allowed_categories": "\n".join(
                            f"- {category}"
                            for category in sorted(ALLOWED_SUPPORT_CATEGORIES)
                        ),
                        "format_instructions": self.parser.get_format_instructions(),
                        "clinical_extraction": json.dumps(request.clinical_extraction),
                        "ndis_context": json.dumps(request.ndis_context),
                    }
                ),
                timeout=self.settings.ndis_request_timeout_seconds,
            )
            return NavigationPlanResponse.model_validate(result)
        except asyncio.TimeoutError as error:
            raise NdisNavigationError("The NDIS planning model timed out.") from error
        except (OutputParserException, ValidationError) as error:
            raise NdisNavigationError(
                "The NDIS planning model returned an invalid response. Please try again."
            ) from error
        except HfHubHTTPError as error:
            raise NdisNavigationError(
                "The NDIS planning model is unavailable. Please try again shortly."
            ) from error
