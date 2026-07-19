import asyncio
import json
import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from app.core.ai_trace import traced
from app.core.config import Settings
from app.core.llm import LlmNotConfiguredError, build_chat_model
from app.schemas.ndis import (
    ALLOWED_SUPPORT_CATEGORIES,
    NavigationPlanRequest,
    NavigationPlanResponse,
)

logger = logging.getLogger(__name__)


class NdisNavigationError(Exception):
    """A user-safe error raised when a navigation plan cannot be generated."""


SYSTEM_PROMPT = """You are an elite Australian NDIS Specialist Support Coordinator
(Level 3), Hospital-to-Home Clinical Liaison Officer, and expert in the NDIS Act
2013. Transform the supplied hospital discharge information and participant context
into an actionable NDIS navigation plan for CareMatch matching.

Apply s34 reasonable-and-necessary criteria and the Health vs Disability interface.
Do not recommend medical treatment, sub-acute rehabilitation, or hospital-in-the-home.
Do recommend disability-related functional supports, capacity building, personal care,
consumables, assistive technology, home modifications, transport, and support
coordination where justified by documented functional deficits. Every recommendation
must address an identified functional deficit, be evidence-based and value for money,
and sustain rather than replace informal supports. When evidence is missing, recommend
the appropriate assessment instead of inventing equipment or needs.

Pathway rules (read ndis_context.pathway and clinical ndis_status carefully):
- If pathway is ndis_access_request, or the person is not yet an NDIS participant:
  focus on urgent NDIS access request evidence, interim disability supports while
  waiting, support coordination to stand up providers, and do NOT recommend an s48
  plan review. call_script must help a family member or coordinator request urgent
  access / planning support (not s48).
- If pathway is plan_review, or the person already has an active plan:
  focus on change-of-circumstances / s48 review where the plan is insufficient.
  call_script may request urgent s48 review.
- If pathway is unknown, prefer access-request language when the extraction says the
  person was not a participant; otherwise keep recommendations pathway-neutral.

Only use these exact support categories:
{allowed_categories}

Writing requirements:
- practical_needs_summary: 3-5 empathetic, accessible sentences for family.
- provider_referral_summary: one concise clinical B2B paragraph for providers.
- call_script: 3-4 assertive spoken sentences matching the pathway above.
- next_steps_checklist: chronological array with 5-7 concrete items (include access
  request or plan review as appropriate, OT home visit, personal care start, AT,
  continence consumables, therapy, and GP/spinal follow-up when documented).
- recommended support justifications: 1-2 sentences tied to documented deficits.

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
        try:
            chat_model = build_chat_model(
                self.settings, temperature=0.1, max_output_tokens=4_000
            )
        except LlmNotConfiguredError as error:
            raise NdisNavigationError(str(error)) from error

        # Gemini 3.5 requires at least one user-content message; system-only
        # requests are rejected with "contents are required".
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Generate the NDIS navigation plan using only the supplied clinical extraction and NDIS context.",
                ),
            ]
        )
        return prompt | chat_model | self.parser

    async def create_plan(
        self, request: NavigationPlanRequest
    ) -> NavigationPlanResponse:
        try:
            result = await asyncio.wait_for(
                traced(self._chain(), "ndis_navigation").ainvoke(
                    {
                        "allowed_categories": "\n".join(
                            f"- {category}"
                            for category in sorted(ALLOWED_SUPPORT_CATEGORIES)
                        ),
                        "format_instructions": self.parser.get_format_instructions(),
                        "clinical_extraction": json.dumps(request.clinical_extraction),
                        "ndis_context": json.dumps(request.ndis_context),
                    },
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
        except NdisNavigationError:
            raise
        except Exception as error:
            logger.exception("NDIS planning provider request failed")
            raise NdisNavigationError(
                "The NDIS planning model is unavailable. Please try again shortly."
            ) from error
