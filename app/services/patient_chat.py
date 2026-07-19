import asyncio
import json
import logging
import re

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import ValidationError

from app.core.ai_trace import traced
from app.core.config import Settings
from app.core.llm import (
    LlmNotConfiguredError,
    build_chat_model,
    is_provider_rate_limited,
    retry_transient_provider_errors,
)
from app.schemas.chat import (
    ChatRole,
    ConversationStage,
    PatientChatRequest,
    PatientChatResponse,
)
from app.schemas.ndis import ALLOWED_SUPPORT_CATEGORIES

logger = logging.getLogger(__name__)

CHAT_DISCLAIMER = (
    "I can provide general NDIS navigation information, but I am not a clinician, "
    "lawyer, emergency service, or substitute for the NDIA or your support coordinator."
)
URGENT_MESSAGE = (
    "If there is immediate danger or a medical emergency, call Triple Zero (000) now. "
    "For urgent health advice, contact your treating team, GP, or Healthdirect on 1800 022 222."
)

URGENT_PATTERNS = (
    r"\b(?:suicid(?:e|al)|self[- ]?harm|want to die)\b",
    r"\b(?:chest pain|can(?:not|'t) breathe|difficulty breathing|severe shortness of breath)\b",
    r"\b(?:face droop|slurred speech|stroke symptoms|sudden weakness)\b",
    r"\b(?:immediate danger|unsafe right now|being abused|violence at home)\b",
)

PATIENT_CHAT_PROMPT = """You are a patient-facing Australian NDIS navigation assistant.
Use plain, compassionate Australian English and be practical, concise, and accessible.
You are not a clinician, lawyer, emergency service, the NDIA, or a support coordinator.
Do not diagnose, provide medical treatment, promise NDIS eligibility or funding, invent
plan details, or give funding amounts.

First listen and understand. When necessary, ask at most two high-value follow-up questions
about functional impact, daily activities, safety, informal carer capacity, goals, living
arrangements, current NDIS plan, or immediate supports. Only make recommendations once enough
functional information is available. Explain the health-versus-disability interface: medical
treatment and acute care are not NDIS-funded, while disability-related daily supports,
capacity building, and assistive technology may be relevant.

Only use these exact NDIS support categories in recommendations:
{allowed_categories}

Always return a JSON object matching these instructions:
{format_instructions}

Set urgent_action to true and conversation_stage to escalation when the patient describes an
immediate health, safety, abuse, or self-harm concern. In that case, clearly direct them to
call Triple Zero (000) or appropriate urgent clinical support, do not continue NDIS planning,
and provide no NDIS recommendations. Do not include a disclaimer; the application adds one.

Participant context, which may be empty:
{ndis_context}

Profile-review context, which is empty for a general chat:
{profile_review}

When profile-review context is present, help the participant confirm their referral
profile. Ask follow-up questions only when a missing, edited, or unconfirmed detail
would materially affect daily support needs, safety, matching, or preferences. Do not
repeat information already confirmed. Keep the participant in control: suggestions are
optional and must be explicitly confirmed by them before they are treated as approved.
"""


class PatientChatError(Exception):
    """A user-safe error raised when a patient chat response cannot be created."""


class PatientChatInputError(PatientChatError):
    """An input exceeds the configured client-history limits."""


class PatientChatService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.parser = JsonOutputParser(pydantic_object=PatientChatResponse)

    def _chain(self):
        try:
            chat_model = build_chat_model(
                self.settings, temperature=0.2, max_output_tokens=1_500
            )
        except LlmNotConfiguredError as error:
            raise PatientChatError(str(error)) from error

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PATIENT_CHAT_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{message}"),
            ]
        )
        return prompt | chat_model | self.parser

    def _validate_limits(self, request: PatientChatRequest) -> None:
        if len(request.history) > self.settings.chat_history_limit:
            raise PatientChatInputError(
                f"history cannot contain more than {self.settings.chat_history_limit} messages"
            )
        all_messages = [*(item.content for item in request.history), request.message]
        if any(
            len(message) > self.settings.chat_message_max_characters
            for message in all_messages
        ):
            raise PatientChatInputError(
                "a message exceeds the configured maximum length"
            )

    @staticmethod
    def _to_langchain_history(request: PatientChatRequest) -> list[HumanMessage | AIMessage]:
        return [
            HumanMessage(content=item.content)
            if item.role == ChatRole.USER
            else AIMessage(content=item.content)
            for item in request.history
        ]

    @staticmethod
    def _requires_urgent_action(message: str) -> bool:
        return any(
            re.search(pattern, message, re.IGNORECASE) for pattern in URGENT_PATTERNS
        )

    def _urgent_response(self) -> PatientChatResponse:
        return PatientChatResponse(
            reply=URGENT_MESSAGE,
            conversation_stage=ConversationStage.ESCALATION,
            urgent_action=True,
            urgent_message=URGENT_MESSAGE,
            disclaimer=CHAT_DISCLAIMER,
        )

    async def reply(self, request: PatientChatRequest) -> PatientChatResponse:
        self._validate_limits(request)
        if self._requires_urgent_action(request.message):
            return self._urgent_response()

        try:
            result = await asyncio.wait_for(
                traced(retry_transient_provider_errors(self._chain()), "patient_chat").ainvoke(
                    {
                        "history": self._to_langchain_history(request),
                        "message": request.message,
                        "ndis_context": json.dumps(request.ndis_context or {}),
                        "profile_review": json.dumps(
                            request.profile_review.model_dump()
                            if request.profile_review
                            else {}
                        ),
                        "allowed_categories": "\n".join(
                            f"- {category}"
                            for category in sorted(ALLOWED_SUPPORT_CATEGORIES)
                        ),
                        "format_instructions": self.parser.get_format_instructions(),
                    },
                ),
                timeout=self.settings.patient_chat_timeout_seconds,
            )
            response = PatientChatResponse.model_validate(result)
            return response.model_copy(update={"disclaimer": CHAT_DISCLAIMER})
        except asyncio.TimeoutError as error:
            raise PatientChatError("The patient chat service timed out. Please try again.") from error
        except (OutputParserException, ValidationError) as error:
            raise PatientChatError(
                "The patient chat service returned an invalid response. Please try again."
            ) from error
        except PatientChatError:
            raise
        except Exception as error:
            if is_provider_rate_limited(error):
                raise PatientChatError(
                    "The Gemma service is temporarily rate limited. Please wait a minute and try again."
                ) from error
            logger.exception("Patient chat provider request failed")
            raise PatientChatError(
                "The patient chat service is temporarily unavailable. Please try again."
            ) from error
