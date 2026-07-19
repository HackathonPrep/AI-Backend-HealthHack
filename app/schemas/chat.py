from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.ndis import ALLOWED_SUPPORT_CATEGORIES


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationMode(str, Enum):
    GENERAL = "general"
    PROFILE_REVIEW = "profile_review"


class ChatMessage(BaseModel):
    """A client-owned chat turn. System instructions are always server-owned."""

    role: ChatRole
    content: str = Field(min_length=1, max_length=4_000)

    @field_validator("content")
    @classmethod
    def reject_blank_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content cannot be blank")
        return value


class ProfileReviewContext(BaseModel):
    """The minimum structured context needed for a profile-review clarification."""

    current_section: str = Field(min_length=1, max_length=160)
    pending_sections: list[str] = Field(default_factory=list, max_length=12)
    confirmed_sections: list[str] = Field(default_factory=list, max_length=12)
    edited_sections: list[str] = Field(default_factory=list, max_length=12)
    support_needs: list[str] = Field(default_factory=list, max_length=12)

    @field_validator(
        "current_section",
        "pending_sections",
        "confirmed_sections",
        "edited_sections",
        "support_needs",
    )
    @classmethod
    def strip_context_values(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str):
            return value.strip()
        return [item.strip() for item in value if item and item.strip()]


class PatientChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=50)
    ndis_context: dict[str, Any] | None = None
    session_id: str | None = Field(default=None, max_length=100)
    mode: ConversationMode = ConversationMode.GENERAL
    profile_review: ProfileReviewContext | None = None

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_history_order(self) -> "PatientChatRequest":
        for index, item in enumerate(self.history):
            expected_role = ChatRole.USER if index % 2 == 0 else ChatRole.ASSISTANT
            if item.role != expected_role:
                raise ValueError("history must alternate user and assistant messages")
        if self.history and self.history[-1].role != ChatRole.ASSISTANT:
            raise ValueError("history must end with an assistant message")
        if self.mode == ConversationMode.PROFILE_REVIEW and self.profile_review is None:
            raise ValueError("profile_review context is required in profile_review mode")
        return self


class ConversationStage(str, Enum):
    INTAKE = "intake"
    CLARIFYING = "clarifying"
    RECOMMENDATION = "recommendation"
    ESCALATION = "escalation"


class ChatRecommendation(BaseModel):
    category: str
    recommendation: str = Field(min_length=1, max_length=1_000)

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str) -> str:
        if value not in ALLOWED_SUPPORT_CATEGORIES:
            raise ValueError("category is not an allowed NDIS support category")
        return value


class PatientChatResponse(BaseModel):
    reply: str = Field(min_length=1, max_length=4_000)
    conversation_stage: ConversationStage
    follow_up_questions: list[str] = Field(default_factory=list, max_length=2)
    recommendations: list[ChatRecommendation] = Field(default_factory=list, max_length=3)
    urgent_action: bool = False
    urgent_message: str | None = Field(default=None, max_length=1_000)
    # Optional from the model; the service always replaces this with CHAT_DISCLAIMER.
    disclaimer: str = Field(default="", max_length=1_000)

    @model_validator(mode="after")
    def validate_urgent_response(self) -> "PatientChatResponse":
        if self.urgent_action and not self.urgent_message:
            raise ValueError("urgent_message is required when urgent_action is true")
        return self
