from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class ProfileSectionStatus(str, Enum):
    CONFIRMED = "confirmed"
    REMOVED = "removed"


class ApprovedProfileSection(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=160)
    status: ProfileSectionStatus
    items: list[str] = Field(default_factory=list, max_length=30)

    @field_validator("items")
    @classmethod
    def normalise_items(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value and value.strip()]


class ProfileFollowUpAnswer(BaseModel):
    question: str = Field(min_length=1, max_length=1_000)
    answer: str = Field(min_length=1, max_length=2_000)

    @field_validator("question", "answer")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


REQUIRED_CONSENTS = frozenset(
    {"reviewed", "matching", "no_automatic_document_sharing"}
)


class ProfileApprovalRequest(BaseModel):
    sections: list[ApprovedProfileSection] = Field(min_length=1, max_length=12)
    follow_up_answers: list[ProfileFollowUpAnswer] = Field(default_factory=list, max_length=10)
    consents: list[str] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_approval(self) -> "ProfileApprovalRequest":
        if len({section.id for section in self.sections}) != len(self.sections):
            raise ValueError("section ids must be unique")
        if set(self.consents) != REQUIRED_CONSENTS:
            raise ValueError("all required consents must be supplied")
        if not any(section.status == ProfileSectionStatus.CONFIRMED for section in self.sections):
            raise ValueError("at least one profile section must be confirmed")
        return self


class ProfileApprovalResponse(BaseModel):
    id: str
    created_at: str
    approved_profile: dict
    follow_up_answers: list[ProfileFollowUpAnswer]
    consents: list[str]
