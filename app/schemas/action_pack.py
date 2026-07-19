from enum import Enum

from pydantic import BaseModel, Field, model_validator

from app.schemas.document import ClinicalExtraction


class EvidenceStatus(str, Enum):
    PRESENT = "present"
    MISSING = "missing"
    UNKNOWN = "unknown"


class EvidenceItem(BaseModel):
    item: str = Field(min_length=1)
    status: EvidenceStatus
    source_hint: str = Field(min_length=1)


class FollowUpTask(BaseModel):
    task: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    priority: str = Field(pattern="^(urgent|high|routine)$")


class NdisActionPackResponse(BaseModel):
    practical_needs_summary: str = Field(min_length=1)
    evidence_checklist: list[EvidenceItem] = Field(min_length=3)
    access_or_review_recommended: bool
    access_or_review_rationale: str | None = None
    provider_service_categories: list[str] = Field(min_length=1)
    provider_referral_summary: str = Field(min_length=1)
    family_call_script: str = Field(min_length=1)
    email_draft: str = Field(min_length=1)
    follow_up_tasks: list[FollowUpTask] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_review_rationale(self) -> "NdisActionPackResponse":
        if self.access_or_review_recommended and not self.access_or_review_rationale:
            raise ValueError("access_or_review_rationale is required when recommendation is true")
        return self


class DocumentActionPackResponse(BaseModel):
    source_filename: str
    extracted_clinical_information: ClinicalExtraction
    action_pack: NdisActionPackResponse
    source_text_preview: str = Field(max_length=2_000)
