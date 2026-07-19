from pydantic import BaseModel, Field

from app.schemas.ndis import NavigationPlanResponse


class ClinicalExtraction(BaseModel):
    """Structured clinical and functional details recovered from a document."""

    diagnosis_reason: str | None = None
    mobility_status: str | None = None
    transfer_status: str | None = None
    personal_care: str | None = None
    bladder_bowel: str | None = None
    skin_pressure_care: str | None = None
    cognition_mental_health: str | None = None
    living_situation: str | None = None
    carer_availability: str | None = None
    equipment_needs: str | None = None
    follow_up_requirements: str | None = None
    discharge_supports: str | None = None
    ndis_status: str | None = None
    red_flags: str | None = None


class DocumentPlanResponse(BaseModel):
    source_filename: str
    extracted_clinical_information: ClinicalExtraction
    plan: NavigationPlanResponse
    source_text_preview: str = Field(max_length=2_000)
