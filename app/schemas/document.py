from pydantic import BaseModel, Field

from app.schemas.ndis import NavigationPlanResponse


class ClinicalExtraction(BaseModel):
    """Structured clinical and functional details from a hospital discharge summary.

    Tuned for Australian hospital-to-home / new-disability discharge summaries
    used in CareMatch demos (e.g. traumatic SCI with NDIS access request).
    """

    patient_summary: str | None = Field(
        default=None,
        description="Brief identity/context line: age band, location, admitting unit if stated.",
    )
    diagnosis_reason: str | None = Field(
        default=None,
        description="Reason for admission and principal diagnosis.",
    )
    procedures_hospital_course: str | None = Field(
        default=None,
        description="Key procedures and hospital course milestones.",
    )
    mobility_status: str | None = None
    transfer_status: str | None = None
    personal_care: str | None = None
    bladder_bowel: str | None = None
    skin_pressure_care: str | None = None
    cognition_mental_health: str | None = None
    living_situation: str | None = None
    carer_availability: str | None = None
    equipment_needs: str | None = Field(
        default=None,
        description="Assistive technology and equipment named in the document.",
    )
    home_modifications: str | None = Field(
        default=None,
        description="Home access or modification needs named in the document.",
    )
    discharge_supports: str | None = Field(
        default=None,
        description=(
            "Supports the treating team says are required at discharge "
            "(support coordination, support workers, OT, physio, nursing, transport, psychology, etc.)."
        ),
    )
    medications_on_discharge: str | None = None
    follow_up_requirements: str | None = None
    ndis_status: str | None = Field(
        default=None,
        description=(
            "NDIS participation status and required action, e.g. not currently a participant "
            "and advised to commence an access request, or existing participant needing review."
        ),
    )
    red_flags: str | None = None


class DocumentPlanResponse(BaseModel):
    source_filename: str
    extracted_clinical_information: ClinicalExtraction
    plan: NavigationPlanResponse
    source_text_preview: str = Field(max_length=2_000)
    pathway: str | None = Field(
        default=None,
        description="ndis_access_request | plan_review | unknown — derived for the demo UI.",
    )
