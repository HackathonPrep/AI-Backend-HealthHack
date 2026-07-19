from typing import Any

from pydantic import BaseModel, Field, field_validator


ALLOWED_SUPPORT_CATEGORIES = {
    "Core: Assistance with Daily Life (Support Workers, SIL)",
    "Core: Consumables (Wound care, continence products, low-cost AT)",
    "Core: Assistance with Social, Economic and Community Participation",
    "Core: Transport",
    "Capacity Building: Improved Daily Living (Occupational Therapy, Physiotherapy, Speech Pathology, Psychology, Dietetics, Community Nursing)",
    "Capacity Building: Support Coordination (Level 2 or Level 3 Specialist)",
    "Capital: Assistive Technology (Mobility equipment, hoists, beds)",
    "Capital: Home Modifications (Ramps, bathroom modifications)",
    "Capital: Medium-Term Accommodation (MTA) / Specialist Disability Accommodation (SDA)",
}

SUPPORT_CATEGORY_ALIASES = {
    "Core: Assistance with Daily Life (Support Workers)": "Core: Assistance with Daily Life (Support Workers, SIL)",
    "Core: Consumables": "Core: Consumables (Wound care, continence products, low-cost AT)",
    "Capacity Building: Support Coordination (Level 2 Specialist)": "Capacity Building: Support Coordination (Level 2 or Level 3 Specialist)",
    "Capacity Building: Support Coordination (Level 3 Specialist)": "Capacity Building: Support Coordination (Level 2 or Level 3 Specialist)",
    "Capital: Assistive Technology": "Capital: Assistive Technology (Mobility equipment, hoists, beds)",
    "Capital: Home Modifications": "Capital: Home Modifications (Ramps, bathroom modifications)",
    "Capital: Medium-Term Accommodation (MTA)": "Capital: Medium-Term Accommodation (MTA) / Specialist Disability Accommodation (SDA)",
}


class NavigationPlanRequest(BaseModel):
    clinical_extraction: dict[str, Any] = Field(
        min_length=1,
        description="Structured hospital discharge and clinical information.",
    )
    ndis_context: dict[str, Any] = Field(
        min_length=1,
        description="Structured participant plan and living-context information.",
    )


class RecommendedSupport(BaseModel):
    category: str
    justification: str = Field(min_length=1, max_length=1_000)

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str) -> str:
        value = SUPPORT_CATEGORY_ALIASES.get(value, value)
        if value.startswith("Capacity Building: Improved Daily Living ("):
            value = (
                "Capacity Building: Improved Daily Living "
                "(Occupational Therapy, Physiotherapy, Speech Pathology, Psychology, Dietetics, Community Nursing)"
            )
        if value not in ALLOWED_SUPPORT_CATEGORIES:
            raise ValueError("category is not an allowed NDIS support category")
        return value


class NavigationPlanResponse(BaseModel):
    practical_needs_summary: str = Field(min_length=1, max_length=3_000)
    recommended_support_categories: list[RecommendedSupport] = Field(min_length=1)
    provider_referral_summary: str = Field(min_length=1, max_length=3_000)
    call_script: str = Field(min_length=1, max_length=3_000)
    next_steps_checklist: list[str] = Field(min_length=5, max_length=7)

    @field_validator("next_steps_checklist")
    @classmethod
    def validate_checklist_items(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("checklist items cannot be blank")
        return value
