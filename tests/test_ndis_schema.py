import pytest
from pydantic import ValidationError

from app.schemas.ndis import (
    ALLOWED_SUPPORT_CATEGORIES,
    NavigationPlanRequest,
    NavigationPlanResponse,
)


def valid_plan() -> dict:
    return {
        "practical_needs_summary": "The participant needs help at home.",
        "recommended_support_categories": [
            {
                "category": next(iter(ALLOWED_SUPPORT_CATEGORIES)),
                "justification": "It addresses a documented disability-related need.",
            }
        ],
        "provider_referral_summary": "Clinical referral summary.",
        "call_script": "Please start an urgent review.",
        "next_steps_checklist": [
            "Gather the discharge summary.",
            "Contact the NDIA.",
            "Contact the plan manager.",
            "Arrange immediate safety support.",
            "Book an assessment.",
        ],
    }


def test_plan_request_requires_both_context_objects() -> None:
    with pytest.raises(ValidationError):
        NavigationPlanRequest(clinical_extraction={})


def test_plan_response_rejects_unknown_support_category() -> None:
    plan = valid_plan()
    plan["recommended_support_categories"][0]["category"] = "Unknown support"

    with pytest.raises(ValidationError):
        NavigationPlanResponse.model_validate(plan)


def test_plan_response_accepts_expected_contract() -> None:
    response = NavigationPlanResponse.model_validate(valid_plan())

    assert len(response.next_steps_checklist) == 5
