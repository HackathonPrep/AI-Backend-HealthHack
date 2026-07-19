import pytest
from pydantic import ValidationError

from app.schemas.profile import ProfileApprovalRequest


def valid_approval() -> dict:
    return {
        "sections": [
            {
                "id": "about",
                "title": "About the participant",
                "status": "confirmed",
                "items": ["Location: Parramatta, NSW"],
            },
            {
                "id": "supports",
                "title": "Recommended supports",
                "status": "removed",
                "items": ["Support worker — high priority"],
            },
        ],
        "follow_up_answers": [
            {"question": "Do you need help at home?", "answer": "Yes, with showering."}
        ],
        "consents": ["reviewed", "matching", "no_automatic_document_sharing"],
    }


def test_profile_approval_requires_all_consents() -> None:
    payload = valid_approval()
    payload["consents"] = ["reviewed", "matching", "other"]

    with pytest.raises(ValidationError, match="required consents"):
        ProfileApprovalRequest.model_validate(payload)


def test_profile_approval_is_structured_and_normalised() -> None:
    approval = ProfileApprovalRequest.model_validate(valid_approval())

    assert approval.sections[0].items == ["Location: Parramatta, NSW"]
    assert approval.follow_up_answers[0].answer == "Yes, with showering."


def test_profile_approval_rejects_duplicate_section_ids() -> None:
    payload = valid_approval()
    payload["sections"].append({
        "id": "about",
        "title": "Duplicate",
        "status": "confirmed",
        "items": [],
    })

    with pytest.raises(ValidationError, match="section ids"):
        ProfileApprovalRequest.model_validate(payload)
