from app.schemas.document import ClinicalExtraction
from app.services.document_ingestion import resolve_ndis_pathway


def test_new_disability_discharge_maps_to_access_request() -> None:
    extraction = ClinicalExtraction(
        ndis_status=(
            "Ms Turner was not an NDIS participant prior to admission and has been "
            "advised to commence an NDIS access request."
        ),
        discharge_supports="Urgent support coordination and daily support workers.",
    )

    pathway, context = resolve_ndis_pathway(extraction, {"has_active_plan": True})

    assert pathway == "ndis_access_request"
    assert context["has_active_plan"] is False
    assert context["pathway"] == "ndis_access_request"
    assert "support coordination" in context["documented_discharge_supports"].lower()


def test_existing_participant_maps_to_plan_review() -> None:
    extraction = ClinicalExtraction(
        ndis_status="Existing participant with an active plan requiring urgent s48 review."
    )

    pathway, context = resolve_ndis_pathway(extraction, {})

    assert pathway == "plan_review"
    assert context["has_active_plan"] is True
