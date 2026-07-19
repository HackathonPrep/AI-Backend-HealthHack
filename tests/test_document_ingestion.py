import pytest

from app.core.config import Settings
from app.services.document_ingestion import (
    DocumentIngestionError,
    DocumentIngestionService,
    DocumentInputError,
)
from app.services.ndis_navigation import NdisNavigationError, NdisNavigationService


def service() -> DocumentIngestionService:
    settings = Settings(hf_token="test-token")
    return DocumentIngestionService(settings, NdisNavigationService(settings))


def test_document_type_accepts_pdf_and_images() -> None:
    ingestion = service()

    assert ingestion._file_type("summary.pdf", None) == "application/pdf"
    assert ingestion._file_type("scan.jpeg", None) == "image/jpeg"
    assert ingestion._file_type("scan.png", None) == "image/png"


def test_document_validation_rejects_unknown_or_empty_files() -> None:
    ingestion = service()

    with pytest.raises(DocumentInputError, match="empty"):
        ingestion._validate_file("summary.pdf", "application/pdf", b"")

    with pytest.raises(DocumentInputError, match="Only PDF"):
        ingestion._validate_file("summary.docx", "application/octet-stream", b"content")


class ExtractionChain:
    async def ainvoke(self, _values: dict) -> dict:
        return {"diagnosis_reason": "Fictional discharge summary"}


class UnavailableNavigationService:
    async def create_plan(self, _request: object) -> object:
        raise NdisNavigationError("The NDIS planning model is unavailable. Please try again shortly.")


def test_document_plan_preserves_the_safe_planning_error() -> None:
    """The UI must not receive the old misleading generic service error."""
    import asyncio

    ingestion = DocumentIngestionService(
        Settings(hf_token="test-token"), UnavailableNavigationService()  # type: ignore[arg-type]
    )
    ingestion._chain = lambda: ExtractionChain()  # type: ignore[method-assign]

    with pytest.raises(DocumentIngestionError, match="NDIS planning model is unavailable"):
        asyncio.run(
            ingestion.create_plan_from_document(
                "summary.png", "image/png", b"fake-image", {"urgency_level": "High"}
            )
        )
