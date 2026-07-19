import pytest

from app.core.config import Settings
from app.services.document_ingestion import (
    DocumentIngestionService,
    DocumentInputError,
)
from app.services.ndis_navigation import NdisNavigationService


def service() -> DocumentIngestionService:
    settings = Settings(huggingfacehub_api_token="test-token")
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
