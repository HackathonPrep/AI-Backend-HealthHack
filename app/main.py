import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import chat, demo, ndis, transcription
from app.core.config import get_settings
from app.services.document_ingestion import DocumentIngestionService
from app.services.ndis_action_pack import NdisActionPackService
from app.services.ndis_navigation import NdisNavigationService
from app.services.patient_chat import PatientChatService
from app.services.transcription import TranscriptionService
from app.services.records import RecordRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.ndis_navigation_service = NdisNavigationService(settings)
    app.state.ndis_action_pack_service = NdisActionPackService(settings)
    app.state.document_ingestion_service = DocumentIngestionService(
        settings, app.state.ndis_navigation_service
    )
    app.state.patient_chat_service = PatientChatService(settings)
    app.state.transcription_service = TranscriptionService(settings)
    app.state.record_repository = RecordRepository(settings) if settings.supabase_enabled else None
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(ndis.router)
    app.include_router(chat.router)
    app.include_router(demo.router)
    app.include_router(transcription.router)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "environment": settings.environment}

    return app


app = create_app()
