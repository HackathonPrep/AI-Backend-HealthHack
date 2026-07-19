from fastapi import APIRouter, WebSocket

from app.services.transcription import TranscriptionService

router = APIRouter(tags=["transcription"])


@router.websocket("/ws/transcribe")
async def transcribe_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    service: TranscriptionService = websocket.app.state.transcription_service
    await service.run_session(websocket)
