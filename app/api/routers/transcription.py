import json
import time

from fastapi import APIRouter, WebSocket

from app.services.transcription import TranscriptionService

router = APIRouter(tags=["transcription"])
DEBUG_LOG_PATH = "/home/luki/Documents/GitHub/AiHealthHack/.cursor/debug-d4bdc1.log"


def _debug_log(location: str, message: str, data: dict, hypothesis_id: str) -> None:
    payload = {
        "sessionId": "d4bdc1",
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as log:
        log.write(json.dumps(payload) + "\n")


@router.websocket("/ws/transcribe")
async def transcribe_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    # region agent log
    _debug_log("transcription.py:websocket", "Backend accepted transcription WebSocket", {}, "H2,H3")
    # endregion
    service: TranscriptionService = websocket.app.state.transcription_service
    try:
        await service.run_session(websocket)
    except Exception as error:
        # region agent log
        _debug_log(
            "transcription.py:websocket",
            "Unhandled transcription WebSocket failure",
            {"errorType": type(error).__name__, "error": str(error)},
            "H3,H5",
        )
        # endregion
        raise
