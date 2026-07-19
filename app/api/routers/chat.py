from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.auth import get_current_user_id
from app.schemas.chat import PatientChatRequest, PatientChatResponse
from app.services.records import RecordRepository
from app.services.patient_chat import (
    PatientChatError,
    PatientChatInputError,
    PatientChatService,
)

router = APIRouter(prefix="/api/v1/patient-chat", tags=["patient chat"])


@router.post("/message", response_model=PatientChatResponse)
async def send_patient_message(
    payload: PatientChatRequest,
    request: Request,
    _user_id: str = Depends(get_current_user_id),
) -> PatientChatResponse:
    service: PatientChatService = request.app.state.patient_chat_service
    try:
        response = await service.reply(payload)
        repository: RecordRepository | None = request.app.state.record_repository
        if repository is not None:
            session_id = payload.session_id or str(uuid4())
            repository.save_chat("patient", payload.message, session_id)
            repository.save_chat("ai", response.reply, session_id)
        return response
    except PatientChatInputError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error
    except PatientChatError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)
        ) from error
