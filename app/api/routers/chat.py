from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.chat import PatientChatRequest, PatientChatResponse
from app.services.patient_chat import (
    PatientChatError,
    PatientChatInputError,
    PatientChatService,
)

router = APIRouter(prefix="/api/v1/patient-chat", tags=["patient chat"])


@router.post("/message", response_model=PatientChatResponse)
async def send_patient_message(
    payload: PatientChatRequest, request: Request
) -> PatientChatResponse:
    service: PatientChatService = request.app.state.patient_chat_service
    try:
        return await service.reply(payload)
    except PatientChatInputError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error
    except PatientChatError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)
        ) from error
