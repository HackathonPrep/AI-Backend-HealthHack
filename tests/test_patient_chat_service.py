import asyncio

import pytest

from app.core.config import Settings
from app.schemas.chat import PatientChatRequest
from app.services.patient_chat import (
    CHAT_DISCLAIMER,
    PatientChatError,
    PatientChatService,
)


class ValidChatChain:
    async def ainvoke(self, _values: dict) -> dict:
        return {
            "reply": "It sounds like personal-care support may be relevant.",
            "conversation_stage": "clarifying",
            "follow_up_questions": ["Who is currently helping you to shower?"],
            "recommendations": [],
            "urgent_action": False,
            "urgent_message": None,
            "disclaimer": "Model-provided disclaimer should be replaced.",
        }


class InvalidChatChain:
    async def ainvoke(self, _values: dict) -> dict:
        return {"reply": "Missing required fields"}


def request_with_history() -> PatientChatRequest:
    return PatientChatRequest(
        message="I cannot shower safely without help.",
        history=[
            {"role": "user", "content": "I was discharged from hospital."},
            {"role": "assistant", "content": "What has changed day to day?"},
        ],
    )


def test_history_is_converted_in_order() -> None:
    history = PatientChatService._to_langchain_history(request_with_history())

    assert [message.type for message in history] == ["human", "ai"]
    assert [message.content for message in history] == [
        "I was discharged from hospital.",
        "What has changed day to day?",
    ]


def test_urgent_message_bypasses_model_and_returns_fixed_action() -> None:
    service = PatientChatService(Settings(google_api_key="test-token"))
    response = asyncio.run(
        service.reply(PatientChatRequest(message="I have chest pain and cannot breathe."))
    )

    assert response.urgent_action is True
    assert "000" in response.urgent_message
    assert response.recommendations == []


def test_chat_response_replaces_model_disclaimer() -> None:
    service = PatientChatService(Settings(google_api_key="test-token"))
    service._chain = lambda: ValidChatChain()  # type: ignore[method-assign]

    response = asyncio.run(service.reply(request_with_history()))

    assert response.disclaimer == CHAT_DISCLAIMER


def test_invalid_model_response_is_exposed_as_safe_error() -> None:
    service = PatientChatService(Settings(google_api_key="test-token"))
    service._chain = lambda: InvalidChatChain()  # type: ignore[method-assign]

    with pytest.raises(PatientChatError, match="invalid response"):
        asyncio.run(service.reply(PatientChatRequest(message="I need daily support.")))
