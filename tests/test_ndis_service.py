import asyncio

import pytest

from app.core.config import Settings
from app.core.llm import build_chat_model, is_provider_rate_limited
from app.schemas.ndis import NavigationPlanRequest
from app.services.ndis_navigation import NdisNavigationError, NdisNavigationService


class InvalidPlanChain:
    async def ainvoke(self, _values: dict) -> dict:
        return {"not": "the expected response"}


def test_huggingface_enabled_when_token_present() -> None:
    assert Settings(hf_token="test-token").huggingface_enabled is True
    assert Settings(hf_token=None).huggingface_enabled is False


def test_gemma_model_and_provider_are_configured() -> None:
    model = build_chat_model(
        Settings(
            hf_token="test-token",
            hd_model="google/gemma-4-26B-A4B-it:novita",
        ),
        temperature=0.0,
        max_output_tokens=16,
    )

    assert model.model_id == "google/gemma-4-26B-A4B-it"
    assert model.llm.provider == "novita"


def test_provider_rate_limit_is_detected() -> None:
    assert is_provider_rate_limited(RuntimeError("429 Too Many Requests"))
    assert not is_provider_rate_limited(RuntimeError("invalid JSON response"))


def test_invalid_model_response_is_exposed_as_safe_error() -> None:
    service = NdisNavigationService(Settings(hf_token="test-token"))
    service._chain = lambda: InvalidPlanChain()  # type: ignore[method-assign]
    request = NavigationPlanRequest(
        clinical_extraction={"diagnosis": "Stroke"},
        ndis_context={"has_active_plan": True},
    )

    with pytest.raises(NdisNavigationError, match="invalid response"):
        asyncio.run(service.create_plan(request))
