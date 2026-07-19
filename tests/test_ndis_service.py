import asyncio

import pytest

from app.core.config import Settings
from app.core.llm import build_chat_model
from app.schemas.ndis import NavigationPlanRequest
from app.services.ndis_navigation import NdisNavigationError, NdisNavigationService


class InvalidPlanChain:
    async def ainvoke(self, _values: dict) -> dict:
        return {"not": "the expected response"}


def test_google_enabled_when_api_key_present() -> None:
    assert Settings(google_api_key="test-token").google_enabled is True
    assert Settings(google_api_key=None).google_enabled is False


def test_legacy_google_model_alias_uses_the_stable_model() -> None:
    model = build_chat_model(
        Settings(google_api_key="test-token", google_model="gemini-flash-latest"),
        temperature=0.0,
        max_output_tokens=16,
    )

    assert model.model == "gemini-3.5-flash"
    assert model.thinking_level == "minimal"


def test_invalid_model_response_is_exposed_as_safe_error() -> None:
    service = NdisNavigationService(Settings(google_api_key="test-token"))
    service._chain = lambda: InvalidPlanChain()  # type: ignore[method-assign]
    request = NavigationPlanRequest(
        clinical_extraction={"diagnosis": "Stroke"},
        ndis_context={"has_active_plan": True},
    )

    with pytest.raises(NdisNavigationError, match="invalid response"):
        asyncio.run(service.create_plan(request))
