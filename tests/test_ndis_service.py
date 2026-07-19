import asyncio

import pytest

from app.core.config import Settings
from app.schemas.ndis import NavigationPlanRequest
from app.services.ndis_navigation import NdisNavigationError, NdisNavigationService


class InvalidPlanChain:
    async def ainvoke(self, _values: dict) -> dict:
        return {"not": "the expected response"}


def test_model_provider_is_parsed_from_environment_value() -> None:
    settings = Settings(huggingface_model="organisation/model:novita")

    assert settings.huggingface_model_and_provider == ("organisation/model", "novita")


def test_invalid_model_response_is_exposed_as_safe_error() -> None:
    service = NdisNavigationService(Settings(huggingfacehub_api_token="test-token"))
    service._chain = lambda: InvalidPlanChain()  # type: ignore[method-assign]
    request = NavigationPlanRequest(
        clinical_extraction={"diagnosis": "Stroke"},
        ndis_context={"has_active_plan": True},
    )

    with pytest.raises(NdisNavigationError, match="invalid response"):
        asyncio.run(service.create_plan(request))
