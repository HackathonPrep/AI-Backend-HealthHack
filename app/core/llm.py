"""Shared Hugging Face chat-model factory for the Gemma services."""

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub.utils import HfHubHTTPError

from app.core.config import Settings


class LlmNotConfiguredError(RuntimeError):
    """Raised when the Hugging Face access token is missing."""


def is_provider_rate_limited(error: BaseException) -> bool:
    """Identify the provider's quota/rate-limit response without exposing it."""
    return "429" in str(error) or "too many requests" in str(error).lower()


def retry_transient_provider_errors(chain):
    """Retry a short-lived Hugging Face/Novita overload before failing a request."""
    # Unit-test doubles deliberately implement only ``ainvoke``.
    if not hasattr(chain, "with_retry"):
        return chain
    return chain.with_retry(
        retry_if_exception_type=(HfHubHTTPError,),
        wait_exponential_jitter=True,
        stop_after_attempt=3,
    )


def _model_and_provider(model_spec: str) -> tuple[str, str | None]:
    """Split `organisation/model:provider` without losing model namespaces."""
    model, separator, provider = model_spec.strip().rpartition(":")
    return (model, provider) if separator and provider else (model_spec.strip(), None)


def build_chat_model(
    settings: Settings,
    *,
    temperature: float,
    max_output_tokens: int,
) -> ChatHuggingFace:
    if not settings.hf_token:
        raise LlmNotConfiguredError(
            "AI features are unavailable because HF_TOKEN is not configured."
        )

    model, provider = _model_and_provider(settings.hd_model)
    endpoint_options = {
        "repo_id": model,
        "task": "text-generation",
        "huggingfacehub_api_token": settings.hf_token,
        "temperature": temperature,
        "max_new_tokens": max_output_tokens,
        "do_sample": temperature > 0,
        "timeout": 120,
    }
    if provider:
        endpoint_options["provider"] = provider

    return ChatHuggingFace(
        llm=HuggingFaceEndpoint(**endpoint_options),
        model_id=model,
    )
