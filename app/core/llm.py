"""Shared Google Gemini chat-model factory for LangChain services."""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import Settings

logger = logging.getLogger(__name__)

_LEGACY_MODEL_ALIASES = {
    # This alias now points to a preview model and may not be enabled for an
    # existing API project. Preserve a working deployment that still has the
    # previous example value in its environment configuration.
    "gemini-flash-latest": "gemini-3.5-flash",
    "gemini-2.5-flash": "gemini-3.5-flash",
}


class LlmNotConfiguredError(RuntimeError):
    """Raised when GOOGLE_API_KEY is missing."""


def build_chat_model(
    settings: Settings,
    *,
    temperature: float,
    max_output_tokens: int,
) -> ChatGoogleGenerativeAI:
    if not settings.google_api_key:
        raise LlmNotConfiguredError(
            "AI features are unavailable because GOOGLE_API_KEY is not configured."
        )
    configured_model = settings.google_model.strip()
    model = _LEGACY_MODEL_ALIASES.get(configured_model, configured_model)
    if model != configured_model:
        logger.warning(
            "GOOGLE_MODEL=%s has been replaced with stable model %s.",
            configured_model,
            model,
        )

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        # Every current caller parses a JSON response. Asking Gemini for JSON at
        # the API layer prevents markdown fences and prose from intermittently
        # breaking the LangChain JSON parser.
        response_mime_type="application/json",
        # Gemini 3.5 allocates output tokens to thinking by default. These
        # extraction/planning calls are tightly structured, so minimal thinking
        # leaves sufficient tokens for the complete JSON response.
        thinking_level="minimal",
    )
