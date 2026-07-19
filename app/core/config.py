from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration sourced from the environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "AI HealthHack Backend"
    environment: str = "development"
    allowed_origins: str = "http://localhost:5173,http://localhost:4173"
    huggingfacehub_api_token: str | None = Field(default=None, repr=False)
    huggingface_model: str = "google/gemma-4-31B-it:novita"
    ndis_request_timeout_seconds: float = 90.0
    patient_chat_timeout_seconds: float = 120.0
    chat_history_limit: int = 20
    chat_message_max_characters: int = 4_000
    document_max_upload_bytes: int = 10 * 1024 * 1024
    document_request_timeout_seconds: float = 120.0
    action_pack_request_timeout_seconds: float = 120.0
    supabase_url: str | None = None
    supabase_secret_key: str | None = Field(default=None, repr=False)
    supabase_jwks_url: str | None = None
    whisper_model: str = "tiny"
    whisper_final_model: str = "small"

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def huggingface_model_and_provider(self) -> tuple[str, str | None]:
        """Split `model:provider`, preserving Hugging Face model namespaces."""
        model, separator, provider = self.huggingface_model.rpartition(":")
        if separator and provider:
            return model, provider
        return self.huggingface_model, None

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_secret_key and self.supabase_jwks_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
