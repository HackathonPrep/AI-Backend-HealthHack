from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration sourced from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", populate_by_name=True
    )

    app_name: str = "AI HealthHack Backend"
    environment: str = "development"
    allowed_origins: str = (
        "http://localhost:5173,http://localhost:4173,"
        "http://127.0.0.1:5173,http://127.0.0.1:4173"
    )
    hf_token: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
    )
    hd_model: str = Field(
        default="google/gemma-4-26B-A4B-it:novita",
        validation_alias=AliasChoices("HD_MODEL", "HUGGINGFACE_MODEL"),
    )
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
    demo_patient_id: str | None = None
    whisper_model: str = "tiny"
    whisper_final_model: str = "small"
    ai_trace_enabled: bool = True

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_secret_key and self.supabase_jwks_url)

    @property
    def huggingface_enabled(self) -> bool:
        return bool(self.hf_token)


@lru_cache
def get_settings() -> Settings:
    return Settings()
