from supabase import Client, create_client

from app.core.config import Settings


class RecordRepository:
    def __init__(self, settings: Settings) -> None:
        if not settings.supabase_enabled:
            raise RuntimeError("Supabase persistence is not configured.")
        self.client: Client = create_client(settings.supabase_url, settings.supabase_secret_key)

    def create(self, owner_id: str, record_type: str, source_filename: str | None, clinical_extraction: dict, result: dict) -> dict:
        return self.client.table("participant_records").insert({
            "owner_id": owner_id, "record_type": record_type,
            "source_filename": source_filename, "clinical_extraction": clinical_extraction,
            "result": result,
        }).execute().data[0]

    def list(self, owner_id: str) -> list[dict]:
        return self.client.table("participant_records").select("*").eq("owner_id", owner_id).order("created_at", desc=True).execute().data

    def get(self, owner_id: str, record_id: str) -> dict | None:
        data = self.client.table("participant_records").select("*").eq("id", record_id).eq("owner_id", owner_id).limit(1).execute().data
        return data[0] if data else None
