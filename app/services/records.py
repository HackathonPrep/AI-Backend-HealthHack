import logging
from collections import defaultdict
from datetime import datetime, timezone
from uuid import uuid4

from postgrest.exceptions import APIError
from supabase import Client, create_client

from app.core.config import Settings
from app.schemas.profile import ProfileApprovalRequest, ProfileSectionStatus

logger = logging.getLogger(__name__)

_PROFILE_APPROVALS_TABLE = "participant_profile_approvals"
_MISSING_TABLE_CODE = "PGRST205"


def _profile_approvals_table_is_missing(error: Exception) -> bool:
    """Detect PostgREST's schema-cache response for an unapplied migration."""
    return getattr(error, "code", None) == _MISSING_TABLE_CODE


class RecordRepository:
    """Repository for the live CareMatch Supabase domain schema."""

    def __init__(self, settings: Settings) -> None:
        if not settings.supabase_enabled:
            raise RuntimeError("Supabase persistence is not configured.")
        self.client: Client = create_client(
            settings.supabase_url, settings.supabase_secret_key
        )
        self.configured_demo_patient_id = settings.demo_patient_id

    def demo_patient_id(self) -> str:
        if self.configured_demo_patient_id:
            data = (
                self.client.table("patients")
                .select("id")
                .eq("id", self.configured_demo_patient_id)
                .limit(1)
                .execute()
                .data
            )
            if data:
                return data[0]["id"]
        data = (
            self.client.table("patients")
            .select("id")
            .order("created_at")
            .limit(1)
            .execute()
            .data
        )
        if not data:
            raise RuntimeError("No demo patient exists in Supabase.")
        return data[0]["id"]

    def profile(self) -> dict:
        patient_id = self.demo_patient_id()
        patient = (
            self.client.table("patients")
            .select("*")
            .eq("id", patient_id)
            .single()
            .execute()
            .data
        )
        documents = (
            self.client.table("patient_documents")
            .select("*")
            .eq("patient_id", patient_id)
            .order("uploaded_at", desc=True)
            .execute()
            .data
        )
        extractions = (
            self.client.table("ai_extractions")
            .select("*")
            .eq("patient_id", patient_id)
            .order("created_at", desc=True)
            .execute()
            .data
        )
        needs = (
            self.client.table("patient_needs")
            .select("*")
            .eq("patient_id", patient_id)
            .order("created_at", desc=True)
            .execute()
            .data
        )
        catalog = {
            item["id"]: item
            for item in self.client.table("service_catalog")
            .select("*")
            .execute()
            .data
        }
        for need in needs:
            need["service"] = catalog.get(need.get("catalog_item_id"))
        return {
            "patient": patient,
            "documents": documents,
            "extractions": extractions,
            "needs": needs,
            "latest_approval": self.latest_profile_approval(patient_id),
        }

    def latest_profile_approval(self, patient_id: str | None = None) -> dict | None:
        """Return the newest immutable participant-approved profile snapshot."""
        try:
            rows = (
                self.client.table(_PROFILE_APPROVALS_TABLE)
                .select("id,approved_profile,follow_up_answers,consents,created_at")
                .eq("patient_id", patient_id or self.demo_patient_id())
                .order("created_at", desc=True)
                .limit(1)
                .execute()
                .data
            )
        except APIError as error:
            if not _profile_approvals_table_is_missing(error):
                raise
            logger.warning(
                "%s is not available yet; no saved profile approval can be loaded.",
                _PROFILE_APPROVALS_TABLE,
            )
            return None
        return rows[0] if rows else None

    def save_profile_approval(self, approval: ProfileApprovalRequest) -> dict:
        """Append, rather than mutate, a profile approval for the demo participant."""
        patient_id = self.demo_patient_id()
        sections = [section.model_dump(mode="json") for section in approval.sections]
        support_texts = [
            item
            for section in sections
            if section["id"] == "supports"
            and section["status"] == ProfileSectionStatus.CONFIRMED.value
            for item in section["items"]
        ]
        catalog = (
            self.client.table("service_catalog")
            .select("id,name,category,active")
            .eq("active", True)
            .execute()
            .data
        )
        approved_profile = {
            "sections": sections,
            "support_catalog_item_ids": [
                item["id"] for item in self._catalog_matches(" | ".join(support_texts), catalog)
            ],
        }
        record = {
            "patient_id": patient_id,
            "approved_profile": approved_profile,
            "follow_up_answers": [answer.model_dump() for answer in approval.follow_up_answers],
            "consents": approval.consents,
        }
        try:
            result = (
                self.client.table(_PROFILE_APPROVALS_TABLE)
                .insert(record)
                .execute()
            )
            if not result.data:
                raise RuntimeError("profile approval insert returned no rows")
            return result.data[0]
        except APIError as error:
            if not _profile_approvals_table_is_missing(error):
                raise
            # Keep the demo journey usable until its one-time database migration
            # is applied. The response is deliberately a valid snapshot but is
            # not persisted, and startup/profile reads remain safe as well.
            logger.warning(
                "%s is missing; returning a non-persisted profile approval. "
                "Apply supabase/migrations/20260719_participant_profile_approvals.sql "
                "to enable persistence.",
                _PROFILE_APPROVALS_TABLE,
            )
            return {
                "id": f"transient-{uuid4()}",
                "created_at": datetime.now(timezone.utc).isoformat(),
                **record,
            }

    def history(self) -> list[dict]:
        patient_id = self.demo_patient_id()
        documents = (
            self.client.table("patient_documents")
            .select("*")
            .eq("patient_id", patient_id)
            .order("uploaded_at", desc=True)
            .execute()
            .data
        )
        extractions = (
            self.client.table("ai_extractions")
            .select("*")
            .eq("patient_id", patient_id)
            .execute()
            .data
        )
        by_document = {
            row.get("document_id"): row for row in extractions if row.get("document_id")
        }
        return [
            {**document, "extraction": by_document.get(document["id"])}
            for document in documents
        ]

    def get_history_item(self, record_id: str) -> dict | None:
        return next(
            (record for record in self.history() if record["id"] == record_id), None
        )

    def providers(self, limit: int = 20) -> list[dict]:
        patient_id = self.demo_patient_id()
        patient = (
            self.client.table("patients")
            .select("suburb,state")
            .eq("id", patient_id)
            .single()
            .execute()
            .data
        )
        needs = (
            self.client.table("patient_needs")
            .select("catalog_item_id")
            .eq("patient_id", patient_id)
            .execute()
            .data
        )
        approval = self.latest_profile_approval(patient_id)
        approved_ids = set(
            (approval or {}).get("approved_profile", {}).get("support_catalog_item_ids", [])
        )
        needed_ids = approved_ids or {row["catalog_item_id"] for row in needs}
        catalog = {
            row["id"]: row
            for row in self.client.table("service_catalog")
            .select("id,name,category")
            .execute()
            .data
        }
        providers = (
            self.client.table("providers")
            .select("*")
            .eq("verified", True)
            .order("accepting_clients", desc=True)
            .limit(100)
            .execute()
            .data
        )
        provider_ids = [provider["id"] for provider in providers]
        offerings = (
            self.client.table("provider_offerings")
            .select("*")
            .in_("provider_id", provider_ids)
            .execute()
            .data
        )
        languages = (
            self.client.table("provider_languages")
            .select("provider_id,language")
            .in_("provider_id", provider_ids)
            .execute()
            .data
        )
        conditions = (
            self.client.table("provider_conditions")
            .select("provider_id,condition_name")
            .in_("provider_id", provider_ids)
            .execute()
            .data
        )
        offerings_by_provider: dict[str, list[dict]] = defaultdict(list)
        languages_by_provider: dict[str, list[str]] = defaultdict(list)
        conditions_by_provider: dict[str, list[str]] = defaultdict(list)
        for row in offerings:
            offerings_by_provider[row["provider_id"]].append(row)
        for row in languages:
            languages_by_provider[row["provider_id"]].append(row["language"])
        for row in conditions:
            conditions_by_provider[row["provider_id"]].append(row["condition_name"])

        matches = []
        for provider in providers:
            provider_offerings = offerings_by_provider[provider["id"]]
            offered_ids = {row["catalog_item_id"] for row in provider_offerings}
            matched_ids = needed_ids & offered_ids
            score = 45
            reasons = []
            if provider.get("accepting_clients"):
                score += 20
                reasons.append("Currently accepting new clients")
            if (
                provider.get("suburb") == patient.get("suburb")
                and provider.get("state") == patient.get("state")
            ):
                score += 20
                reasons.append(f"Located in {patient.get('suburb')}")
            if needed_ids:
                support_score = round(15 * len(matched_ids) / len(needed_ids))
                score += support_score
            matched_names = [
                catalog[item_id]["name"]
                for item_id in matched_ids
                if item_id in catalog
            ]
            if matched_names:
                reasons.append(f"Offers {', '.join(matched_names)}")
            service_names = [
                catalog[row["catalog_item_id"]]["name"]
                for row in provider_offerings
                if row.get("catalog_item_id") in catalog
            ]
            wait_days = provider.get("wait_time_days") or 0
            matches.append(
                {
                    "id": provider["id"],
                    "name": provider["business_name"],
                    "score": min(score, 99),
                    "location": f"{provider.get('suburb')}, {provider.get('state')}",
                    "status": (
                        "Available"
                        if provider.get("accepting_clients")
                        else "Waitlist"
                    ),
                    "response": (
                        "Within one day"
                        if wait_days <= 1
                        else f"Within {wait_days} days"
                    ),
                    "services": service_names[:6],
                    "languages": languages_by_provider[provider["id"]] or ["English"],
                    "conditions": conditions_by_provider[provider["id"]],
                    "reasons": reasons
                    or ["Verified provider in the CareMatch directory"],
                }
            )
        return sorted(matches, key=lambda row: row["score"], reverse=True)[:limit]

    def referrals(self, patient_only: bool = False, limit: int = 50) -> list[dict]:
        query = self.client.table("referrals").select("*")
        if patient_only:
            query = query.eq("patient_id", self.demo_patient_id())
        rows = query.order("created_at", desc=True).limit(limit).execute().data
        if not rows:
            return []
        patient_ids = list({row["patient_id"] for row in rows})
        provider_ids = list({row["provider_id"] for row in rows})
        patients = {
            row["id"]: row
            for row in self.client.table("patients")
            .select("id,first_name,last_name,suburb,state")
            .in_("id", patient_ids)
            .execute()
            .data
        }
        providers = {
            row["id"]: row
            for row in self.client.table("providers")
            .select("id,business_name,suburb,state")
            .in_("id", provider_ids)
            .execute()
            .data
        }
        return [
            {
                **row,
                "patient": patients.get(row["patient_id"]),
                "provider": providers.get(row["provider_id"]),
            }
            for row in rows
        ]

    def create_referral(self, provider_id: str, summary: str | None = None) -> dict:
        patient_id = self.demo_patient_id()
        payload = {
            "patient_id": patient_id,
            "provider_id": provider_id,
            "status": "pending",
            "referral_summary": summary
            or "Referral created from the CareMatch AI demo profile.",
        }
        return self.client.table("referrals").insert(payload).execute().data[0]

    def update_referral(self, referral_id: str, status: str) -> dict | None:
        data = (
            self.client.table("referrals")
            .update({"status": status})
            .eq("id", referral_id)
            .execute()
            .data
        )
        return data[0] if data else None

    def save_chat(
        self, sender: str, message: str, session_id: str | None = None
    ) -> dict:
        return (
            self.client.table("ai_chat_history")
            .insert(
                {
                    "patient_id": self.demo_patient_id(),
                    "session_id": session_id or str(uuid4()),
                    "sender": sender,
                    "message": message,
                }
            )
            .execute()
            .data[0]
        )

    def list_chat_history(self, session_id: str | None = None) -> list[dict]:
        query = (
            self.client.table("ai_chat_history")
            .select("id,session_id,sender,message,created_at")
            .eq("patient_id", self.demo_patient_id())
            .order("created_at")
        )
        if session_id:
            query = query.eq("session_id", session_id)
        rows = query.execute().data or []
        return [
            {
                "id": row["id"],
                "session_id": row.get("session_id"),
                "sender": row.get("sender"),
                "role": "user" if row.get("sender") == "patient" else "assistant",
                "message": row.get("message") or "",
                "created_at": row.get("created_at"),
            }
            for row in rows
        ]

    @staticmethod
    def _document_type_for(_record_type: str) -> str:
        # Live schema check constraint currently allows discharge_summary only.
        return "discharge_summary"

    @staticmethod
    def _catalog_matches(support_text: str, catalog: list[dict]) -> list[dict]:
        """Map LangChain support wording onto service_catalog rows by keyword."""
        text = support_text.lower()
        keywords = {
            "occupational therapy": ("occupational", "ot "),
            "physiotherapy": ("physiotherapy", "physio"),
            "speech pathology": ("speech",),
            "psychology": ("psychology", "psycholog"),
            "support worker": ("support worker", "daily life", "personal care", "sil"),
            "community participation": ("community participation", "social", "community access"),
            "community nursing": ("nursing",),
            "accessible transport": ("transport",),
            "support coordination": ("support coordination", "coordinator"),
            "sil": ("sil", "supported independent living"),
            "sda": ("sda", "specialist disability accommodation"),
            "home modification": ("home modification", "ramp", "bathroom modification"),
            "manual wheelchair": ("manual wheelchair",),
            "power wheelchair": ("power wheelchair", "powered wheelchair"),
            "pressure cushion": ("pressure cushion", "pressure care"),
            "hospital bed": ("hospital bed", "bed rail"),
            "slide board": ("slide board",),
            "grab rails": ("grab rail", "grab rails", "handrail"),
            "transfer belt": ("transfer belt",),
            "walker": ("walker", "walking frame"),
            "shower chair": ("shower chair", "shower stool"),
            "commode": ("commode",),
            "continence consumables": ("continence",),
            "catheter supplies": ("catheter",),
        }
        matched: list[dict] = []
        seen: set[str] = set()
        for item in catalog:
            name = (item.get("name") or "").lower()
            aliases = keywords.get(name, (name,))
            if any(alias.strip() and alias in text for alias in aliases) or name in text:
                if item["id"] not in seen:
                    matched.append(item)
                    seen.add(item["id"])
        return matched

    def _sync_needs(self, patient_id: str, support_texts: list[str]) -> None:
        catalog = (
            self.client.table("service_catalog")
            .select("id,name,category,active")
            .eq("active", True)
            .execute()
            .data
        )
        combined = " | ".join(support_texts)
        matched = self._catalog_matches(combined, catalog)
        self.client.table("patient_needs").delete().eq("patient_id", patient_id).execute()
        if not matched:
            return
        rows = [
            {
                "patient_id": patient_id,
                "catalog_item_id": item["id"],
                "priority": "high" if index < 2 else "medium",
                "frequency": "As needed",
                "quantity": 1,
                "status": "recommended",
                "notes": f"Derived from uploaded document AI plan: {item['name']}",
            }
            for index, item in enumerate(matched[:8])
        ]
        self.client.table("patient_needs").insert(rows).execute()

    def create(
        self,
        _owner_id: str,
        record_type: str,
        source_filename: str | None,
        clinical_extraction: dict,
        result: dict,
    ) -> dict:
        """Persist a generated result into patient_documents + ai_extractions."""
        patient_id = self.demo_patient_id()
        document = (
            self.client.table("patient_documents")
            .insert(
                {
                    "patient_id": patient_id,
                    "document_name": source_filename or "Uploaded document",
                    "document_type": self._document_type_for(record_type),
                    "storage_path": f"processed://{source_filename or 'document'}",
                }
            )
            .execute()
            .data[0]
        )
        plan = result.get("plan") or result.get("action_pack") or {}
        support_items = plan.get("recommended_support_categories") or plan.get(
            "provider_service_categories", []
        )
        supports = []
        for item in support_items:
            supports.append(item.get("category", "") if isinstance(item, dict) else item)
        red_flags = clinical_extraction.get("red_flags")
        payload = {
            "patient_id": patient_id,
            "document_id": document["id"],
            "diagnosis": clinical_extraction.get("diagnosis_reason"),
            "mobility": clinical_extraction.get("mobility_status"),
            "transfers": clinical_extraction.get("transfer_status"),
            "cognition": clinical_extraction.get("cognition_mental_health"),
            "mental_health": clinical_extraction.get("cognition_mental_health"),
            "living_situation": clinical_extraction.get("living_situation"),
            "equipment_needed": clinical_extraction.get("equipment_needs"),
            "support_needed": ", ".join(filter(None, supports))
            or clinical_extraction.get("discharge_supports"),
            "risk_level": "high" if red_flags else "low",
            "follow_up": clinical_extraction.get("follow_up_requirements"),
            "ai_summary": plan.get("practical_needs_summary")
            or plan.get("provider_referral_summary"),
            "confidence_score": 85,
        }
        extraction = (
            self.client.table("ai_extractions").insert(payload).execute().data[0]
        )
        need_texts = list(filter(None, supports))
        if plan.get("practical_needs_summary"):
            need_texts.append(plan["practical_needs_summary"])
        if clinical_extraction.get("equipment_needs"):
            need_texts.append(clinical_extraction["equipment_needs"])
        try:
            self._sync_needs(patient_id, need_texts)
        except Exception:
            logger.exception("Failed to sync patient_needs from upload result")
        return {**document, "extraction": extraction}

    # Backwards-compatible aliases used by the existing NDIS router.
    def list(self, _owner_id: str) -> list[dict]:
        return self.history()

    def get(self, _owner_id: str, record_id: str) -> dict | None:
        return self.get_history_item(record_id)
