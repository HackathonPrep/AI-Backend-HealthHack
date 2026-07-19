"""Seed a demo user and fake NDIS history into Supabase.

Creates (or reuses) a fixed demo user, then replaces that user's
`participant_records` with realistic fake `document_plan` and `action_pack`
rows that match the API response contracts. The Supabase service key bypasses
row-level security, so this can insert on the user's behalf.

Usage:
    python scripts/seed_demo_data.py
"""

from __future__ import annotations

import os
import sys

# Allow running as `python scripts/seed_demo_data.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supabase import Client, create_client

from app.core.config import get_settings

DEMO_EMAIL = os.getenv("DEMO_EMAIL", "demo@carematch.ai")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "demo-carematch-2026")


def _get_or_create_user(client: Client, email: str, password: str) -> str:
    """Return the demo user's id, creating the account if it does not exist."""
    try:
        created = client.auth.admin.create_user(
            {"email": email, "password": password, "email_confirm": True}
        )
        if created and created.user:
            print(f"Created demo user {email}")
            return created.user.id
    except Exception as error:  # noqa: BLE001 - account likely already exists
        print(f"create_user failed ({error}); looking up existing user")

    # Fall back to locating the existing user by email.
    page = 1
    while True:
        users = client.auth.admin.list_users(page=page, per_page=200)
        if not users:
            break
        for user in users:
            if (getattr(user, "email", None) or "").lower() == email.lower():
                print(f"Reusing existing demo user {email}")
                return user.id
        page += 1

    raise RuntimeError(f"Could not create or find demo user {email}")


def _document_plan_record(owner_id: str) -> dict:
    extraction = {
        "diagnosis_reason": "Right middle cerebral artery ischaemic stroke with left-sided hemiparesis",
        "mobility_status": "Requires one-person assistance and a four-wheel walker for short distances",
        "transfer_status": "Stand-by assistance for bed and chair transfers",
        "personal_care": "Needs prompting and physical help with showering and dressing",
        "cognition_mental_health": "Mild short-term memory difficulty; low mood since discharge",
        "living_situation": "Lives alone in a first-floor unit with internal stairs",
        "carer_availability": "Daughter visits on weekends only",
        "equipment_needs": "Shower chair and bed rail recommended by occupational therapist",
        "follow_up_requirements": "Outpatient physiotherapy and stroke clinic review in 6 weeks",
        "ndis_status": "Active plan, plan-managed",
    }
    plan = {
        "practical_needs_summary": (
            "Sarah needs daily personal care support, help re-establishing safe mobility at home, "
            "and equipment to shower safely while she recovers from a recent stroke."
        ),
        "recommended_support_categories": [
            {
                "category": "Core: Assistance with Daily Life (Support Workers, SIL)",
                "justification": "Daily help with showering, dressing and meal preparation while living alone.",
            },
            {
                "category": (
                    "Capacity Building: Improved Daily Living (Occupational Therapy, Physiotherapy, "
                    "Speech Pathology, Psychology, Dietetics, Community Nursing)"
                ),
                "justification": "Physiotherapy and occupational therapy to rebuild mobility and home safety.",
            },
            {
                "category": "Capital: Assistive Technology (Mobility equipment, hoists, beds)",
                "justification": "Shower chair and bed rail identified by the treating occupational therapist.",
            },
        ],
        "provider_referral_summary": (
            "56-year-old participant recovering from a right MCA stroke, living alone with internal stairs. "
            "Seeking in-home personal care, allied health, and minor assistive technology."
        ),
        "call_script": (
            "Hi, I'm calling on behalf of a plan-managed NDIS participant recovering from a stroke. "
            "She needs weekday morning personal care and is looking for a provider with availability in Parramatta. "
            "Do you have capacity to take a new referral this fortnight?"
        ),
        "next_steps_checklist": [
            "Confirm plan-managed funding for Core supports",
            "Request occupational therapy home safety assessment",
            "Arrange weekday morning personal care support worker",
            "Order shower chair and bed rail",
            "Book outpatient physiotherapy",
            "Schedule stroke clinic review at 6 weeks",
        ],
    }
    return {
        "owner_id": owner_id,
        "record_type": "document_plan",
        "source_filename": "Hospital Discharge Summary.pdf",
        "clinical_extraction": extraction,
        "result": {
            "source_filename": "Hospital Discharge Summary.pdf",
            "extracted_clinical_information": extraction,
            "plan": plan,
            "source_text_preview": (
                "DISCHARGE SUMMARY - Right MCA ischaemic stroke. Left hemiparesis. "
                "Independent prior to admission; now requires assistance with mobility and personal care..."
            ),
        },
    }


def _second_document_plan_record(owner_id: str) -> dict:
    extraction = {
        "diagnosis_reason": "Functional capacity assessment following stroke rehabilitation",
        "mobility_status": "Independent indoors with walker; unsafe on stairs and uneven ground",
        "personal_care": "Independent with set-up; slower with fine motor tasks",
        "cognition_mental_health": "Managing well; motivated to return to community activities",
        "living_situation": "Lives alone; wants to travel independently by public transport",
        "follow_up_requirements": "Travel training and community participation goals identified",
        "ndis_status": "Active plan, plan-managed",
    }
    plan = {
        "practical_needs_summary": (
            "Sarah is regaining independence and now wants support to rebuild confidence in the "
            "community and with public transport."
        ),
        "recommended_support_categories": [
            {
                "category": "Core: Assistance with Social, Economic and Community Participation",
                "justification": "Support worker to attend community activities and build social confidence.",
            },
            {
                "category": "Core: Transport",
                "justification": "Assistance to attend appointments and community programs while rebuilding travel skills.",
            },
        ],
        "provider_referral_summary": (
            "Participant progressing well post-stroke, seeking community participation support and "
            "travel training on Tuesdays and Thursdays in the Parramatta area."
        ),
        "call_script": (
            "Hi, I'm following up for a plan-managed participant who is ready for community participation "
            "support and travel training. She's available Tuesdays and Thursdays. Can you take a referral?"
        ),
        "next_steps_checklist": [
            "Confirm community participation funding",
            "Match a female support worker where possible",
            "Set weekly Tuesday and Thursday schedule",
            "Begin travel training plan",
            "Review progress against goals in 8 weeks",
        ],
    }
    return {
        "owner_id": owner_id,
        "record_type": "document_plan",
        "source_filename": "Functional Capacity Assessment.pdf",
        "clinical_extraction": extraction,
        "result": {
            "source_filename": "Functional Capacity Assessment.pdf",
            "extracted_clinical_information": extraction,
            "plan": plan,
            "source_text_preview": (
                "FUNCTIONAL CAPACITY ASSESSMENT - Participant independent indoors with walker. "
                "Goals: community participation, public transport confidence..."
            ),
        },
    }


def _action_pack_record(owner_id: str) -> dict:
    extraction = {
        "diagnosis_reason": "Right MCA stroke with left hemiparesis",
        "mobility_status": "One-person assistance for transfers; walker for short distances",
        "personal_care": "Assistance with showering and dressing",
        "living_situation": "Lives alone with internal stairs",
        "ndis_status": "Active plan, plan-managed",
    }
    action_pack = {
        "practical_needs_summary": (
            "Sarah needs a coordinated support package covering personal care, allied health, and "
            "assistive technology to recover safely at home after a stroke."
        ),
        "evidence_checklist": [
            {
                "item": "Occupational therapy functional assessment",
                "status": "missing",
                "source_hint": "Request from treating OT or community allied health provider",
            },
            {
                "item": "Hospital discharge summary",
                "status": "present",
                "source_hint": "Provided by the discharging hospital",
            },
            {
                "item": "Current NDIS plan with funding categories",
                "status": "present",
                "source_hint": "Participant's plan manager or myplace portal",
            },
        ],
        "access_or_review_recommended": True,
        "access_or_review_rationale": (
            "Functional capacity has changed significantly since the last plan; a plan review may be "
            "needed to fund additional daily living and assistive technology supports."
        ),
        "provider_service_categories": [
            "Personal care and daily living support",
            "Occupational therapy",
            "Physiotherapy",
        ],
        "provider_referral_summary": (
            "Participant recovering from stroke, living alone, seeking weekday personal care, allied "
            "health, and minor assistive technology in the Parramatta area."
        ),
        "family_call_script": (
            "Hi, I'm calling about my mum who recently had a stroke and is now home alone. We're looking "
            "for morning personal care support and a physiotherapist. What availability do you have?"
        ),
        "email_draft": (
            "Dear Provider,\n\nWe are seeking supports for a plan-managed NDIS participant recovering from "
            "a stroke. She requires weekday personal care, physiotherapy, and occupational therapy, plus a "
            "shower chair and bed rail. Please let us know your current availability.\n\nKind regards,\nSarah's family"
        ),
        "follow_up_tasks": [
            {
                "task": "Book occupational therapy home safety assessment",
                "owner": "Support coordinator",
                "timeframe": "This week",
                "priority": "high",
            },
            {
                "task": "Arrange weekday morning personal care",
                "owner": "Plan manager",
                "timeframe": "Within 2 weeks",
                "priority": "urgent",
            },
            {
                "task": "Order shower chair and bed rail",
                "owner": "Occupational therapist",
                "timeframe": "This month",
                "priority": "routine",
            },
        ],
    }
    return {
        "owner_id": owner_id,
        "record_type": "action_pack",
        "source_filename": "Participant Support Form.pdf",
        "clinical_extraction": extraction,
        "result": {
            "source_filename": "Participant Support Form.pdf",
            "extracted_clinical_information": extraction,
            "action_pack": action_pack,
            "source_text_preview": (
                "PARTICIPANT SUPPORT FORM - Post-stroke, lives alone, needs personal care and allied "
                "health. Family requesting help arranging supports..."
            ),
        },
    }


def main() -> None:
    settings = get_settings()
    if not settings.supabase_enabled:
        raise SystemExit(
            "Supabase is not configured. Set SUPABASE_URL, SUPABASE_SECRET_KEY and "
            "SUPABASE_JWKS_URL in AI-Backend-HealthHack/.env before seeding."
        )

    client: Client = create_client(settings.supabase_url, settings.supabase_secret_key)
    owner_id = _get_or_create_user(client, DEMO_EMAIL, DEMO_PASSWORD)

    # Idempotent: clear then re-insert this user's records.
    client.table("participant_records").delete().eq("owner_id", owner_id).execute()

    records = [
        _document_plan_record(owner_id),
        _second_document_plan_record(owner_id),
        _action_pack_record(owner_id),
    ]
    inserted = client.table("participant_records").insert(records).execute()

    print(f"Seeded {len(inserted.data)} records for {DEMO_EMAIL} (owner_id={owner_id})")
    print("Demo login credentials (use these in carematch-ai/.env):")
    print(f"  VITE_DEMO_EMAIL={DEMO_EMAIL}")
    print(f"  VITE_DEMO_PASSWORD={DEMO_PASSWORD}")


if __name__ == "__main__":
    main()
