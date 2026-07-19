from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.core.auth import get_current_user_id
from app.schemas.profile import ProfileApprovalRequest, ProfileApprovalResponse
from app.services.records import RecordRepository

router = APIRouter(prefix="/api/v1/demo", tags=["demo data"])


class ReferralCreateRequest(BaseModel):
    provider_id: str
    summary: str | None = Field(default=None, max_length=3_000)


class ReferralStatusRequest(BaseModel):
    status: Literal[
        "pending",
        "under_review",
        "information_requested",
        "accepted",
        "declined",
        "contacted",
        "intake_booked",
        "service_agreement",
        "ready_to_commence",
        "commenced",
    ]


def _repository(request: Request) -> RecordRepository:
    repository: RecordRepository | None = request.app.state.record_repository
    if repository is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")
    return repository


@router.get("/profile")
async def get_demo_profile(
    request: Request, _user_id: str = Depends(get_current_user_id)
) -> dict:
    return _repository(request).profile()


@router.post("/profile/approval", response_model=ProfileApprovalResponse, status_code=201)
async def create_profile_approval(
    payload: ProfileApprovalRequest,
    request: Request,
    _user_id: str = Depends(get_current_user_id),
) -> ProfileApprovalResponse:
    """Create an immutable participant-approved profile snapshot."""
    # #region agent log
    import json as _json, time as _time
    _dbg = "/home/luki/Documents/GitHub/AiHealthHack/.cursor/debug-a5c484.log"
    def _agent_log(hypothesis_id: str, message: str, data: dict) -> None:
        with open(_dbg, "a", encoding="utf-8") as _f:
            _f.write(_json.dumps({
                "sessionId": "a5c484",
                "runId": "pre-fix",
                "hypothesisId": hypothesis_id,
                "location": "demo.py:create_profile_approval",
                "message": message,
                "data": data,
                "timestamp": int(_time.time() * 1000),
            }) + "\n")
    # #endregion
    try:
        row = _repository(request).save_profile_approval(payload)
        # #region agent log
        _agent_log("C", "row before response validation", {
            "row_keys": list(row.keys()) if isinstance(row, dict) else type(row).__name__,
            "has_id": bool(isinstance(row, dict) and row.get("id")),
            "has_created_at": bool(isinstance(row, dict) and row.get("created_at")),
        })
        # #endregion
        return ProfileApprovalResponse.model_validate(row)
    except Exception as error:
        # #region agent log
        _agent_log("C", "create_profile_approval failed", {
            "error_type": type(error).__name__,
            "error_message": str(error)[:500],
        })
        # #endregion
        raise


@router.get("/history")
async def get_demo_history(
    request: Request, _user_id: str = Depends(get_current_user_id)
) -> list[dict]:
    return _repository(request).history()


@router.get("/chat-history")
async def get_demo_chat_history(
    request: Request,
    session_id: str | None = Query(default=None),
    _user_id: str = Depends(get_current_user_id),
) -> list[dict]:
    return _repository(request).list_chat_history(session_id)


@router.get("/providers")
async def get_provider_matches(
    request: Request,
    limit: int = Query(default=20, ge=1, le=50),
    _user_id: str = Depends(get_current_user_id),
) -> list[dict]:
    return _repository(request).providers(limit)


@router.get("/referrals")
async def get_referrals(
    request: Request,
    scope: Literal["patient", "all"] = "patient",
    _user_id: str = Depends(get_current_user_id),
) -> list[dict]:
    return _repository(request).referrals(patient_only=scope == "patient")


@router.post("/referrals", status_code=201)
async def create_referral(
    payload: ReferralCreateRequest,
    request: Request,
    _user_id: str = Depends(get_current_user_id),
) -> dict:
    return _repository(request).create_referral(payload.provider_id, payload.summary)


@router.patch("/referrals/{referral_id}")
async def update_referral(
    referral_id: str,
    payload: ReferralStatusRequest,
    request: Request,
    _user_id: str = Depends(get_current_user_id),
) -> dict:
    referral = _repository(request).update_referral(referral_id, payload.status)
    if referral is None:
        raise HTTPException(status_code=404, detail="Referral not found.")
    return referral
