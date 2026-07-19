import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.schemas.document import DocumentPlanResponse
from app.schemas.action_pack import DocumentActionPackResponse
from app.schemas.ndis import NavigationPlanRequest, NavigationPlanResponse
from app.services.document_ingestion import (
    DocumentIngestionError,
    DocumentInputError,
    DocumentIngestionService,
)
from app.services.ndis_navigation import NdisNavigationError, NdisNavigationService
from app.services.ndis_action_pack import NdisActionPackError, NdisActionPackService
from app.core.auth import get_current_user_id
from app.services.records import RecordRepository

router = APIRouter(prefix="/api/v1/ndis-navigation", tags=["NDIS navigation"])


@router.post("/plan", response_model=NavigationPlanResponse)
async def create_navigation_plan(
    payload: NavigationPlanRequest, request: Request
) -> NavigationPlanResponse:
    service: NdisNavigationService = request.app.state.ndis_navigation_service
    try:
        return await service.create_plan(payload)
    except NdisNavigationError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)
        ) from error


@router.post("/document-plan", response_model=DocumentPlanResponse)
async def create_plan_from_document(
    request: Request,
    document: UploadFile = File(...),
    ndis_context: str = Form("{}"),
    user_id: str = Depends(get_current_user_id),
) -> DocumentPlanResponse:
    try:
        parsed_ndis_context = json.loads(ndis_context)
        if not isinstance(parsed_ndis_context, dict):
            raise ValueError("ndis_context must be a JSON object")
    except (json.JSONDecodeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ndis_context must be a valid JSON object.",
        ) from error

    service: DocumentIngestionService = request.app.state.document_ingestion_service
    try:
        result = await service.create_plan_from_document(
            filename=document.filename or "",
            content_type=document.content_type,
            content=await document.read(),
            ndis_context=parsed_ndis_context,
        )
        repository: RecordRepository = request.app.state.record_repository
        repository.create(user_id, "document_plan", result.source_filename, result.extracted_clinical_information.model_dump(exclude_none=True), result.model_dump())
        return result
    except DocumentInputError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error
    except DocumentIngestionError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)
        ) from error


@router.post("/action-pack", response_model=DocumentActionPackResponse)
async def create_action_pack(
    request: Request, document: UploadFile = File(...), ndis_context: str = Form("{}"), user_id: str = Depends(get_current_user_id)
) -> DocumentActionPackResponse:
    try:
        context = json.loads(ndis_context)
        if not isinstance(context, dict):
            raise ValueError
    except (json.JSONDecodeError, ValueError) as error:
        raise HTTPException(status_code=400, detail="ndis_context must be a valid JSON object.") from error
    ingestion: DocumentIngestionService = request.app.state.document_ingestion_service
    action_service: NdisActionPackService = request.app.state.ndis_action_pack_service
    try:
        extraction, preview = await ingestion.extract_clinical_information(
            document.filename or "", document.content_type, await document.read()
        )
        action_pack = await action_service.create(extraction.model_dump(exclude_none=True), context)
        result = DocumentActionPackResponse(
            source_filename=document.filename or "",
            extracted_clinical_information=extraction,
            action_pack=action_pack,
            source_text_preview=preview,
        )
        repository: RecordRepository = request.app.state.record_repository
        repository.create(user_id, "action_pack", result.source_filename, result.extracted_clinical_information.model_dump(exclude_none=True), result.model_dump())
        return result
    except DocumentInputError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (DocumentIngestionError, NdisActionPackError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.get("/records")
async def list_records(request: Request, user_id: str = Depends(get_current_user_id)) -> list[dict]:
    repository: RecordRepository | None = request.app.state.record_repository
    if repository is None:
        raise HTTPException(status_code=503, detail="Record persistence is not configured.")
    return repository.list(user_id)


@router.get("/records/{record_id}")
async def get_record(record_id: str, request: Request, user_id: str = Depends(get_current_user_id)) -> dict:
    repository: RecordRepository | None = request.app.state.record_repository
    if repository is None:
        raise HTTPException(status_code=503, detail="Record persistence is not configured.")
    record = repository.get(user_id, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found.")
    return record
