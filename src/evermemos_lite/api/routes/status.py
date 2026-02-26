from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/v1/status", tags=["status"])


@router.get("/{request_id}")
async def get_request_status(request: Request, request_id: str) -> dict:
    row = request.app.state.request_status_repo.get(request_id)
    if row is None:
        return {"status": "ok", "result": {"request_id": request_id, "state": "not_found"}}
    return {"status": "ok", "result": row}


@router.get("/vector-index/summary")
async def get_vector_index_summary(request: Request) -> dict:
    memory_service = request.app.state.memory_service
    return {"status": "ok", "result": memory_service.get_vector_index_stats()}
