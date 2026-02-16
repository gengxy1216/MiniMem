from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/v1/status", tags=["status"])


@router.get("/{request_id}")
async def get_request_status(request: Request, request_id: str) -> dict:
    row = request.app.state.request_status_repo.get(request_id)
    if row is None:
        return {"status": "ok", "result": {"request_id": request_id, "state": "not_found"}}
    return {"status": "ok", "result": row}
