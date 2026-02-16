from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from evermemos_lite.domain.policy import RuntimePolicy

router = APIRouter(prefix="/api/v1/runtime-policy", tags=["runtime-policy"])


class PolicyPatch(BaseModel):
    tenant_id: str = "default"
    ttl_sec: int | None = None
    vector_enabled: bool | None = None
    keyword_enabled: bool | None = None
    agentic_enabled: bool | None = None
    importance_threshold: float | None = None
    keyword_top_k: int | None = None
    vector_top_k: int | None = None
    rrf_k: int | None = None
    profile: str | None = None
    reason: str | None = None


@router.get("")
async def get_runtime_policy(request: Request, tenant_id: str = "default") -> dict:
    repo = request.app.state.runtime_policy_repo
    policy = repo.get(tenant_id)
    return {"status": "ok", "result": {"tenant_id": tenant_id, "policy": policy.to_dict() if policy else {}}}


# Keep handlers explicit to app state access.
@router.get("/resolve")
async def resolve_runtime_policy(request: Request, tenant_id: str = "default") -> dict:
    repo = request.app.state.runtime_policy_repo
    policy = repo.get(tenant_id)
    return {"status": "ok", "result": {"tenant_id": tenant_id, "policy": policy.to_dict() if policy else {}}}


@router.put("")
async def upsert_runtime_policy(request: Request, payload: PolicyPatch) -> dict:
    repo = request.app.state.runtime_policy_repo
    policy = RuntimePolicy(
        vector_enabled=payload.vector_enabled,
        keyword_enabled=payload.keyword_enabled,
        agentic_enabled=payload.agentic_enabled,
        importance_threshold=payload.importance_threshold,
        keyword_top_k=payload.keyword_top_k,
        vector_top_k=payload.vector_top_k,
        rrf_k=payload.rrf_k,
        profile=payload.profile,
        reason=payload.reason,
    )
    repo.upsert(payload.tenant_id, policy, ttl_sec=payload.ttl_sec)
    return {"status": "ok", "result": {"tenant_id": payload.tenant_id, "policy": policy.to_dict()}}


@router.delete("")
async def delete_runtime_policy(request: Request, tenant_id: str = "default") -> dict:
    repo = request.app.state.runtime_policy_repo
    deleted = repo.delete(tenant_id)
    return {"status": "ok", "result": {"deleted_count": deleted}}
