from __future__ import annotations

from functools import partial

import anyio
from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/v1/graph", tags=["graph"])


@router.get("/search")
async def search_graph(
    request: Request,
    query: str,
    user_id: str | None = None,
    group_id: str | None = None,
    limit: int = 10,
) -> dict:
    q = query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query must not be blank")
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit out of range")
    rows = await anyio.to_thread.run_sync(
        partial(
            request.app.state.memory_service.search_graph,
            query=q,
            user_id=user_id,
            group_id=group_id,
            limit=limit,
        )
    )
    return {"status": "ok", "result": {"items": rows, "total_count": len(rows)}}


@router.get("/neighbors")
async def graph_neighbors(
    request: Request,
    entity: str,
    user_id: str | None = None,
    group_id: str | None = None,
    limit: int = 20,
) -> dict:
    name = entity.strip()
    if not name:
        raise HTTPException(status_code=400, detail="entity must not be blank")
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit out of range")
    rows = await anyio.to_thread.run_sync(
        partial(
            request.app.state.memory_service.graph_neighbors,
            entity_name=name,
            user_id=user_id,
            group_id=group_id,
            limit=limit,
        )
    )
    return {"status": "ok", "result": {"items": rows, "total_count": len(rows)}}
