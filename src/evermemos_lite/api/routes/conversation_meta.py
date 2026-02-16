from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/conversation-meta", tags=["conversation-meta"])


class ConversationMetaPatch(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    group_id: str | None = None
    title: str = Field(min_length=1, max_length=200)
    conversation_id: str | None = None


@router.get("")
async def list_conversations(request: Request, user_id: str, group_id: str | None = None) -> dict:
    rows = request.app.state.conversation_meta_repo.list_by_user(user_id=user_id, group_id=group_id)
    return {"status": "ok", "result": {"items": rows, "total_count": len(rows)}}


@router.put("")
async def upsert_conversation(request: Request, payload: ConversationMetaPatch) -> dict:
    row = request.app.state.conversation_meta_repo.upsert(
        user_id=payload.user_id,
        group_id=payload.group_id,
        title=payload.title,
        conversation_id=payload.conversation_id,
    )
    return {"status": "ok", "result": row}
