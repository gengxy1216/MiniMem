from __future__ import annotations

import time
import uuid
from typing import Any

from evermemos_lite.infra.sqlite.db import SQLiteEngine


class ConversationMetaRepository:
    def __init__(self, engine: SQLiteEngine) -> None:
        self.engine = engine

    def list_by_user(self, user_id: str, group_id: str | None = None) -> list[dict[str, Any]]:
        if group_id:
            return self.engine.query_all(
                """
                SELECT id,user_id,group_id,title,updated_at
                FROM conversation_meta
                WHERE user_id=? AND group_id=?
                ORDER BY updated_at DESC
                """,
                (user_id, group_id),
            )
        return self.engine.query_all(
            """
            SELECT id,user_id,group_id,title,updated_at
            FROM conversation_meta
            WHERE user_id=?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        )

    def upsert(
        self,
        *,
        user_id: str,
        group_id: str | None,
        title: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        cid = conversation_id or uuid.uuid4().hex
        now = int(time.time())
        self.engine.execute(
            """
            INSERT INTO conversation_meta(id,user_id,group_id,title,updated_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
              user_id=excluded.user_id,
              group_id=excluded.group_id,
              title=excluded.title,
              updated_at=excluded.updated_at
            """,
            (cid, user_id, group_id, title, now),
        )
        return {
            "id": cid,
            "user_id": user_id,
            "group_id": group_id,
            "title": title,
            "updated_at": now,
        }
