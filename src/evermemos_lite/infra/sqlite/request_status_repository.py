from __future__ import annotations

import time
from typing import Any

from evermemos_lite.infra.sqlite.db import SQLiteEngine


class RequestStatusRepository:
    def __init__(self, engine: SQLiteEngine) -> None:
        self.engine = engine

    def upsert(
        self,
        *,
        request_id: str,
        status: str,
        ttl_sec: int,
        url: str | None = None,
        method: str | None = None,
        http_code: int | None = None,
        time_ms: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> None:
        now = int(time.time())
        self.engine.execute(
            """
            INSERT INTO request_status(
                request_id,status,ttl_sec,url,method,http_code,time_ms,start_time,end_time,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(request_id) DO UPDATE SET
                status=excluded.status,
                ttl_sec=excluded.ttl_sec,
                url=excluded.url,
                method=excluded.method,
                http_code=excluded.http_code,
                time_ms=excluded.time_ms,
                start_time=excluded.start_time,
                end_time=excluded.end_time,
                updated_at=excluded.updated_at
            """,
            (
                request_id,
                status,
                ttl_sec,
                url,
                method,
                http_code,
                time_ms,
                start_time,
                end_time,
                now,
            ),
        )

    def get(self, request_id: str) -> dict[str, Any] | None:
        row = self.engine.query_one(
            """
            SELECT request_id,status,ttl_sec,url,method,http_code,time_ms,start_time,end_time,updated_at
            FROM request_status
            WHERE request_id=?
            """,
            (request_id,),
        )
        if row is None:
            return None
        return row

    def cleanup_expired(self) -> int:
        now = int(time.time())
        return self.engine.execute(
            """
            DELETE FROM request_status
            WHERE updated_at + ttl_sec < ?
            """,
            (now,),
        )
