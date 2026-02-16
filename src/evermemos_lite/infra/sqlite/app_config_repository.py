from __future__ import annotations

import time

from evermemos_lite.infra.sqlite.db import SQLiteEngine


class AppConfigRepository:
    def __init__(self, engine: SQLiteEngine) -> None:
        self.engine = engine

    def get(self, key: str) -> str | None:
        row = self.engine.query_one("SELECT value FROM app_config WHERE key=?", (key,))
        return None if row is None else str(row["value"])

    def get_many(self, keys: list[str]) -> dict[str, str]:
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        rows = self.engine.query_all(
            f"SELECT key,value FROM app_config WHERE key IN ({placeholders})", keys
        )
        return {str(row["key"]): str(row["value"]) for row in rows}

    def upsert(self, key: str, value: str) -> None:
        now = int(time.time())
        self.engine.execute(
            """
            INSERT INTO app_config(key,value,updated_at)
            VALUES(?,?,?)
            ON CONFLICT(key) DO UPDATE SET
              value=excluded.value,
              updated_at=excluded.updated_at
            """,
            (key, value, now),
        )

    def upsert_many(self, items: dict[str, str]) -> None:
        for key, value in items.items():
            self.upsert(key, value)
