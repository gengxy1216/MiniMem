from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable


class SQLiteEngine:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self.connect() as conn:
            cur = conn.execute(sql, tuple(params))
            conn.commit()
            return cur.rowcount

    def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> int:
        if not rows:
            return 0
        with self.connect() as conn:
            cur = conn.executemany(sql, rows)
            conn.commit()
            return cur.rowcount

    def query_all(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            cur = conn.execute(sql, tuple(params))
            return [dict(row) for row in cur.fetchall()]

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        with self.connect() as conn:
            cur = conn.execute(sql, tuple(params))
            row = cur.fetchone()
            return dict(row) if row else None
