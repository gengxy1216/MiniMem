from __future__ import annotations

import time

from evermemos_lite.infra.sqlite.db import SQLiteEngine


def init_schema(engine: SQLiteEngine) -> None:
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS episodic_memory (
            id TEXT PRIMARY KEY,
            event_id TEXT UNIQUE NOT NULL,
            source_message_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            group_id TEXT,
            timestamp INTEGER NOT NULL,
            role TEXT NOT NULL,
            sender TEXT NOT NULL,
            sender_name TEXT,
            group_name TEXT,
            episode TEXT NOT NULL,
            summary TEXT NOT NULL,
            subject TEXT NOT NULL,
            importance_score REAL NOT NULL DEFAULT 0,
            scene_id TEXT,
            storage_tier TEXT NOT NULL DEFAULT 'text_only',
            is_deleted INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_fact (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            fact TEXT NOT NULL,
            FOREIGN KEY(memory_id) REFERENCES episodic_memory(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_foresight (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            content TEXT NOT NULL,
            start_time INTEGER,
            end_time INTEGER,
            confidence REAL NOT NULL DEFAULT 0.5,
            FOREIGN KEY(memory_id) REFERENCES episodic_memory(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS profile_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            group_id TEXT,
            profile_json TEXT NOT NULL,
            timestamp INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS app_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS runtime_policy (
            tenant_id TEXT PRIMARY KEY,
            policy_json TEXT NOT NULL,
            expires_at INTEGER,
            updated_at INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS request_status (
            request_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            ttl_sec INTEGER NOT NULL,
            url TEXT,
            method TEXT,
            http_code INTEGER,
            time_ms INTEGER,
            start_time INTEGER,
            end_time INTEGER,
            updated_at INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS conversation_meta (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            group_id TEXT,
            title TEXT,
            updated_at INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_vector (
            id TEXT PRIMARY KEY,
            memory_type TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            group_id TEXT,
            timestamp INTEGER NOT NULL,
            importance_score REAL NOT NULL,
            vector_dim INTEGER NOT NULL,
            vector_dtype TEXT NOT NULL,
            model_name TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memscene (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            group_id TEXT,
            summary TEXT NOT NULL,
            centroid_vector_json TEXT NOT NULL,
            memory_count INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            last_memory_ts INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memscene_memory_link (
            memory_id TEXT PRIMARY KEY,
            scene_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            group_id TEXT,
            created_at INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS profile_conflict_log (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            group_id TEXT,
            field_name TEXT NOT NULL,
            old_value TEXT,
            new_value TEXT,
            happened_at INTEGER NOT NULL,
            evidence_event_id TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS conversation_segment (
            conversation_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            group_id TEXT,
            segment_seq INTEGER NOT NULL DEFAULT 1,
            turns_markdown TEXT NOT NULL,
            last_query TEXT NOT NULL,
            turn_count INTEGER NOT NULL DEFAULT 0,
            start_time INTEGER NOT NULL,
            last_time INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """,
    ]
    for sql in ddl:
        engine.execute(sql)
    _ensure_column(
        engine,
        table="episodic_memory",
        column="scene_id",
        ddl="ALTER TABLE episodic_memory ADD COLUMN scene_id TEXT",
    )
    _ensure_column(
        engine,
        table="episodic_memory",
        column="storage_tier",
        ddl="ALTER TABLE episodic_memory ADD COLUMN storage_tier TEXT NOT NULL DEFAULT 'text_only'",
    )
    now = int(time.time())
    engine.execute(
        "UPDATE episodic_memory SET storage_tier='text_only' WHERE storage_tier IS NULL OR storage_tier=''"
    )
    engine.execute(
        "UPDATE episodic_memory SET updated_at=? WHERE updated_at IS NULL",
        (now,),
    )


def _ensure_column(engine: SQLiteEngine, *, table: str, column: str, ddl: str) -> None:
    rows = engine.query_all(f"PRAGMA table_info({table})")
    cols = {str(r.get("name")) for r in rows}
    if column in cols:
        return
    engine.execute(ddl)
