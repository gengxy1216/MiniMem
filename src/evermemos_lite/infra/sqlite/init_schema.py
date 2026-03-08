from __future__ import annotations

import os
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
            memory_category TEXT NOT NULL DEFAULT 'general',
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
        CREATE TABLE IF NOT EXISTS memory_memcell (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            group_id TEXT,
            cell_order INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(memory_id) REFERENCES episodic_memory(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_event_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            group_id TEXT,
            fact_order INTEGER NOT NULL,
            fact TEXT NOT NULL,
            fact_norm TEXT NOT NULL,
            created_at INTEGER NOT NULL,
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
        """
        CREATE INDEX IF NOT EXISTS idx_memory_memcell_memory
        ON memory_memcell(memory_id, cell_order)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memory_memcell_event
        ON memory_memcell(event_id, user_id, group_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memory_event_log_memory
        ON memory_event_log(memory_id, fact_order)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memory_event_log_event
        ON memory_event_log(event_id, user_id, group_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memory_event_log_fact_norm
        ON memory_event_log(fact_norm)
        """,
    ]
    for sql in ddl:
        engine.execute(sql)
    _ensure_keyword_index(engine)
    _ensure_event_log_keyword_index(engine)
    _ensure_foresight_keyword_index(engine)
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
    _ensure_column(
        engine,
        table="episodic_memory",
        column="memory_category",
        ddl="ALTER TABLE episodic_memory ADD COLUMN memory_category TEXT NOT NULL DEFAULT 'general'",
    )
    now = int(time.time())
    engine.execute(
        "UPDATE episodic_memory SET storage_tier='text_only' WHERE storage_tier IS NULL OR storage_tier=''"
    )
    engine.execute(
        "UPDATE episodic_memory SET memory_category='general' WHERE memory_category IS NULL OR memory_category=''"
    )
    engine.execute(
        "UPDATE episodic_memory SET updated_at=? WHERE updated_at IS NULL",
        (now,),
    )
    _rebuild_keyword_index_if_needed(engine)
    _rebuild_event_log_keyword_index_if_needed(engine)
    _rebuild_foresight_keyword_index_if_needed(engine)


def _ensure_column(engine: SQLiteEngine, *, table: str, column: str, ddl: str) -> None:
    rows = engine.query_all(f"PRAGMA table_info({table})")
    cols = {str(r.get("name")) for r in rows}
    if column in cols:
        return
    engine.execute(ddl)


def _ensure_keyword_index(engine: SQLiteEngine) -> None:
    tokenizer = _resolve_fts_tokenizer()
    try:
        existing = engine.query_one(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type='table' AND name='memory_keyword_fts'
            """
        )
        existing_sql = str((existing or {}).get("sql") or "").lower()
        if existing_sql and tokenizer not in existing_sql:
            engine.execute("DROP TABLE IF EXISTS memory_keyword_fts")
    except Exception:
        existing_sql = ""
    try:
        engine.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_keyword_fts
            USING fts5(
                memory_id UNINDEXED,
                user_id UNINDEXED,
                group_id UNINDEXED,
                search_text,
                tokenize='{tokenizer}'
            )
            """
        )
    except Exception:
        # Fallback for unsupported tokenizer/runtime, keep startup resilient.
        if tokenizer != "unicode61":
            try:
                engine.execute("DROP TABLE IF EXISTS memory_keyword_fts")
                engine.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_keyword_fts
                    USING fts5(
                        memory_id UNINDEXED,
                        user_id UNINDEXED,
                        group_id UNINDEXED,
                        search_text,
                        tokenize='unicode61'
                    )
                    """
                )
                return
            except Exception:
                return
        return


def _ensure_event_log_keyword_index(engine: SQLiteEngine) -> None:
    tokenizer = _resolve_fts_tokenizer()
    try:
        existing = engine.query_one(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type='table' AND name='event_log_keyword_fts'
            """
        )
        existing_sql = str((existing or {}).get("sql") or "").lower()
        if existing_sql and tokenizer not in existing_sql:
            engine.execute("DROP TABLE IF EXISTS event_log_keyword_fts")
    except Exception:
        existing_sql = ""
    try:
        engine.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS event_log_keyword_fts
            USING fts5(
                event_log_id UNINDEXED,
                memory_id UNINDEXED,
                user_id UNINDEXED,
                group_id UNINDEXED,
                fact_text,
                tokenize='{tokenizer}'
            )
            """
        )
    except Exception:
        # Fallback for unsupported tokenizer/runtime, keep startup resilient.
        if tokenizer != "unicode61":
            try:
                engine.execute("DROP TABLE IF EXISTS event_log_keyword_fts")
                engine.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS event_log_keyword_fts
                    USING fts5(
                        event_log_id UNINDEXED,
                        memory_id UNINDEXED,
                        user_id UNINDEXED,
                        group_id UNINDEXED,
                        fact_text,
                        tokenize='unicode61'
                    )
                    """
                )
                return
            except Exception:
                return
        return


def _ensure_foresight_keyword_index(engine: SQLiteEngine) -> None:
    tokenizer = _resolve_fts_tokenizer()
    try:
        existing = engine.query_one(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type='table' AND name='foresight_keyword_fts'
            """
        )
        existing_sql = str((existing or {}).get("sql") or "").lower()
        if existing_sql and tokenizer not in existing_sql:
            engine.execute("DROP TABLE IF EXISTS foresight_keyword_fts")
    except Exception:
        existing_sql = ""
    try:
        engine.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS foresight_keyword_fts
            USING fts5(
                foresight_id UNINDEXED,
                memory_id UNINDEXED,
                user_id UNINDEXED,
                group_id UNINDEXED,
                foresight_text,
                tokenize='{tokenizer}'
            )
            """
        )
    except Exception:
        if tokenizer != "unicode61":
            try:
                engine.execute("DROP TABLE IF EXISTS foresight_keyword_fts")
                engine.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS foresight_keyword_fts
                    USING fts5(
                        foresight_id UNINDEXED,
                        memory_id UNINDEXED,
                        user_id UNINDEXED,
                        group_id UNINDEXED,
                        foresight_text,
                        tokenize='unicode61'
                    )
                    """
                )
                return
            except Exception:
                return
        return


def _resolve_fts_tokenizer() -> str:
    raw = str(os.getenv("LITE_FTS_TOKENIZER", "unicode61") or "").strip().lower()
    if not raw:
        return "unicode61"
    allowed = {
        "unicode61": "unicode61",
        "trigram": "trigram",
        "porter": "porter unicode61",
    }
    return allowed.get(raw, "unicode61")


def _rebuild_keyword_index_if_needed(engine: SQLiteEngine) -> None:
    try:
        engine.execute(
            """
            DELETE FROM memory_keyword_fts
            WHERE memory_id IN (
                SELECT id FROM episodic_memory WHERE is_deleted=1
            )
            """
        )
        engine.execute(
            """
            INSERT INTO memory_keyword_fts(memory_id,user_id,group_id,search_text)
            SELECT
              m.id,
              m.user_id,
              COALESCE(m.group_id,''),
              trim(
                COALESCE(m.episode,'') || ' ' ||
                COALESCE(m.summary,'') || ' ' ||
                COALESCE(m.subject,'') || ' ' ||
                COALESCE(f.fact_text,'') || ' ' ||
                'cat_' || REPLACE(COALESCE(NULLIF(m.memory_category,''),'general'), ' ', '_') || ' ' ||
                'tier_' || REPLACE(COALESCE(NULLIF(m.storage_tier,''),'text_only'), ' ', '_')
              )
            FROM episodic_memory m
            LEFT JOIN (
              SELECT memory_id, group_concat(fact, ' ') AS fact_text
              FROM memory_fact
              GROUP BY memory_id
            ) f ON f.memory_id = m.id
            WHERE m.is_deleted=0
              AND NOT EXISTS (
                SELECT 1
                FROM memory_keyword_fts idx
                WHERE idx.memory_id = m.id
              )
            """
        )
    except Exception:
        return


def _rebuild_event_log_keyword_index_if_needed(engine: SQLiteEngine) -> None:
    try:
        engine.execute(
            """
            DELETE FROM event_log_keyword_fts
            WHERE memory_id IN (
                SELECT id FROM episodic_memory WHERE is_deleted=1
            )
            """
        )
        engine.execute(
            """
            INSERT INTO event_log_keyword_fts(event_log_id,memory_id,user_id,group_id,fact_text)
            SELECT
              CAST(e.id AS TEXT),
              e.memory_id,
              e.user_id,
              COALESCE(e.group_id, ''),
              trim(COALESCE(e.fact, ''))
            FROM memory_event_log e
            JOIN episodic_memory m ON m.id = e.memory_id
            WHERE m.is_deleted=0
              AND COALESCE(trim(e.fact), '')<>''
              AND NOT EXISTS (
                SELECT 1
                FROM event_log_keyword_fts idx
                WHERE idx.event_log_id = CAST(e.id AS TEXT)
              )
            """
        )
    except Exception:
        return


def _rebuild_foresight_keyword_index_if_needed(engine: SQLiteEngine) -> None:
    try:
        engine.execute(
            """
            DELETE FROM foresight_keyword_fts
            WHERE memory_id IN (
                SELECT id FROM episodic_memory WHERE is_deleted=1
            )
            """
        )
        engine.execute(
            """
            INSERT INTO foresight_keyword_fts(foresight_id,memory_id,user_id,group_id,foresight_text)
            SELECT
              CAST(f.id AS TEXT),
              f.memory_id,
              m.user_id,
              COALESCE(m.group_id, ''),
              trim(COALESCE(f.content, ''))
            FROM memory_foresight f
            JOIN episodic_memory m ON m.id = f.memory_id
            WHERE m.is_deleted=0
              AND COALESCE(trim(f.content), '')<>''
              AND NOT EXISTS (
                SELECT 1
                FROM foresight_keyword_fts idx
                WHERE idx.foresight_id = CAST(f.id AS TEXT)
              )
            """
        )
    except Exception:
        return
