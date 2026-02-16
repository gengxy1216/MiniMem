from __future__ import annotations

import json
import time

from evermemos_lite.domain.policy import RuntimePolicy
from evermemos_lite.infra.sqlite.db import SQLiteEngine


class RuntimePolicyRepository:
    def __init__(self, engine: SQLiteEngine) -> None:
        self.engine = engine

    def get(self, tenant_id: str) -> RuntimePolicy | None:
        if not tenant_id:
            return None
        now = int(time.time())
        row = self.engine.query_one(
            """
            SELECT policy_json,expires_at
            FROM runtime_policy
            WHERE tenant_id=?
            """,
            (tenant_id,),
        )
        if row is None:
            return None
        expires_at = row.get("expires_at")
        if expires_at is not None and int(expires_at) < now:
            self.delete(tenant_id)
            return None
        return RuntimePolicy.from_dict(json.loads(str(row["policy_json"])))

    def upsert(self, tenant_id: str, policy: RuntimePolicy, ttl_sec: int | None = None) -> None:
        now = int(time.time())
        expires_at = (now + int(ttl_sec)) if ttl_sec and ttl_sec > 0 else None
        self.engine.execute(
            """
            INSERT INTO runtime_policy(tenant_id,policy_json,expires_at,updated_at)
            VALUES(?,?,?,?)
            ON CONFLICT(tenant_id) DO UPDATE SET
              policy_json=excluded.policy_json,
              expires_at=excluded.expires_at,
              updated_at=excluded.updated_at
            """,
            (tenant_id, json.dumps(policy.to_dict(), ensure_ascii=False), expires_at, now),
        )

    def delete(self, tenant_id: str) -> int:
        return self.engine.execute("DELETE FROM runtime_policy WHERE tenant_id=?", (tenant_id,))

    def cleanup_expired(self) -> int:
        return self.engine.execute(
            "DELETE FROM runtime_policy WHERE expires_at IS NOT NULL AND expires_at < ?",
            (int(time.time()),),
        )
