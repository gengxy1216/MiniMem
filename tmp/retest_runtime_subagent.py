import json
import os
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from flockmem.bootstrap.app_factory import create_app
from flockmem.config.settings import LiteSettings
from flockmem.testing.writable_tempdir import WritableTempDir


def build_client(prefix: str):
    tmp = WritableTempDir(ignore_cleanup_errors=True)
    env = {
        "LITE_DATA_DIR": str(Path(tmp.name) / f"{prefix}-data"),
        "LITE_CONFIG_DIR": str(Path(tmp.name) / f"{prefix}-config"),
        "LITE_ADMIN_TOKEN": f"{prefix}-admin-token",
        "LITE_ADMIN_ALLOW_LOCALHOST": "false",
        "LITE_RETRIEVAL_PROFILE": "keyword",
        "LITE_CHAT_PROVIDER": "openai",
        "LITE_CHAT_BASE_URL": "https://chat.example/v1",
        "LITE_CHAT_API_KEY": "qa-chat-key",
        "LITE_CHAT_MODEL": "qa-chat-model",
        "LITE_EMBEDDING_PROVIDER": "openai",
        "LITE_EMBEDDING_BASE_URL": "https://embed.example/v1",
        "LITE_EMBEDDING_API_KEY": "qa-embed-key",
        "LITE_EMBEDDING_MODEL": "qa-embed-model",
        "LITE_EXTRACTOR_PROVIDER": "rule",
        "LITE_RERANK_PROVIDER": "chat_model",
    }
    with patch.dict(os.environ, env, clear=True):
        settings = LiteSettings.from_env()
    app = create_app(settings)
    return TestClient(app), tmp


def post_json(client: TestClient, path: str, payload: dict):
    r = client.post(path, json=payload)
    body = {}
    try:
        body = r.json()
    except Exception:
        body = {"_raw": r.text}
    return r.status_code, body


runtime_ids = ["openclaw", "codex", "nanobot"]
all_rows = []
overall_pass = True

for runtime_id in runtime_ids:
    run_id = uuid.uuid4().hex[:8]
    client, tmp = build_client(prefix=f"qa-subagent-{runtime_id}-{run_id}")
    try:
        actor_a = f"{runtime_id}-subA"
        actor_b = f"{runtime_id}-subB"
        team_id = f"team-{runtime_id}-{run_id}"
        scope_a = f"u-{runtime_id}-a-{run_id}"
        scope_b = f"u-{runtime_id}-b-{run_id}"
        session_id = f"sess-{runtime_id}-{run_id}"

        k_own_a = f"k-{runtime_id}-ownA-{run_id}"
        k_own_b = f"k-{runtime_id}-ownB-{run_id}"
        k_shared = f"k-{runtime_id}-shared-{run_id}"

        def ingest_payload(*, knowledge_id: str, scope_type: str, scope_id: str, actor: str, subagent: str, read_acl: list[str], write_acl: list[str]):
            return {
                "knowledge_id": knowledge_id,
                "scope_type": scope_type,
                "scope_id": scope_id,
                "content": {"text": f"{knowledge_id}-payload"},
                "change_type": "update",
                "changed_by": "agent",
                "actor_id": actor,
                "read_acl": read_acl,
                "write_acl": write_acl,
                "coordination_mode": "inruntime_a2a",
                "coordination_id": f"coord-{knowledge_id}-{uuid.uuid4().hex[:6]}",
                "runtime_id": runtime_id,
                "agent_id": f"{runtime_id}-agent",
                "subagent_id": subagent,
                "team_id": team_id,
                "session_id": session_id,
            }

        ing_a_status, ing_a_body = post_json(
            client,
            "/api/v1/collective/ingest",
            ingest_payload(
                knowledge_id=k_own_a,
                scope_type="personal",
                scope_id=scope_a,
                actor=actor_a,
                subagent="subA",
                read_acl=[actor_a],
                write_acl=[actor_a],
            ),
        )
        ing_b_status, ing_b_body = post_json(
            client,
            "/api/v1/collective/ingest",
            ingest_payload(
                knowledge_id=k_own_b,
                scope_type="personal",
                scope_id=scope_b,
                actor=actor_b,
                subagent="subB",
                read_acl=[actor_b],
                write_acl=[actor_b],
            ),
        )
        ing_s_status, ing_s_body = post_json(
            client,
            "/api/v1/collective/ingest",
            ingest_payload(
                knowledge_id=k_shared,
                scope_type="team",
                scope_id=team_id,
                actor=actor_a,
                subagent="subA",
                read_acl=[actor_a, actor_b],
                write_acl=[actor_a, actor_b],
            ),
        )

        shared_revision_id = (ing_s_body.get("result") or {}).get("revision_id")

        ctx_a_status, ctx_a_body = post_json(
            client,
            "/api/v1/collective/context",
            {
                "query": f"context-{runtime_id}-subA",
                "actor_id": actor_a,
                "personal_scope_id": scope_a,
                "team_scope_id": team_id,
                "include_global": False,
                "top_k": 10,
                "coordination_mode": "inruntime_a2a",
                "coordination_id": f"coord-ctx-a-{runtime_id}-{uuid.uuid4().hex[:6]}",
                "runtime_id": runtime_id,
                "agent_id": f"{runtime_id}-agent",
                "subagent_id": "subA",
                "team_id": team_id,
                "session_id": session_id,
            },
        )

        ctx_b_status, ctx_b_body = post_json(
            client,
            "/api/v1/collective/context",
            {
                "query": f"context-{runtime_id}-subB",
                "actor_id": actor_b,
                "personal_scope_id": scope_b,
                "team_scope_id": team_id,
                "include_global": False,
                "top_k": 10,
                "coordination_mode": "inruntime_a2a",
                "coordination_id": f"coord-ctx-b-{runtime_id}-{uuid.uuid4().hex[:6]}",
                "runtime_id": runtime_id,
                "agent_id": f"{runtime_id}-agent",
                "subagent_id": "subB",
                "team_id": team_id,
                "session_id": session_id,
            },
        )

        ids_a = [item.get("knowledge_id") for item in ((ctx_a_body.get("result") or {}).get("items") or [])]
        ids_b = [item.get("knowledge_id") for item in ((ctx_b_body.get("result") or {}).get("items") or [])]

        suba_expected = {k_own_a, k_shared}
        subb_expected = {k_own_b, k_shared}

        suba_visibility_pass = (ctx_a_status == 200) and (set(ids_a) == suba_expected)
        subb_visibility_pass = (ctx_b_status == 200) and (set(ids_b) == subb_expected)

        fb_a_status, fb_a_body = post_json(
            client,
            "/api/v1/collective/feedback",
            {
                "knowledge_id": k_shared,
                "revision_id": shared_revision_id,
                "feedback_type": "execution_signal",
                "feedback_payload": {
                    "outcome_status": "success",
                    "tool_error_count": 0,
                    "retry_count": 0,
                    "rollback_flag": False,
                    "reuse_hit": True,
                },
                "actor": actor_a,
                "coordination_mode": "inruntime_a2a",
                "coordination_id": f"coord-fb-a-{runtime_id}-{uuid.uuid4().hex[:6]}",
                "runtime_id": runtime_id,
                "agent_id": f"{runtime_id}-agent",
                "subagent_id": "subA",
                "team_id": team_id,
                "session_id": session_id,
            },
        )
        fb_b_status, fb_b_body = post_json(
            client,
            "/api/v1/collective/feedback",
            {
                "knowledge_id": k_shared,
                "revision_id": shared_revision_id,
                "feedback_type": "execution_signal",
                "feedback_payload": {
                    "outcome_status": "success",
                    "tool_error_count": 0,
                    "retry_count": 0,
                    "rollback_flag": False,
                    "reuse_hit": False,
                },
                "actor": actor_b,
                "coordination_mode": "inruntime_a2a",
                "coordination_id": f"coord-fb-b-{runtime_id}-{uuid.uuid4().hex[:6]}",
                "runtime_id": runtime_id,
                "agent_id": f"{runtime_id}-agent",
                "subagent_id": "subB",
                "team_id": team_id,
                "session_id": session_id,
            },
        )

        fb_a_id = (fb_a_body.get("result") or {}).get("feedback_id")
        fb_b_id = (fb_b_body.get("result") or {}).get("feedback_id")
        shared_feedback_dual_pass = (
            fb_a_status == 200
            and fb_b_status == 200
            and bool(str(fb_a_id or "").strip())
            and bool(str(fb_b_id or "").strip())
        )

        runtime_pass = (
            ing_a_status == 200
            and ing_b_status == 200
            and ing_s_status == 200
            and suba_visibility_pass
            and subb_visibility_pass
            and shared_feedback_dual_pass
        )

        overall_pass = overall_pass and runtime_pass

        all_rows.append(
            {
                "runtime_id": runtime_id,
                "knowledge_ids": {
                    "own_subA": k_own_a,
                    "own_subB": k_own_b,
                    "shared": k_shared,
                },
                "http_status": {
                    "ingest_own_subA": ing_a_status,
                    "ingest_own_subB": ing_b_status,
                    "ingest_shared": ing_s_status,
                    "context_subA": ctx_a_status,
                    "context_subB": ctx_b_status,
                    "feedback_shared_subA": fb_a_status,
                    "feedback_shared_subB": fb_b_status,
                },
                "assertions": {
                    "subA_only_own_plus_shared": suba_visibility_pass,
                    "subB_only_own_plus_shared": subb_visibility_pass,
                    "both_feedback_shared_success": shared_feedback_dual_pass,
                },
                "context_visible": {
                    "subA": ids_a,
                    "subB": ids_b,
                },
                "ids": {
                    "shared_revision_id": shared_revision_id,
                    "feedback_id_subA": fb_a_id,
                    "feedback_id_subB": fb_b_id,
                },
                "runtime_pass": runtime_pass,
            }
        )
    finally:
        client.close()
        tmp.cleanup()

print(json.dumps({"overall_pass": overall_pass, "results": all_rows}, ensure_ascii=False, indent=2))
