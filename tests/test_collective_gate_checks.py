from __future__ import annotations

import os
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from flockmem.bootstrap.app_factory import create_app
from flockmem.config.settings import LiteSettings
from flockmem.testing.writable_tempdir import WritableTempDir

_REQUIRED_ROUTES = {
    "/api/v1/collective/ingest",
    "/api/v1/collective/context",
    "/api/v1/collective/feedback",
}


@dataclass(frozen=True)
class _GateCase:
    gate_id: str
    category: str
    method: str
    path: str
    payload: dict[str, object]
    expect_client_error: bool = False


_GATE_CASES: tuple[_GateCase, ...] = (
    _GateCase(
        gate_id="GATE-NORMAL-INGEST",
        category="normal",
        method="POST",
        path="/api/v1/collective/ingest",
        payload={
            "knowledge_id": "k-gate-normal-1",
            "scope_type": "personal",
            "scope_id": "u-gate",
            "content": {"text": "gate normal ingest"},
            "change_type": "create",
            "changed_by": "agent",
            "actor_id": "qa-gate",
            "coordination_mode": "inruntime_a2a",
            "coordination_id": "coord-gate-normal-1",
            "runtime_id": "codex",
            "agent_id": "qa-gate",
            "subagent_id": "qa-gate-sub",
        },
    ),
    _GateCase(
        gate_id="GATE-NORMAL-CONTEXT",
        category="normal",
        method="POST",
        path="/api/v1/collective/context",
        payload={
            "query": "gate context lookup",
            "actor_id": "qa-gate",
            "personal_scope_id": "u-gate",
            "include_global": True,
            "top_k": 5,
        },
    ),
    _GateCase(
        gate_id="GATE-BOUNDARY-INVALID-SCOPE",
        category="boundary",
        method="POST",
        path="/api/v1/collective/ingest",
        payload={
            "knowledge_id": "k-gate-boundary-scope",
            "scope_type": "unknown_scope",
            "scope_id": "u-gate",
            "content": {"text": "invalid scope"},
            "changed_by": "agent",
            "actor_id": "qa-gate",
        },
        expect_client_error=True,
    ),
    _GateCase(
        gate_id="GATE-BOUNDARY-MISSING-REQUIRED",
        category="boundary",
        method="POST",
        path="/api/v1/collective/feedback",
        payload={
            "knowledge_id": "k-gate-boundary-feedback",
            "feedback_type": "execution_signal",
            "feedback_payload": {"outcome_status": "failed"},
        },
        expect_client_error=True,
    ),
    _GateCase(
        gate_id="GATE-RECOVERY-UNKNOWN-KNOWLEDGE",
        category="recovery",
        method="POST",
        path="/api/v1/collective/feedback",
        payload={
            "knowledge_id": "k-gate-unknown",
            "revision_id": "rev-gate-unknown",
            "feedback_type": "execution_signal",
            "feedback_payload": {
                "outcome_status": "failed",
                "tool_error_count": 1,
                "retry_count": 0,
                "rollback_flag": False,
                "reuse_hit": False,
            },
            "actor": "qa-gate",
            "coordination_mode": "inruntime_a2a",
            "coordination_id": "coord-gate-recovery-1",
        },
        expect_client_error=True,
    ),
)


def _build_client() -> tuple[TestClient, set[str], WritableTempDir]:
    tmp = WritableTempDir(ignore_cleanup_errors=True)
    env = {
        "LITE_DATA_DIR": str(Path(tmp.name) / "qa-gate-data"),
        "LITE_CONFIG_DIR": str(Path(tmp.name) / "qa-gate-config"),
        "LITE_ADMIN_TOKEN": "qa-gate-admin-token",
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
    routes = {getattr(route, "path", "") for route in app.routes}
    return TestClient(app), routes, tmp


class CollectiveGateChecksTests(unittest.TestCase):
    def _request(self, client: TestClient, case: _GateCase) -> int:
        if case.method == "POST":
            response = client.post(case.path, json=case.payload)
        elif case.method == "GET":
            response = client.get(case.path, params=case.payload)
        else:
            raise ValueError(f"unsupported method: {case.method}")
        return int(response.status_code)

    def test_gate_matrix_has_normal_boundary_recovery(self) -> None:
        categories = {case.category for case in _GATE_CASES}
        self.assertIn("normal", categories)
        self.assertIn("boundary", categories)
        self.assertIn("recovery", categories)

    def test_collective_routes_are_registered_for_v1_gate(self) -> None:
        client, routes, tmp = _build_client()
        self.addCleanup(tmp.cleanup)
        missing = sorted(_REQUIRED_ROUTES - routes)
        self.assertEqual(
            [],
            missing,
            "collective v1 gate requires all routes to be registered",
        )
        client.close()

    def test_gate_cases_execute_without_server_error(self) -> None:
        client, routes, tmp = _build_client()
        self.addCleanup(tmp.cleanup)
        missing = sorted(_REQUIRED_ROUTES - routes)
        if missing:
            self.fail("collective gate checks blocked by missing routes: " + ", ".join(missing))

        for case in _GATE_CASES:
            with self.subTest(gate_id=case.gate_id):
                status_code = self._request(client, case)
                self.assertNotIn(status_code, {404, 405} if not case.expect_client_error else {405})
                self.assertLess(status_code, 500)
                if case.expect_client_error:
                    self.assertGreaterEqual(status_code, 400)
                    self.assertLess(status_code, 500)
        client.close()

    def test_gate_required_artifacts_exist(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        benchmark_script = repo_root / "tools" / "benchmark_collective_baseline.py"
        qa_report = repo_root / "docs" / "reports" / "team-qa-gate-report-2026-03-08.md"
        self.assertTrue(
            benchmark_script.exists(),
            "v1 perf gate requires tools/benchmark_collective_baseline.py",
        )
        self.assertTrue(
            qa_report.exists(),
            "v1 qa gate requires docs/reports/team-qa-gate-report-2026-03-08.md",
        )

    def test_gate_timestamp_anchor_is_available(self) -> None:
        ts = int(time.time())
        self.assertGreater(ts, 0)


if __name__ == "__main__":
    unittest.main()
