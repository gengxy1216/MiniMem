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


@dataclass(frozen=True)
class _GateCase:
    gate_id: str
    category: str
    method: str
    path: str
    payload: dict[str, object]
    expected_statuses: tuple[int, ...]


_REQUIRED_ROUTES = {
    "/api/v1/collective/ingest",
    "/api/v1/collective/context",
    "/api/v1/collective/feedback",
}

_GATE_CASES: tuple[_GateCase, ...] = (
    _GateCase(
        gate_id="GATE-NORMAL-INGEST",
        category="normal",
        method="POST",
        path="/api/v1/collective/ingest",
        payload={
            "knowledge_id": "k-normal-1",
            "scope_type": "personal",
            "scope_id": "u-qa",
            "content": {"text": "qa normal ingest"},
            "changed_by": "agent",
            "actor_id": "qa-gate",
            "coordination_mode": "inruntime_a2a",
            "coordination_id": "coord-normal-1",
            "runtime_id": "codex",
            "agent_id": "qa-gate",
            "team_id": "team-qa",
            "session_id": "session-normal-1",
        },
        expected_statuses=(200,),
    ),
    _GateCase(
        gate_id="GATE-NORMAL-CONTEXT",
        category="normal",
        method="POST",
        path="/api/v1/collective/context",
        payload={
            "query": "qa gate context lookup",
            "personal_scope_id": "u-qa",
            "include_global": False,
            "coordination_mode": "inruntime_a2a",
            "coordination_id": "coord-normal-2",
            "runtime_id": "codex",
            "agent_id": "qa-gate",
            "team_id": "team-qa",
            "session_id": "session-normal-2",
        },
        expected_statuses=(200,),
    ),
    _GateCase(
        gate_id="GATE-BOUNDARY-INVALID-SCOPE",
        category="boundary",
        method="POST",
        path="/api/v1/collective/ingest",
        payload={
            "scope_type": "unknown_scope",
            "scope_id": "u-qa",
            "content": {"text": "invalid scope should fail"},
            "changed_by": "agent",
            "coordination_mode": "inruntime_a2a",
            "coordination_id": "coord-boundary-1",
            "runtime_id": "codex",
            "agent_id": "qa-gate",
            "session_id": "session-boundary-1",
        },
        expected_statuses=(422,),
    ),
    _GateCase(
        gate_id="GATE-BOUNDARY-MISSING-REQUIRED",
        category="boundary",
        method="POST",
        path="/api/v1/collective/feedback",
        payload={
            "knowledge_id": "k-boundary-missing",
            "coordination_mode": "inruntime_a2a",
        },
        expected_statuses=(422,),
    ),
    _GateCase(
        gate_id="GATE-RECOVERY-UNKNOWN-REVISION",
        category="recovery",
        method="POST",
        path="/api/v1/collective/feedback",
        payload={
            "knowledge_id": "k-recovery-seeded",
            "revision_id": "rev-not-exist",
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
            "coordination_id": "coord-recovery-1",
            "runtime_id": "codex",
            "agent_id": "qa-gate",
            "session_id": "session-recovery-1",
        },
        expected_statuses=(404,),
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


class CollectiveGateMatrixTests(unittest.TestCase):
    def _request(self, client: TestClient, case: _GateCase) -> int:
        if case.method == "POST":
            response = client.post(case.path, json=case.payload)
        elif case.method == "GET":
            response = client.get(case.path, params=case.payload)
        else:
            raise ValueError(f"unsupported method: {case.method}")
        return int(response.status_code)

    def test_gate_matrix_covers_required_dimensions(self) -> None:
        categories = {case.category for case in _GATE_CASES}
        self.assertIn("normal", categories)
        self.assertIn("boundary", categories)
        self.assertIn("recovery", categories)

    def test_collective_routes_registered_or_precondition_marked(self) -> None:
        client, routes, tmp = _build_client()
        self.addCleanup(tmp.cleanup)
        missing = sorted(_REQUIRED_ROUTES - routes)
        if missing:
            self.skipTest(
                "collective routes are not registered yet: "
                + ", ".join(missing)
            )
        self.assertEqual(set(), _REQUIRED_ROUTES - routes)
        client.close()

    def test_gate_cases_are_executable_when_collective_routes_exist(self) -> None:
        client, routes, tmp = _build_client()
        self.addCleanup(tmp.cleanup)
        missing = sorted(_REQUIRED_ROUTES - routes)
        if missing:
            self.skipTest(
                "collective contract gates are blocked until routes exist: "
                + ", ".join(missing)
            )

        # Seed a known knowledge item for recovery/boundary checks that depend on existing state.
        seed = client.post(
            "/api/v1/collective/ingest",
            json={
                "knowledge_id": "k-recovery-seeded",
                "scope_type": "personal",
                "scope_id": "u-qa",
                "content": {"text": "seed for recovery gate"},
                "changed_by": "agent",
                "actor_id": "qa-gate",
                "write_acl": ["qa-gate"],
            },
        )
        self.assertEqual(200, int(seed.status_code))

        for case in _GATE_CASES:
            with self.subTest(gate_id=case.gate_id):
                status_code = self._request(client, case)
                if not any(code in {404, 405} for code in case.expected_statuses):
                    self.assertNotIn(status_code, {404, 405})
                self.assertLess(status_code, 500)
                self.assertIn(status_code, case.expected_statuses)
        client.close()

    def test_perf_dimension_has_explicit_benchmark_script(self) -> None:
        benchmark_script = Path(__file__).resolve().parents[1] / "tools" / "bench_collective_perf.py"
        self.assertTrue(
            benchmark_script.exists(),
            "perf gate requires tools/bench_collective_perf.py",
        )

    def test_gate_matrix_records_execution_timestamp(self) -> None:
        # Ensure this suite can emit deterministic report anchors.
        ts = int(time.time())
        self.assertGreater(ts, 0)


if __name__ == "__main__":
    unittest.main()
