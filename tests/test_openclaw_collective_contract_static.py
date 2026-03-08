from __future__ import annotations

from pathlib import Path


PLUGIN_PATH = (
    Path(__file__).resolve().parents[1]
    / "integrations"
    / "openclaw-plugin"
    / "index.ts"
)


def test_openclaw_collective_tools_and_envelope_fields_are_present() -> None:
    raw = PLUGIN_PATH.read_text(encoding="utf-8")

    # Keep existing compatibility surface.
    assert 'name: "minimem_memory_write"' in raw
    assert 'name: "minimem_memory_retrieve"' in raw

    # New collective integration surface.
    assert 'name: "minimem_collective_ingest"' in raw
    assert 'name: "minimem_collective_context"' in raw
    assert 'name: "minimem_collective_feedback"' in raw
    assert '"/api/v1/collective/ingest"' in raw
    assert '"/api/v1/collective/context"' in raw
    assert '"/api/v1/collective/feedback"' in raw

    # Envelope passthrough fields.
    for field in (
        "coordination_mode",
        "coordination_id",
        "runtime_id",
        "agent_id",
        "subagent_id",
        "team_id",
        "session_id",
    ):
        assert field in raw
