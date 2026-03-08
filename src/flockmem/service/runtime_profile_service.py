from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

_DEFAULT_RUNTIME_PROFILES = ("balanced", "recall", "keyword", "hybrid", "agentic")
_DEFAULT_COORDINATION_MODES = ("federated_acp", "inruntime_a2a")
_ENVELOPE_KEYS = (
    "coordination_mode",
    "coordination_id",
    "runtime_id",
    "agent_id",
    "subagent_id",
    "team_id",
    "session_id",
)
_DEFAULT_ADAPTER_KEYSETS = {
    "mcp": {"known_fields": ("runtime_profile", *_ENVELOPE_KEYS), "required_fields": ()},
    "hook": {"known_fields": ("runtime_profile", *_ENVELOPE_KEYS), "required_fields": ()},
    "plugin": {"known_fields": ("runtime_profile", *_ENVELOPE_KEYS), "required_fields": ()},
    "cli_bridge": {"known_fields": ("runtime_profile", *_ENVELOPE_KEYS), "required_fields": ()},
    "webhook_bridge": {
        "known_fields": ("runtime_profile", *_ENVELOPE_KEYS),
        "required_fields": (),
    },
}


def _default_schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "runtime_profile_schema.json"


def _normalize_token(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_str_set(raw: Any, fallback: tuple[str, ...]) -> set[str]:
    if not isinstance(raw, (list, tuple)):
        return {item.lower() for item in fallback}
    out: set[str] = set()
    for item in raw:
        text = _normalize_token(item)
        if text:
            out.add(text.lower())
    if out:
        return out
    return {item.lower() for item in fallback}


@lru_cache(maxsize=1)
def _load_default_schema() -> dict[str, Any]:
    path = _default_schema_path()
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def load_runtime_profile_schema(schema_path: str | Path | None = None) -> dict[str, Any]:
    if schema_path is None:
        return dict(_load_default_schema())
    path = Path(schema_path).resolve()
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def allowed_runtime_profiles(schema: Mapping[str, Any] | None = None) -> set[str]:
    source = schema if isinstance(schema, Mapping) else load_runtime_profile_schema()
    return _normalize_str_set(source.get("runtime_profiles"), _DEFAULT_RUNTIME_PROFILES)


def allowed_coordination_modes(schema: Mapping[str, Any] | None = None) -> set[str]:
    source = schema if isinstance(schema, Mapping) else load_runtime_profile_schema()
    return _normalize_str_set(source.get("coordination_modes"), _DEFAULT_COORDINATION_MODES)


def envelope_keys(schema: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    source = schema if isinstance(schema, Mapping) else load_runtime_profile_schema()
    raw = source.get("envelope_fields")
    if not isinstance(raw, (list, tuple)):
        return _ENVELOPE_KEYS
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        key = _normalize_token(item)
        if not key:
            continue
        key = key.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return tuple(out or _ENVELOPE_KEYS)


def normalize_runtime_profile(
    value: Any,
    *,
    strict: bool = True,
    schema: Mapping[str, Any] | None = None,
) -> str | None:
    token = _normalize_token(value)
    if token is None:
        return None
    profile = token.lower()
    if strict and profile not in allowed_runtime_profiles(schema):
        raise ValueError(f"invalid runtime_profile: {value}")
    return profile


def build_collective_envelope(
    *,
    coordination_mode: Any = None,
    coordination_id: Any = None,
    runtime_id: Any = None,
    agent_id: Any = None,
    subagent_id: Any = None,
    team_id: Any = None,
    session_id: Any = None,
    strict_coordination_mode: bool = True,
    schema: Mapping[str, Any] | None = None,
) -> dict[str, str | None]:
    mode = _normalize_token(coordination_mode)
    if mode is not None:
        mode = mode.lower()
        if strict_coordination_mode and mode not in allowed_coordination_modes(schema):
            raise ValueError(f"invalid coordination_mode: {coordination_mode}")
    return {
        "coordination_mode": mode,
        "coordination_id": _normalize_token(coordination_id),
        "runtime_id": _normalize_token(runtime_id),
        "agent_id": _normalize_token(agent_id),
        "subagent_id": _normalize_token(subagent_id),
        "team_id": _normalize_token(team_id),
        "session_id": _normalize_token(session_id),
    }


def _adapter_schema(adapter: str, schema: Mapping[str, Any] | None = None) -> dict[str, Any]:
    adapter_name = _normalize_token(adapter)
    if not adapter_name:
        raise ValueError("adapter is required")
    normalized = adapter_name.lower()
    source = schema if isinstance(schema, Mapping) else load_runtime_profile_schema()
    keysets = source.get("adapter_keysets")
    if isinstance(keysets, Mapping) and isinstance(keysets.get(normalized), Mapping):
        selected = keysets.get(normalized)
        assert isinstance(selected, Mapping)
        return {
            "known_fields": tuple(
                str(x).strip() for x in selected.get("known_fields", []) if str(x).strip()
            ),
            "required_fields": tuple(
                str(x).strip() for x in selected.get("required_fields", []) if str(x).strip()
            ),
        }
    fallback = _DEFAULT_ADAPTER_KEYSETS.get(normalized)
    if fallback is None:
        raise ValueError(f"unknown adapter keyset: {adapter}")
    return {
        "known_fields": tuple(fallback["known_fields"]),
        "required_fields": tuple(fallback["required_fields"]),
    }


def validate_adapter_payload(
    adapter: str,
    payload: Mapping[str, Any] | None,
    *,
    strict_runtime_profile: bool = True,
    strict_coordination_mode: bool = True,
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body = dict(payload or {})
    adapter_cfg = _adapter_schema(adapter, schema=schema)
    known_fields = set(adapter_cfg["known_fields"])
    required_fields = set(adapter_cfg["required_fields"])

    if "runtime_profile" in known_fields:
        body["runtime_profile"] = normalize_runtime_profile(
            body.get("runtime_profile"),
            strict=strict_runtime_profile,
            schema=schema,
        )

    if any(key in known_fields for key in envelope_keys(schema)):
        envelope = build_collective_envelope(
            coordination_mode=body.get("coordination_mode"),
            coordination_id=body.get("coordination_id"),
            runtime_id=body.get("runtime_id"),
            agent_id=body.get("agent_id"),
            subagent_id=body.get("subagent_id"),
            team_id=body.get("team_id"),
            session_id=body.get("session_id"),
            strict_coordination_mode=strict_coordination_mode,
            schema=schema,
        )
        for key, value in envelope.items():
            if key in known_fields:
                body[key] = value

    for key in required_fields:
        if _normalize_token(body.get(key)) is None:
            raise ValueError(f"{key} is required for adapter {adapter}")

    return body


@dataclass(frozen=True)
class RuntimeProfileValidationResult:
    valid: bool
    errors: tuple[str, ...]
    normalized: dict[str, Any]


class RuntimeProfileService:
    """Minimal runtime profile schema validator for mcp/hook/plugin/cli_bridge."""

    def __init__(self, schema: Mapping[str, Any] | None = None) -> None:
        self._schema: dict[str, Any] = (
            dict(schema) if isinstance(schema, Mapping) else load_runtime_profile_schema()
        )

    def validate(self, payload: Mapping[str, Any] | None) -> RuntimeProfileValidationResult:
        if not isinstance(payload, Mapping):
            return RuntimeProfileValidationResult(
                valid=False,
                errors=("profile must be an object",),
                normalized={},
            )

        raw = dict(payload)
        errors: list[str] = []
        runtime_id = _normalize_token(raw.get("runtime_id"))
        if runtime_id is None:
            errors.append("runtime_id is required")

        adapter_type = _normalize_token(raw.get("adapter_type"))
        adapter = adapter_type.lower() if adapter_type else None
        if adapter is None:
            errors.append("adapter_type is required")
        elif adapter not in {"mcp", "hook", "plugin", "cli_bridge", "webhook_bridge"}:
            errors.append("adapter_type must be one of: mcp/hook/plugin/cli_bridge/webhook_bridge")

        enabled = raw.get("enabled")
        if not isinstance(enabled, bool):
            errors.append("enabled must be boolean")

        launch = raw.get("launch")
        launch_obj = dict(launch) if isinstance(launch, Mapping) else {}
        if launch is not None and not isinstance(launch, Mapping):
            errors.append("launch must be an object")

        command = _normalize_token(launch_obj.get("command"))
        if adapter in {"mcp", "cli_bridge"} and command is None:
            errors.append(f"launch.command is required for adapter_type={adapter}")

        if "args" in launch_obj and not isinstance(launch_obj.get("args"), list):
            errors.append("launch.args must be an array")
        if "timeout_seconds" in launch_obj:
            timeout = launch_obj.get("timeout_seconds")
            if not isinstance(timeout, int) or timeout <= 0:
                errors.append("launch.timeout_seconds must be a positive integer")

        contracts = raw.get("contracts")
        contracts_obj = dict(contracts) if isinstance(contracts, Mapping) else {}
        if contracts is not None and not isinstance(contracts, Mapping):
            errors.append("contracts must be an object")

        has_contract = False
        for section in ("ingest", "context", "feedback"):
            if section not in contracts_obj:
                continue
            has_contract = True
            node = contracts_obj.get(section)
            if not isinstance(node, Mapping):
                errors.append(f"contracts.{section} must be an object")
                continue
            mode = _normalize_token(dict(node).get("mode"))
            if mode is None:
                errors.append(f"contracts.{section}.mode is required")

        if adapter in {"mcp", "hook", "plugin", "cli_bridge"} and not has_contract:
            errors.append(f"contracts must define at least one section for adapter_type={adapter}")

        try:
            normalized_runtime_profile = normalize_runtime_profile(
                raw.get("runtime_profile"),
                strict=True,
                schema=self._schema,
            )
        except ValueError as exc:
            errors.append(str(exc))
            normalized_runtime_profile = None

        try:
            envelope = build_collective_envelope(
                coordination_mode=raw.get("coordination_mode"),
                coordination_id=raw.get("coordination_id"),
                runtime_id=raw.get("runtime_id"),
                agent_id=raw.get("agent_id"),
                subagent_id=raw.get("subagent_id"),
                team_id=raw.get("team_id"),
                session_id=raw.get("session_id"),
                strict_coordination_mode=True,
                schema=self._schema,
            )
        except ValueError as exc:
            errors.append(str(exc))
            envelope = build_collective_envelope(
                coordination_mode=None,
                coordination_id=raw.get("coordination_id"),
                runtime_id=raw.get("runtime_id"),
                agent_id=raw.get("agent_id"),
                subagent_id=raw.get("subagent_id"),
                team_id=raw.get("team_id"),
                session_id=raw.get("session_id"),
                strict_coordination_mode=False,
                schema=self._schema,
            )

        normalized: dict[str, Any] = dict(raw)
        normalized["runtime_id"] = runtime_id
        normalized["adapter_type"] = adapter
        normalized["enabled"] = bool(enabled) if isinstance(enabled, bool) else enabled
        normalized["launch"] = launch_obj
        normalized["contracts"] = contracts_obj
        normalized["runtime_profile"] = normalized_runtime_profile
        normalized.update(envelope)
        return RuntimeProfileValidationResult(
            valid=not errors,
            errors=tuple(errors),
            normalized=normalized,
        )

    def validate_or_raise(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        result = self.validate(payload)
        if result.valid:
            return result.normalized
        raise ValueError("; ".join(result.errors))


def validate_runtime_profile(payload: Mapping[str, Any] | None) -> RuntimeProfileValidationResult:
    return RuntimeProfileService().validate(payload)


def validate_runtime_profile_or_raise(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    return RuntimeProfileService().validate_or_raise(payload)
