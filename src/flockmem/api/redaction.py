from __future__ import annotations

from typing import Any

_SENSITIVE_FRAGMENTS = ("api_key", "token", "password", "secret")
REDACTED_LITERAL = "[REDACTED]"


def _is_sensitive_key(key: str) -> bool:
    lowered = str(key or "").strip().lower()
    return any(fragment in lowered for fragment in _SENSITIVE_FRAGMENTS)


def _mask_secret(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return REDACTED_LITERAL


def redact_sensitive(data: Any) -> Any:
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            if _is_sensitive_key(key):
                out[str(key)] = _mask_secret(value)
                continue
            out[str(key)] = redact_sensitive(value)
        return out
    if isinstance(data, list):
        return [redact_sensitive(item) for item in data]
    return data


def is_redacted_literal(value: Any) -> bool:
    return str(value or "").strip() == REDACTED_LITERAL


def restore_redacted(candidate: Any, baseline: Any) -> Any:
    if isinstance(candidate, dict):
        baseline_map = baseline if isinstance(baseline, dict) else {}
        out: dict[str, Any] = {}
        for key, value in candidate.items():
            key_text = str(key)
            base_value = baseline_map.get(key_text)
            if _is_sensitive_key(key_text) and is_redacted_literal(value):
                out[key_text] = base_value
                continue
            out[key_text] = restore_redacted(value, base_value)
        return out
    if isinstance(candidate, list):
        baseline_list = baseline if isinstance(baseline, list) else []
        out_list: list[Any] = []
        for idx, item in enumerate(candidate):
            base_item = baseline_list[idx] if idx < len(baseline_list) else None
            out_list.append(restore_redacted(item, base_item))
        return out_list
    return candidate
