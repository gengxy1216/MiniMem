from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

from flockmem.config.settings import LiteSettings

CONFIG_VERSION = 1

MODEL_CONFIG_KEYS = [
    "chat_provider",
    "chat_base_url",
    "chat_api_key",
    "chat_model",
    "chat_provider_options",
    "embedding_provider",
    "embedding_base_url",
    "embedding_api_key",
    "embedding_model",
    "extractor_provider",
    "extractor_base_url",
    "extractor_api_key",
    "extractor_model",
    "rerank_provider",
    "rerank_base_url",
    "rerank_api_key",
    "rerank_model",
]

MODEL_ROLE_KEY_MAP: dict[str, tuple[str, str, str, str]] = {
    "chat": ("chat_provider", "chat_base_url", "chat_api_key", "chat_model"),
    "embedding": (
        "embedding_provider",
        "embedding_base_url",
        "embedding_api_key",
        "embedding_model",
    ),
    "extractor": (
        "extractor_provider",
        "extractor_base_url",
        "extractor_api_key",
        "extractor_model",
    ),
    "rerank": (
        "rerank_provider",
        "rerank_base_url",
        "rerank_api_key",
        "rerank_model",
    ),
}


def _to_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if not text:
        return fallback
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return fallback


def _coerce_value(raw: Any, default: Any) -> Any:
    if isinstance(default, Path):
        text = str(raw or "").strip()
        if not text:
            return default
        return Path(text).resolve()
    if isinstance(default, bool):
        return _to_bool(raw, default)
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(raw)
        except Exception:
            return default
    if isinstance(default, float):
        try:
            return float(raw)
        except Exception:
            return default
    text = str(raw if raw is not None else default)
    return text


def _serialize_settings(settings: LiteSettings) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in asdict(settings).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def _deserialize_settings(raw: dict[str, Any], fallback: LiteSettings) -> LiteSettings:
    updates: dict[str, Any] = {}
    for f in fields(LiteSettings):
        if f.name not in raw:
            continue
        current = getattr(fallback, f.name)
        updates[f.name] = _coerce_value(raw.get(f.name), current)
    if not updates:
        return fallback
    return replace(fallback, **updates)


def _build_default_provider_options(settings: LiteSettings) -> dict[str, dict[str, str]]:
    return {
        "openai": {
            "base_url": settings.chat_base_url,
            "api_key": settings.chat_api_key,
            "model": settings.chat_model,
        },
        "siliconflow": {
            "base_url": settings.chat_base_url,
            "api_key": settings.chat_api_key,
            "model": settings.chat_model,
        },
    }


def _normalize_provider_options(raw: Any, settings: LiteSettings) -> dict[str, dict[str, str]]:
    if not isinstance(raw, dict):
        raw = {}
    out: dict[str, dict[str, str]] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name or name == "mock" or not isinstance(value, dict):
            continue
        out[name] = {
            "base_url": str(value.get("base_url", "")).strip(),
            "api_key": str(value.get("api_key", "")).strip(),
            "model": str(value.get("model", "")).strip(),
        }
    if not out:
        out = _build_default_provider_options(settings)
    return out


def _apply_structured_block_to_flat(models: dict[str, Any], role: str, raw: Any) -> None:
    keys = MODEL_ROLE_KEY_MAP.get(role)
    if not keys or not isinstance(raw, dict):
        return
    provider_key, base_url_key, api_key_key, model_key = keys
    if "provider" in raw:
        models[provider_key] = str(raw.get("provider") or "").strip()
    if "base_url" in raw:
        models[base_url_key] = str(raw.get("base_url") or "").strip()
    if "api_key" in raw:
        models[api_key_key] = str(raw.get("api_key") or "").strip()
    if "model" in raw:
        models[model_key] = str(raw.get("model") or "").strip()


def _set_structured_blocks(models: dict[str, Any]) -> None:
    for role, keys in MODEL_ROLE_KEY_MAP.items():
        provider_key, base_url_key, api_key_key, model_key = keys
        models[role] = {
            "provider": str(models.get(provider_key, "") or "").strip(),
            "base_url": str(models.get(base_url_key, "") or "").strip(),
            "api_key": str(models.get(api_key_key, "") or "").strip(),
            "model": str(models.get(model_key, "") or "").strip(),
        }


def _default_runtime_model_config(settings: LiteSettings) -> dict[str, Any]:
    out = {
        "chat_provider": settings.chat_provider,
        "chat_base_url": settings.chat_base_url,
        "chat_api_key": settings.chat_api_key,
        "chat_model": settings.chat_model,
        "chat_provider_options": _build_default_provider_options(settings),
        "embedding_provider": settings.embedding_provider,
        "embedding_base_url": settings.embedding_base_url,
        "embedding_api_key": settings.embedding_api_key,
        "embedding_model": settings.embedding_model,
        "extractor_provider": settings.extractor_provider,
        "extractor_base_url": settings.extractor_base_url,
        "extractor_api_key": settings.extractor_api_key,
        "extractor_model": settings.extractor_model,
        "rerank_provider": settings.rerank_provider,
        "rerank_base_url": settings.rerank_base_url,
        "rerank_api_key": settings.rerank_api_key,
        "rerank_model": settings.rerank_model,
    }
    _set_structured_blocks(out)
    return out


def _normalize_prefixed_model_with_provider_options(
    out: dict[str, Any],
    *,
    role: str,
    provider_options: dict[str, dict[str, str]],
) -> None:
    keys = MODEL_ROLE_KEY_MAP.get(role)
    if not keys:
        return
    provider_key, _, _, model_key = keys
    provider = str(out.get(provider_key, "")).strip()
    model = str(out.get(model_key, "")).strip()
    if not provider or not model or "/" not in model:
        return
    prefix, model_id = model.split("/", 1)
    prefix = str(prefix).strip()
    model_id = str(model_id).strip()
    if not prefix or not model_id or prefix != provider:
        return
    selected = provider_options.get(provider) if isinstance(provider_options, dict) else {}
    if not isinstance(selected, dict):
        return
    known_model = str(selected.get("model", "")).strip()
    if known_model and known_model == model_id:
        out[model_key] = model_id


def _normalize_runtime_model_config(raw: Any, settings: LiteSettings) -> dict[str, Any]:
    default_cfg = _default_runtime_model_config(settings)
    source = raw if isinstance(raw, dict) else {}
    out = dict(default_cfg)
    for role in MODEL_ROLE_KEY_MAP:
        _apply_structured_block_to_flat(out, role, source.get(role))
    for key in MODEL_CONFIG_KEYS:
        if key not in source:
            continue
        if key == "chat_provider_options":
            out[key] = _normalize_provider_options(source.get(key), settings)
            continue
        out[key] = str(source.get(key, out[key]) or out[key]).strip()
    if out.get("chat_provider") == "mock":
        out["chat_provider"] = settings.chat_provider
    provider_options = _normalize_provider_options(out.get("chat_provider_options"), settings)
    chat_provider = str(out.get("chat_provider", "")).strip()
    if chat_provider and chat_provider not in provider_options:
        provider_options[chat_provider] = {
            "base_url": str(out.get("chat_base_url", "")).strip(),
            "api_key": str(out.get("chat_api_key", "")).strip(),
            "model": str(out.get("chat_model", "")).strip(),
        }
    out["chat_provider_options"] = provider_options
    if not chat_provider:
        out["chat_provider"] = next(iter(provider_options.keys()))
    if str(out.get("extractor_provider", "")).strip().lower() == "chat_model":
        out["extractor_base_url"] = str(
            out.get("extractor_base_url") or out.get("chat_base_url") or ""
        ).strip()
        out["extractor_api_key"] = str(
            out.get("extractor_api_key") or out.get("chat_api_key") or ""
        ).strip()
        out["extractor_model"] = str(
            out.get("extractor_model") or out.get("chat_model") or ""
        ).strip()
    if str(out.get("rerank_provider", "")).strip().lower() == "chat_model":
        out["rerank_base_url"] = str(
            out.get("rerank_base_url") or out.get("chat_base_url") or ""
        ).strip()
        out["rerank_api_key"] = str(
            out.get("rerank_api_key") or out.get("chat_api_key") or ""
        ).strip()
        out["rerank_model"] = str(out.get("rerank_model") or out.get("chat_model") or "").strip()
    _normalize_prefixed_model_with_provider_options(
        out, role="chat", provider_options=provider_options
    )
    _normalize_prefixed_model_with_provider_options(
        out, role="extractor", provider_options=provider_options
    )
    _normalize_prefixed_model_with_provider_options(
        out, role="rerank", provider_options=provider_options
    )
    _set_structured_blocks(out)
    return out


def _compact_model_config_for_storage(raw: Any, settings: LiteSettings) -> dict[str, Any]:
    runtime_cfg = _normalize_runtime_model_config(raw, settings)
    out: dict[str, Any] = {}
    for role, keys in MODEL_ROLE_KEY_MAP.items():
        provider_key, base_url_key, api_key_key, model_key = keys
        role_obj = runtime_cfg.get(role)
        if isinstance(role_obj, dict):
            out[role] = {
                "provider": str(role_obj.get("provider", "")).strip(),
                "base_url": str(role_obj.get("base_url", "")).strip(),
                "api_key": str(role_obj.get("api_key", "")).strip(),
                "model": str(role_obj.get("model", "")).strip(),
            }
            continue
        out[role] = {
            "provider": str(runtime_cfg.get(provider_key, "")).strip(),
            "base_url": str(runtime_cfg.get(base_url_key, "")).strip(),
            "api_key": str(runtime_cfg.get(api_key_key, "")).strip(),
            "model": str(runtime_cfg.get(model_key, "")).strip(),
        }
    out["chat_provider_options"] = _normalize_provider_options(
        runtime_cfg.get("chat_provider_options"), settings
    )
    return out


def normalize_runtime_model_config(raw: Any, settings: LiteSettings) -> dict[str, Any]:
    return _normalize_runtime_model_config(raw, settings)


def compact_model_config_for_storage(raw: Any, settings: LiteSettings) -> dict[str, Any]:
    return _compact_model_config_for_storage(raw, settings)


def _build_default_payload(settings: LiteSettings) -> dict[str, Any]:
    now = int(time.time())
    return {
        "version": CONFIG_VERSION,
        "updated_at": now,
        "settings": _serialize_settings(settings),
        "models": _compact_model_config_for_storage({}, settings),
    }


class JsonConfigRepository:
    def __init__(self, config_path: Path) -> None:
        self.config_path = Path(config_path).resolve()
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def ensure(self, bootstrap_settings: LiteSettings) -> dict[str, Any]:
        existing = self.load()
        if existing is not None:
            normalized = self._normalize_payload(existing, bootstrap_settings)
            if normalized != existing:
                self.save(normalized)
            return normalized
        legacy = self._load_legacy_payload(bootstrap_settings)
        if legacy is not None:
            migrated = self._normalize_payload(legacy, bootstrap_settings)
            self.save(migrated)
            return migrated
        payload = _build_default_payload(bootstrap_settings)
        self.save(payload)
        return payload

    def load(self) -> dict[str, Any] | None:
        if not self.config_path.exists():
            return None
        try:
            raw = self.config_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def save(self, payload: dict[str, Any]) -> None:
        doc = dict(payload)
        doc["updated_at"] = int(time.time())
        if self.config_path.exists():
            backup_path = self.config_path.with_suffix(self.config_path.suffix + ".bak")
            shutil.copy2(self.config_path, backup_path)
        self.config_path.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def get_effective_settings(self, bootstrap_settings: LiteSettings) -> LiteSettings:
        payload = self.ensure(bootstrap_settings)
        return _deserialize_settings(payload.get("settings", {}), bootstrap_settings)

    def get_runtime_model_config(self, settings: LiteSettings) -> dict[str, Any]:
        payload = self.ensure(settings)
        return _normalize_runtime_model_config(payload.get("models", {}), settings)

    def get_raw_config(self, bootstrap_settings: LiteSettings) -> dict[str, Any]:
        return self.ensure(bootstrap_settings)

    def patch_model_config(
        self,
        *,
        settings: LiteSettings,
        patch: dict[str, Any],
    ) -> dict[str, Any]:
        payload = self.ensure(settings)
        models = _normalize_runtime_model_config(payload.get("models", {}), settings)
        settings_doc = payload.get("settings", {})
        if not isinstance(settings_doc, dict):
            settings_doc = _serialize_settings(settings)
        changed: dict[str, Any] = {}
        sync_keys = {
            "chat_provider",
            "chat_base_url",
            "chat_api_key",
            "chat_model",
            "embedding_provider",
            "embedding_base_url",
            "embedding_api_key",
            "embedding_model",
            "extractor_provider",
            "extractor_base_url",
            "extractor_api_key",
            "extractor_model",
            "rerank_provider",
            "rerank_base_url",
            "rerank_api_key",
            "rerank_model",
        }
        for key, value in patch.items():
            if key not in MODEL_CONFIG_KEYS:
                continue
            if key == "chat_provider_options":
                models[key] = _normalize_provider_options(value, settings)
                changed[key] = models[key]
                continue
            models[key] = str(value).strip()
            changed[key] = models[key]
            if key in sync_keys:
                settings_doc[key] = models[key]
        payload["settings"] = settings_doc
        payload["models"] = _compact_model_config_for_storage(models, settings)
        self.save(payload)
        return changed

    def replace_raw_config(
        self,
        *,
        bootstrap_settings: LiteSettings,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = self._normalize_payload(payload, bootstrap_settings)
        self.save(normalized)
        return normalized

    def _normalize_payload(
        self, payload: dict[str, Any], bootstrap_settings: LiteSettings
    ) -> dict[str, Any]:
        out = dict(payload)
        out["version"] = int(out.get("version", CONFIG_VERSION))
        settings_doc = out.get("settings", {})
        if not isinstance(settings_doc, dict):
            settings_doc = {}
        effective_settings = _deserialize_settings(settings_doc, bootstrap_settings)
        out["settings"] = _serialize_settings(effective_settings)
        out["models"] = _compact_model_config_for_storage(
            out.get("models", {}), effective_settings
        )
        return out

    def _load_legacy_payload(self, bootstrap_settings: LiteSettings) -> dict[str, Any] | None:
        legacy_path = (Path(bootstrap_settings.data_dir) / "config.json").resolve()
        if legacy_path == self.config_path:
            return None
        if not legacy_path.exists():
            return None
        try:
            raw = legacy_path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
        return None

