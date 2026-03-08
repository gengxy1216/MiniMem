from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from flockmem.config.config_json import (
    compact_model_config_for_storage,
    normalize_runtime_model_config,
)
from flockmem.config.settings import LiteSettings


def _to_str(value: Any) -> str:
    return str(value or "").strip()


def _default_minimem_config_path() -> Path:
    explicit = _to_str(os.getenv("LITE_CONFIG_PATH"))
    if explicit:
        return Path(explicit)
    config_dir = _to_str(os.getenv("LITE_CONFIG_DIR"))
    if config_dir:
        return Path(config_dir) / "config.json"
    return Path.home() / ".minimem" / "config.json"


def _snapshot_from_obj(obj: Any, provider_hint: str = "") -> dict[str, str] | None:
    if isinstance(obj, str):
        model = _to_str(obj)
        if not model:
            return None
        provider = _to_str(provider_hint)
        if not provider and "/" in model:
            provider = _to_str(model.split("/", 1)[0])
        return {
            "provider": provider,
            "base_url": "",
            "api_key": "",
            "model": model,
        }
    if not isinstance(obj, dict):
        return None
    provider = _to_str(obj.get("provider") or provider_hint)
    model_raw = obj.get("model")
    if isinstance(model_raw, (dict, list)):
        model_raw = ""
    name_raw = obj.get("name")
    if isinstance(name_raw, (dict, list)):
        name_raw = ""
    id_raw = obj.get("id")
    if isinstance(id_raw, (dict, list)):
        id_raw = ""
    model = _to_str(model_raw or name_raw or id_raw)
    if not model:
        return None
    if not provider and "/" in model:
        provider = _to_str(model.split("/", 1)[0])
    return {
        "provider": provider,
        "base_url": _to_str(
            obj.get("base_url") or obj.get("baseUrl") or obj.get("url") or obj.get("endpoint")
        ),
        "api_key": _to_str(obj.get("api_key") or obj.get("apiKey") or obj.get("token")),
        "model": model,
    }


def _provider_credentials_from_obj(obj: Any) -> dict[str, str]:
    if not isinstance(obj, dict):
        return {"base_url": "", "api_key": ""}
    return {
        "base_url": _to_str(
            obj.get("base_url") or obj.get("baseUrl") or obj.get("url") or obj.get("endpoint")
        ),
        "api_key": _to_str(obj.get("api_key") or obj.get("apiKey") or obj.get("token")),
    }


def _collect_provider_catalog(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def _merge(raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        for name, item in raw.items():
            key = _to_str(name)
            if not key or not isinstance(item, dict):
                continue
            out[key] = item

    _merge(cfg.get("providers"))
    models_cfg = cfg.get("models")
    if isinstance(models_cfg, dict):
        _merge(models_cfg.get("providers"))
    return out


def _provider_model_ids(provider_obj: Any) -> set[str]:
    out: set[str] = set()
    if not isinstance(provider_obj, dict):
        return out
    models = provider_obj.get("models")
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict):
                ident = _to_str(item.get("id") or item.get("name") or item.get("model"))
                if ident:
                    out.add(ident)
            elif isinstance(item, str):
                ident = _to_str(item)
                if ident:
                    out.add(ident)
    return out


def _resolve_snapshot_model_for_runtime(
    snapshot: dict[str, str], openclaw_config: dict[str, Any]
) -> dict[str, str]:
    out = {
        "provider": _to_str(snapshot.get("provider")),
        "base_url": _to_str(snapshot.get("base_url")),
        "api_key": _to_str(snapshot.get("api_key")),
        "model": _to_str(snapshot.get("model")),
    }
    provider = out["provider"]
    model = out["model"]
    if not provider or not model or "/" not in model:
        return out
    prefix, maybe_id = model.split("/", 1)
    if _to_str(prefix) != provider or not _to_str(maybe_id):
        return out
    provider_obj = _collect_provider_catalog(openclaw_config).get(provider)
    ids = _provider_model_ids(provider_obj)
    if maybe_id in ids:
        out["model"] = maybe_id
    return out


def _enrich_snapshot_with_provider(
    snapshot: dict[str, str], openclaw_config: dict[str, Any]
) -> dict[str, str]:
    out = {
        "provider": _to_str(snapshot.get("provider")),
        "base_url": _to_str(snapshot.get("base_url")),
        "api_key": _to_str(snapshot.get("api_key")),
        "model": _to_str(snapshot.get("model")),
    }
    provider = out["provider"]
    if not provider and "/" in out["model"]:
        provider = _to_str(out["model"].split("/", 1)[0])
        out["provider"] = provider
    if not provider:
        return out
    provider_obj = _collect_provider_catalog(openclaw_config).get(provider)
    creds = _provider_credentials_from_obj(provider_obj)
    if not out["base_url"]:
        out["base_url"] = _to_str(creds.get("base_url"))
    if not out["api_key"]:
        out["api_key"] = _to_str(creds.get("api_key"))
    return out


def to_public_primary_snapshot(snapshot: dict[str, Any]) -> dict[str, str]:
    return {
        "provider": _to_str(snapshot.get("provider")),
        "base_url": _to_str(snapshot.get("base_url")),
        "model": _to_str(snapshot.get("model")),
    }


def _get_in(path: list[str], root: dict[str, Any]) -> Any:
    current: Any = root
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current.get(key)
    return current


def detect_primary_model_snapshot(openclaw_config: dict[str, Any]) -> dict[str, str]:
    cfg = openclaw_config if isinstance(openclaw_config, dict) else {}
    provider_catalog = _collect_provider_catalog(cfg)
    known_provider_names = {
        _to_str(name).lower()
        for name in provider_catalog.keys()
        if _to_str(name)
    }
    known_provider_names.update({"openai", "siliconflow", "anthropic", "azure"})

    direct_paths = [
        ["model", "primary"],
        ["llm", "primary"],
        ["chat", "primary"],
        ["primary_model"],
        ["primaryModel"],
        ["default_model"],
        ["defaultModel"],
    ]
    for path in direct_paths:
        raw = _get_in(path, cfg)
        snap = _snapshot_from_obj(raw)
        if snap:
            return _enrich_snapshot_with_provider(snap, cfg)

    models = cfg.get("models")
    if isinstance(models, dict):
        primary_ref = models.get("primary")
        if isinstance(primary_ref, str):
            key = _to_str(primary_ref)
            snap = _snapshot_from_obj(models.get(key))
            if snap:
                return _enrich_snapshot_with_provider(snap, cfg)
        if isinstance(primary_ref, dict):
            snap = _snapshot_from_obj(primary_ref)
            if snap:
                return _enrich_snapshot_with_provider(snap, cfg)

    providers = provider_catalog
    if providers:
        provider_name = _to_str(
            cfg.get("primary_provider")
            or cfg.get("primaryProvider")
            or cfg.get("default_provider")
            or cfg.get("defaultProvider")
            or cfg.get("provider")
        )
        if provider_name and isinstance(providers.get(provider_name), dict):
            snap = _snapshot_from_obj(providers.get(provider_name), provider_name)
            if snap:
                return _enrich_snapshot_with_provider(snap, cfg)
        for name, item in providers.items():
            snap = _snapshot_from_obj(item, _to_str(name))
            if snap:
                return _enrich_snapshot_with_provider(snap, cfg)

    def _walk(obj: Any, provider_hint: str = "") -> dict[str, str] | None:
        if isinstance(obj, dict):
            for key in ("primary", "default", "main"):
                if key in obj:
                    snap = _snapshot_from_obj(obj.get(key), provider_hint)
                    if snap:
                        return snap
            snap = _snapshot_from_obj(obj, provider_hint)
            if snap and _to_str(obj.get("model")):
                return snap
            for key, value in obj.items():
                next_hint = provider_hint
                if _to_str(key).lower() in known_provider_names:
                    next_hint = _to_str(key)
                found = _walk(value, next_hint)
                if found:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _walk(item, provider_hint)
                if found:
                    return found
        return None

    found = _walk(cfg)
    if found:
        return _enrich_snapshot_with_provider(found, cfg)
    return {"provider": "", "base_url": "", "api_key": "", "model": ""}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except Exception:
            return {}
    except Exception:
        return {}
    try:
        if raw.startswith("\ufeff"):
            raw = raw.lstrip("\ufeff")
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return {}
    return {}


def _write_json_with_backup(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_minimem_bootstrap() -> dict[str, Any]:
    models: dict[str, Any] = {}
    try:
        settings = LiteSettings.from_env()
        models = compact_model_config_for_storage({}, settings)
    except Exception:
        models = {}
    return {
        "version": 1,
        "updated_at": int(time.time()),
        "settings": {},
        "models": models,
    }


def _normalize_models_for_sync(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    try:
        return normalize_runtime_model_config(raw, LiteSettings.from_env())
    except Exception:
        return dict(raw)


def _compact_models_for_sync(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    try:
        return compact_model_config_for_storage(raw, LiteSettings.from_env())
    except Exception:
        return dict(raw)


def _current_manually_overridden(
    models: dict[str, Any], last_applied: dict[str, Any]
) -> bool:
    if not isinstance(last_applied, dict) or not last_applied:
        return False
    for key, value in last_applied.items():
        if key not in models:
            continue
        if _to_str(models.get(key)) != _to_str(value):
            return True
    return False


def _apply_snapshot_to_models(
    models: dict[str, Any], snapshot: dict[str, str]
) -> dict[str, str]:
    provider = _to_str(snapshot.get("provider")) or _to_str(models.get("chat_provider")) or "openai"
    base_url = _to_str(snapshot.get("base_url")) or _to_str(models.get("chat_base_url"))
    api_key = _to_str(snapshot.get("api_key")) or _to_str(models.get("chat_api_key"))
    model = _to_str(snapshot.get("model")) or _to_str(models.get("chat_model"))

    models["chat_provider"] = provider
    models["chat_base_url"] = base_url
    models["chat_api_key"] = api_key
    models["chat_model"] = model
    models["extractor_provider"] = "chat_model"
    models["extractor_base_url"] = base_url
    models["extractor_api_key"] = api_key
    models["extractor_model"] = model

    options = models.get("chat_provider_options")
    if not isinstance(options, dict):
        options = {}
    selected = options.get(provider) if isinstance(options.get(provider), dict) else {}
    if not isinstance(selected, dict):
        selected = {}
    selected["base_url"] = base_url
    selected["api_key"] = api_key
    selected["model"] = model
    options[provider] = selected
    models["chat_provider_options"] = options

    return {
        "chat_provider": provider,
        "chat_base_url": base_url,
        "chat_api_key": api_key,
        "chat_model": model,
        "extractor_provider": "chat_model",
        "extractor_base_url": base_url,
        "extractor_api_key": api_key,
        "extractor_model": model,
    }


def sync_openclaw_primary_to_minimem_config(
    *,
    openclaw_config_path: Path,
    minimem_config_path: Path | None = None,
    inherit_primary_model: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    openclaw_path = Path(openclaw_config_path).resolve()
    if not openclaw_path.exists():
        raise FileNotFoundError(f"OpenClaw config not found: {openclaw_path}")

    target_path = (minimem_config_path or _default_minimem_config_path()).resolve()
    openclaw_cfg = _load_json(openclaw_path)
    snapshot = detect_primary_model_snapshot(openclaw_cfg)
    runtime_snapshot = _resolve_snapshot_model_for_runtime(snapshot, openclaw_cfg)
    model = _to_str(runtime_snapshot.get("model"))

    cfg = _load_json(target_path) if target_path.exists() else _build_minimem_bootstrap()
    if not isinstance(cfg, dict):
        cfg = _build_minimem_bootstrap()
    integration = cfg.setdefault("integration", {})
    if not isinstance(integration, dict):
        integration = {}
        cfg["integration"] = integration
    openclaw_meta = integration.setdefault("openclaw", {})
    if not isinstance(openclaw_meta, dict):
        openclaw_meta = {}
        integration["openclaw"] = openclaw_meta
    openclaw_meta["inherit_primary_model"] = bool(inherit_primary_model)
    openclaw_meta["primary_model_snapshot"] = snapshot
    openclaw_meta["runtime_model_snapshot"] = runtime_snapshot
    openclaw_meta["source_config_path"] = str(openclaw_path)

    models = _normalize_models_for_sync(cfg.get("models", {}))

    applied = False
    status = "noop"
    manual_override_detected = False

    if not inherit_primary_model:
        status = "inherit_disabled"
    elif not model:
        status = "primary_model_not_found"
    else:
        last_applied = (
            openclaw_meta.get("last_applied_models")
            if isinstance(openclaw_meta.get("last_applied_models"), dict)
            else {}
        )
        manual_override_detected = _current_manually_overridden(models, last_applied)
        if manual_override_detected and not force:
            status = "skipped_manual_override"
        else:
            last_applied_models = _apply_snapshot_to_models(models, runtime_snapshot)
            openclaw_meta["last_applied_models"] = last_applied_models
            applied = True
            status = "applied"

    openclaw_meta["last_sync_status"] = status
    openclaw_meta["last_sync_at"] = int(time.time())
    cfg["models"] = _compact_models_for_sync(models)
    cfg["updated_at"] = int(time.time())
    _write_json_with_backup(target_path, cfg)

    return {
        "ok": True,
        "status": status,
        "applied": bool(applied),
        "manual_override_detected": bool(manual_override_detected),
        "openclaw_config_path": str(openclaw_path),
        "minimem_config_path": str(target_path),
        "primary_model_snapshot": snapshot,
        "runtime_model_snapshot": runtime_snapshot,
    }

