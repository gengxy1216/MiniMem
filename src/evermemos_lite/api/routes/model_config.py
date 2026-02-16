from __future__ import annotations

import json

from fastapi import APIRouter, Request
from pydantic import BaseModel

from evermemos_lite.service.extractor_factory import build_memory_extractor

router = APIRouter(prefix="/api/v1/model-config", tags=["model-config"])


class ModelConfigPatch(BaseModel):
    chat_provider: str | None = None
    chat_base_url: str | None = None
    chat_api_key: str | None = None
    chat_model: str | None = None
    chat_provider_options: dict[str, dict[str, str]] | None = None
    embedding_provider: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str | None = None
    extractor_provider: str | None = None
    extractor_base_url: str | None = None
    extractor_api_key: str | None = None
    extractor_model: str | None = None


@router.get("")
async def get_model_config(request: Request) -> dict:
    return {"status": "ok", "result": dict(request.app.state.runtime_model_config)}


@router.put("")
async def update_model_config(request: Request, payload: ModelConfigPatch) -> dict:
    changed: dict[str, object] = {}
    extractor_keys = {
        "extractor_provider",
        "extractor_base_url",
        "extractor_api_key",
        "extractor_model",
    }
    extractor_dependent_chat_keys = {
        "chat_provider",
        "chat_base_url",
        "chat_api_key",
        "chat_model",
        "chat_provider_options",
    }
    should_refresh_extractor = False
    for key, value in payload.model_dump(exclude_none=True).items():
        if key == "chat_provider" and str(value).strip() == "mock":
            continue
        if key == "chat_provider_options" and isinstance(value, dict):
            value = {k: v for k, v in value.items() if str(k).strip() and str(k).strip() != "mock"}
        if key in extractor_keys:
            should_refresh_extractor = True
        if key in extractor_dependent_chat_keys:
            should_refresh_extractor = True
        request.app.state.runtime_model_config[key] = value
        if isinstance(value, (dict, list)):
            request.app.state.app_config_repo.upsert(
                key, json.dumps(value, ensure_ascii=False)
            )
        else:
            request.app.state.app_config_repo.upsert(key, str(value))
        changed[key] = value
    if should_refresh_extractor:
        request.app.state.memory_service.extractor = build_memory_extractor(
            settings=request.app.state.settings,
            runtime_model_config=request.app.state.runtime_model_config,
        )
    return {"status": "ok", "result": {"updated": changed}}


@router.post("/test")
async def test_model_config(request: Request) -> dict:
    cfg = dict(request.app.state.runtime_model_config)
    provider = str(cfg.get("chat_provider", "openai"))
    options = (
        cfg.get("chat_provider_options")
        if isinstance(cfg.get("chat_provider_options"), dict)
        else {}
    )
    selected = options.get(provider) if isinstance(options.get(provider), dict) else {}
    provider_reachable = bool(
        selected
        and str(selected.get("base_url", "")).strip()
        and str(selected.get("api_key", "")).strip()
        and str(selected.get("model", "")).strip()
    )
    extractor_provider = str(cfg.get("extractor_provider", "chat_model") or "chat_model")
    extractor_reachable = bool(
        str(cfg.get("extractor_base_url", "")).strip()
        and str(cfg.get("extractor_api_key", "")).strip()
        and str(cfg.get("extractor_model", "")).strip()
    )
    if extractor_provider.strip().lower() == "rule":
        extractor_reachable = True
    return {
        "status": "ok",
        "result": {
            "reachable": bool(provider_reachable),
            "provider": provider,
            "extractor_reachable": bool(extractor_reachable),
            "extractor_provider": extractor_provider,
            "config": cfg,
        },
    }
