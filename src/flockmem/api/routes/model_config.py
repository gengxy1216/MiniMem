from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from flockmem.api.redaction import (
    is_redacted_literal,
    redact_sensitive,
    restore_redacted,
)
from flockmem.api.security import require_admin_access
from flockmem.service.extractor_factory import build_memory_extractor

router = APIRouter(
    prefix="/api/v1/model-config",
    tags=["model-config"],
    dependencies=[Depends(require_admin_access)],
)


class ModelRolePatch(BaseModel):
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None


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
    rerank_provider: str | None = None
    rerank_base_url: str | None = None
    rerank_api_key: str | None = None
    rerank_model: str | None = None
    chat: ModelRolePatch | None = None
    embedding: ModelRolePatch | None = None
    extractor: ModelRolePatch | None = None
    rerank: ModelRolePatch | None = None


@router.get("")
async def get_model_config(request: Request) -> dict:
    return {
        "status": "ok",
        "result": redact_sensitive(dict(request.app.state.runtime_model_config)),
    }


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
    role_to_keys = {
        "chat": {
            "provider": "chat_provider",
            "base_url": "chat_base_url",
            "api_key": "chat_api_key",
            "model": "chat_model",
        },
        "embedding": {
            "provider": "embedding_provider",
            "base_url": "embedding_base_url",
            "api_key": "embedding_api_key",
            "model": "embedding_model",
        },
        "extractor": {
            "provider": "extractor_provider",
            "base_url": "extractor_base_url",
            "api_key": "extractor_api_key",
            "model": "extractor_model",
        },
        "rerank": {
            "provider": "rerank_provider",
            "base_url": "rerank_base_url",
            "api_key": "rerank_api_key",
            "model": "rerank_model",
        },
    }
    should_refresh_extractor = False
    normalized_patch: dict[str, object] = {}
    incoming = payload.model_dump(exclude_none=True)
    for role, field_map in role_to_keys.items():
        block = incoming.pop(role, None)
        if not isinstance(block, dict):
            continue
        for src_key, dst_key in field_map.items():
            if src_key in block:
                incoming[dst_key] = block[src_key]

    for key, value in incoming.items():
        if key == "chat_provider" and str(value).strip() == "mock":
            continue
        if key == "chat_provider_options" and isinstance(value, dict):
            value = restore_redacted(
                value,
                request.app.state.runtime_model_config.get("chat_provider_options"),
            )
            value = {k: v for k, v in value.items() if str(k).strip() and str(k).strip() != "mock"}
        if key.endswith("_api_key") and is_redacted_literal(value):
            continue
        if key in extractor_keys:
            should_refresh_extractor = True
        if key in extractor_dependent_chat_keys:
            should_refresh_extractor = True
        normalized_patch[key] = value
    if normalized_patch:
        changed = request.app.state.config_repo.patch_model_config(
            settings=request.app.state.settings,
            patch=normalized_patch,
        )
        runtime_model_config = request.app.state.config_repo.get_runtime_model_config(
            request.app.state.settings
        )
        request.app.state.runtime_model_config.clear()
        request.app.state.runtime_model_config.update(runtime_model_config)
        request.app.state.chat_responder.base_url = str(
            runtime_model_config.get("chat_base_url", "")
        )
        request.app.state.chat_responder.api_key = str(
            runtime_model_config.get("chat_api_key", "")
        )
        request.app.state.chat_responder.model = str(
            runtime_model_config.get("chat_model", "")
        )
        request.app.state.chat_responder.provider = str(
            runtime_model_config.get("chat_provider", "openai")
        )
    if should_refresh_extractor:
        request.app.state.memory_service.extractor = build_memory_extractor(
            settings=request.app.state.settings,
            runtime_model_config=request.app.state.runtime_model_config,
        )
    return {
        "status": "ok",
        "result": {"updated": redact_sensitive(changed)},
    }


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
            "config": redact_sensitive(cfg),
        },
    }

