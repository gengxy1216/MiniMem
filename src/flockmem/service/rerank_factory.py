from __future__ import annotations

from typing import Any

from flockmem.service.chat_model_rerank import ChatModelRerankProvider
from flockmem.config.settings import LiteSettings
from flockmem.service.local_rerank import LocalHeuristicRerankProvider
from flockmem.service.openai_rerank import OpenAIRerankProvider


def _norm(value: Any) -> str:
    return str(value or "").strip()


def build_rerank_provider(
    *, settings: LiteSettings, runtime_model_config: dict[str, Any]
) -> LocalHeuristicRerankProvider | OpenAIRerankProvider | ChatModelRerankProvider | None:
    provider = _norm(runtime_model_config.get("rerank_provider", settings.rerank_provider)).lower()
    if provider in {"none", "disabled", "rule", ""}:
        return None
    if provider in {"local", "local_rerank", "local-rerank"}:
        return LocalHeuristicRerankProvider(
            model=_norm(
                runtime_model_config.get("rerank_model", settings.local_rerank_model)
            )
            or settings.local_rerank_model,
            device=settings.local_rerank_device,
            batch_size=settings.local_rerank_batch_size,
            max_concurrency=settings.local_rerank_max_concurrency,
        )
    if provider == "chat_model":
        base_url = _norm(runtime_model_config.get("rerank_base_url", settings.rerank_base_url))
        api_key = _norm(runtime_model_config.get("rerank_api_key", settings.rerank_api_key))
        model = _norm(runtime_model_config.get("rerank_model", settings.rerank_model))
        if not base_url or not api_key or not model:
            return None
        return ChatModelRerankProvider(base_url=base_url, api_key=api_key, model=model)
    base_url = _norm(runtime_model_config.get("rerank_base_url", settings.rerank_base_url))
    api_key = _norm(runtime_model_config.get("rerank_api_key", settings.rerank_api_key))
    model = _norm(runtime_model_config.get("rerank_model", settings.rerank_model))
    if not base_url or not api_key or not model:
        return None
    return OpenAIRerankProvider(base_url=base_url, api_key=api_key, model=model)

