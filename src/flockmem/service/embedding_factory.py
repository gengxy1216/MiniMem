from __future__ import annotations

from typing import Any

from flockmem.config.settings import LiteSettings
from flockmem.service.local_embedding import LocalHashEmbeddingProvider
from flockmem.service.openai_embedding import OpenAIEmbeddingProvider


def _norm(value: Any) -> str:
    return str(value or "").strip()


def build_embedding_provider(
    *, settings: LiteSettings, runtime_model_config: dict[str, Any]
) -> OpenAIEmbeddingProvider | LocalHashEmbeddingProvider:
    provider = _norm(
        runtime_model_config.get("embedding_provider", settings.embedding_provider)
    ).lower()
    model = _norm(runtime_model_config.get("embedding_model", settings.embedding_model))
    if provider in {"local", "local_hash", "local-hash"}:
        return LocalHashEmbeddingProvider(
            model=model or settings.local_embedding_model,
            device=settings.local_embedding_device,
            batch_size=settings.local_embedding_batch_size,
            max_concurrency=settings.local_embedding_max_concurrency,
        )
    return OpenAIEmbeddingProvider(
        base_url=_norm(
            runtime_model_config.get("embedding_base_url", settings.embedding_base_url)
        ),
        api_key=_norm(
            runtime_model_config.get("embedding_api_key", settings.embedding_api_key)
        ),
        model=model,
    )

