from __future__ import annotations

from typing import Any

from evermemos_lite.config.settings import LiteSettings
from evermemos_lite.service.extractor import (
    ChatModelMemoryExtractor,
    MemoryExtractor,
    OpenAIMemoryExtractor,
    RuleMemoryExtractor,
)


def build_memory_extractor(
    *, settings: LiteSettings, runtime_model_config: dict[str, Any]
) -> MemoryExtractor:
    provider = str(
        runtime_model_config.get("extractor_provider", settings.extractor_provider) or "rule"
    ).strip().lower()
    if provider == "rule":
        return RuleMemoryExtractor()

    if provider == "openai":
        base_url = str(
            runtime_model_config.get("extractor_base_url", settings.extractor_base_url)
            or settings.extractor_base_url
        ).strip()
        api_key = str(
            runtime_model_config.get("extractor_api_key", settings.extractor_api_key)
            or settings.extractor_api_key
        ).strip()
        model = str(
            runtime_model_config.get("extractor_model", settings.extractor_model)
            or settings.extractor_model
        ).strip()
        if base_url and api_key and model:
            return OpenAIMemoryExtractor(base_url=base_url, api_key=api_key, model=model)
        return RuleMemoryExtractor()

    base_url = str(
        runtime_model_config.get("extractor_base_url", settings.extractor_base_url)
        or settings.extractor_base_url
    ).strip()
    api_key = str(
        runtime_model_config.get("extractor_api_key", settings.extractor_api_key)
        or settings.extractor_api_key
    ).strip()
    model = str(
        runtime_model_config.get("extractor_model", settings.extractor_model)
        or settings.extractor_model
    ).strip()

    # Fallback to active chat provider config when extractor credentials are empty.
    provider_options = (
        runtime_model_config.get("chat_provider_options")
        if isinstance(runtime_model_config.get("chat_provider_options"), dict)
        else {}
    )
    active_chat_provider = str(
        runtime_model_config.get("chat_provider", settings.chat_provider) or settings.chat_provider
    ).strip()
    selected_provider = (
        provider_options.get(active_chat_provider, {})
        if isinstance(provider_options, dict)
        else {}
    )
    if not isinstance(selected_provider, dict):
        selected_provider = {}
    if not base_url:
        base_url = str(
            selected_provider.get("base_url")
            or runtime_model_config.get("chat_base_url")
            or settings.chat_base_url
        ).strip()
    if not api_key:
        api_key = str(
            selected_provider.get("api_key")
            or runtime_model_config.get("chat_api_key")
            or settings.chat_api_key
        ).strip()
    if not model:
        model = str(
            selected_provider.get("model")
            or runtime_model_config.get("chat_model")
            or settings.chat_model
        ).strip()

    if not base_url or not api_key or not model:
        return RuleMemoryExtractor()
    return ChatModelMemoryExtractor(base_url=base_url, api_key=api_key, model=model)
