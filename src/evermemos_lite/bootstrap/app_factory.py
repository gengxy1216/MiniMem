from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI

from evermemos_lite.api.routes.chat import router as chat_router
from evermemos_lite.api.routes.conversation_meta import router as conversation_meta_router
from evermemos_lite.api.routes.health import router as health_router
from evermemos_lite.api.routes.graph import router as graph_router
from evermemos_lite.api.routes.memory import router as memory_router
from evermemos_lite.api.routes.model_config import router as model_config_router
from evermemos_lite.api.routes.policy import router as policy_router
from evermemos_lite.api.routes.status import router as status_router
from evermemos_lite.api.routes.ui import router as ui_router
from evermemos_lite.config.profiles import PROFILE_PRESETS
from evermemos_lite.config.settings import LiteSettings
from evermemos_lite.infra.sqlite.app_config_repository import AppConfigRepository
from evermemos_lite.infra.graph.kuzu_store import KuzuGraphStore
from evermemos_lite.infra.sqlite.conversation_meta_repository import ConversationMetaRepository
from evermemos_lite.infra.sqlite.request_status_repository import RequestStatusRepository
from evermemos_lite.infra.runtime_policy.repository import RuntimePolicyRepository
from evermemos_lite.infra.sqlite.db import SQLiteEngine
from evermemos_lite.infra.sqlite.init_schema import init_schema
from evermemos_lite.infra.vector.lancedb_store import LanceVectorStore
from evermemos_lite.service.chat_responder import ChatResponder
from evermemos_lite.service.extractor_factory import build_memory_extractor
from evermemos_lite.service.formation_enhancer import ChatModelFormationEnhancer
from evermemos_lite.service.memory_service import MemoryService
from evermemos_lite.service.openai_embedding import OpenAIEmbeddingProvider
from evermemos_lite.service.policy_resolver import PolicyResolver
from evermemos_lite.service.query_rewriter import ChatModelQueryRewriter
from evermemos_lite.service.retrieval_verifier import ChatModelRetrievalVerifier
from evermemos_lite.service.retrieval_mode_selector import (
    OpenAIRetrievalModeSelector,
    RuleRetrievalModeSelector,
)


def create_app(settings: LiteSettings) -> FastAPI:
    engine = SQLiteEngine(settings.db_path)
    init_schema(engine)
    app_config_repo = AppConfigRepository(engine)
    config_keys = [
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
    ]
    persisted = app_config_repo.get_many(config_keys)
    raw_provider_options = persisted.get("chat_provider_options", "")
    provider_options: dict[str, dict[str, str]] = {}
    if raw_provider_options:
        try:
            parsed = json.loads(raw_provider_options)
            if isinstance(parsed, dict):
                provider_options = {
                    str(k): {
                        "base_url": str(v.get("base_url", "")),
                        "api_key": str(v.get("api_key", "")),
                        "model": str(v.get("model", "")),
                    }
                    for k, v in parsed.items()
                    if isinstance(v, dict)
                }
        except Exception:
            provider_options = {}
    provider_options.pop("mock", None)
    if not provider_options:
        provider_options = {
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
    persisted_chat_provider = str(
        persisted.get("chat_provider", settings.chat_provider)
    ).strip() or settings.chat_provider
    if persisted_chat_provider == "mock":
        persisted_chat_provider = settings.chat_provider
    if provider_options and persisted_chat_provider not in provider_options:
        persisted_chat_provider = next(iter(provider_options.keys()))

    runtime_model_config = {
        "chat_provider": persisted_chat_provider,
        "chat_base_url": persisted.get("chat_base_url", settings.chat_base_url),
        "chat_api_key": persisted.get("chat_api_key", settings.chat_api_key),
        "chat_model": persisted.get("chat_model", settings.chat_model),
        "chat_provider_options": provider_options,
        "embedding_provider": persisted.get(
            "embedding_provider", settings.embedding_provider
        ),
        "embedding_base_url": persisted.get(
            "embedding_base_url", settings.embedding_base_url
        ),
        "embedding_api_key": persisted.get(
            "embedding_api_key", settings.embedding_api_key
        ),
        "embedding_model": persisted.get("embedding_model", settings.embedding_model),
        "extractor_provider": persisted.get(
            "extractor_provider", settings.extractor_provider
        ),
        "extractor_base_url": persisted.get(
            "extractor_base_url", settings.extractor_base_url
        ),
        "extractor_api_key": persisted.get(
            "extractor_api_key", settings.extractor_api_key
        ),
        "extractor_model": persisted.get("extractor_model", settings.extractor_model),
    }
    runtime_policy_repo = RuntimePolicyRepository(engine=engine)
    policy_resolver = PolicyResolver(runtime_policy_repo)
    request_status_repo = RequestStatusRepository(engine)
    conversation_meta_repo = ConversationMetaRepository(engine)
    profile = PROFILE_PRESETS.get(settings.retrieval_profile, PROFILE_PRESETS["agentic"])
    if runtime_model_config["embedding_provider"] != "openai":
        raise ValueError("Only openai-compatible embedding provider is supported")
    embedding_provider = OpenAIEmbeddingProvider(
        base_url=runtime_model_config["embedding_base_url"],
        api_key=runtime_model_config["embedding_api_key"],
        model=runtime_model_config["embedding_model"],
    )

    extractor = build_memory_extractor(
        settings=settings, runtime_model_config=runtime_model_config
    )
    phase1_base_url = str(
        runtime_model_config.get("extractor_base_url")
        or runtime_model_config.get("chat_base_url")
        or settings.extractor_base_url
        or settings.chat_base_url
    ).strip()
    phase1_api_key = str(
        runtime_model_config.get("extractor_api_key")
        or runtime_model_config.get("chat_api_key")
        or settings.extractor_api_key
        or settings.chat_api_key
    ).strip()
    phase1_model = str(
        runtime_model_config.get("extractor_model")
        or runtime_model_config.get("chat_model")
        or settings.extractor_model
        or settings.chat_model
    ).strip()
    formation_enhancer = ChatModelFormationEnhancer(
        base_url=phase1_base_url,
        api_key=phase1_api_key,
        model=phase1_model,
        enabled=settings.phase1_enabled,
        boundary_enabled=settings.phase1_semantic_boundary_enabled,
        narrative_enabled=settings.phase1_narrative_enabled,
    )
    retrieval_verifier = ChatModelRetrievalVerifier(
        base_url=phase1_base_url,
        api_key=phase1_api_key,
        model=phase1_model,
        enabled=settings.phase2_agentic_verifier_enabled,
    )
    query_rewriter = ChatModelQueryRewriter(
        base_url=phase1_base_url,
        api_key=phase1_api_key,
        model=phase1_model,
        enabled=settings.phase3_query_rewriter_enabled,
    )

    vector_store = LanceVectorStore(
        settings.lancedb_dir,
        vector_dim=max(8, profile.vector_dim or 384),
        use_lancedb=settings.vector_lancedb_enabled,
        lance_persist_min_importance=settings.vector_lancedb_min_importance,
        index_type=settings.vector_index_type,
        index_metric=settings.vector_index_metric,
        hnsw_m=settings.vector_index_m,
        hnsw_ef_construction=settings.vector_index_ef_construction,
        search_nprobes=settings.vector_search_nprobes,
        search_ef=settings.vector_search_ef,
        search_refine_factor=settings.vector_search_refine_factor,
    )
    graph_store = KuzuGraphStore(db_dir=settings.graph_dir, enabled=settings.graph_enabled)
    memory_service = MemoryService(
        engine,
        vector_store,
        embedding_provider,
        extractor,
        formation_enhancer=formation_enhancer,
        semantic_boundary_min_confidence=settings.phase1_boundary_min_confidence,
        retrieval_verifier=retrieval_verifier,
        retrieval_verifier_min_confidence=settings.phase2_agentic_verifier_min_confidence,
        query_rewriter=query_rewriter,
        query_rewriter_min_confidence=settings.phase3_query_rewriter_min_confidence,
        phase4_reasoning_enabled=settings.phase4_reasoning_enabled,
        temporal_rerank_weight=settings.phase4_temporal_rerank_weight,
        multi_hop_max_queries=settings.phase4_multi_hop_max_queries,
        graph_store=graph_store,
        graph_top_k=settings.graph_top_k,
        graph_write_min_importance=settings.graph_write_min_importance,
        key_memory_importance_threshold=settings.key_memory_importance_threshold,
        vector_write_min_importance=settings.vector_write_min_importance,
        vector_embed_chunk_chars=settings.vector_embed_chunk_chars,
        vector_embed_max_chunks=settings.vector_embed_max_chunks,
        search_budget_factor=settings.search_budget_factor,
        search_min_probe_k=settings.search_min_probe_k,
        keyword_confident_best_score=settings.keyword_confident_best_score,
        keyword_confident_kth_score=settings.keyword_confident_kth_score,
        semantic_vector_budget_cap=settings.semantic_vector_budget_cap,
        semantic_keyword_budget_cap=settings.semantic_keyword_budget_cap,
        query_embed_cache_size=settings.query_embed_cache_size,
        query_embed_cache_ttl_sec=settings.query_embed_cache_ttl_sec,
        search_trace_enabled=settings.search_trace_enabled,
        search_trace_slow_ms=settings.search_trace_slow_ms,
    )
    chat_responder = ChatResponder(
        base_url=runtime_model_config["chat_base_url"],
        api_key=runtime_model_config["chat_api_key"],
        model=runtime_model_config["chat_model"],
        provider=runtime_model_config["chat_provider"],
    )
    rule_selector = RuleRetrievalModeSelector()
    if (
        settings.agent_policy_enabled
        and settings.agent_policy_provider == "openai"
        and settings.openai_api_key
    ):
        agent_selector = OpenAIRetrievalModeSelector(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.agent_policy_model,
        )
    else:
        agent_selector = rule_selector

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime_policy_repo.cleanup_expired()
        request_status_repo.cleanup_expired()
        yield

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.settings = settings
    app.state.sqlite_engine = engine
    app.state.app_config_repo = app_config_repo
    app.state.runtime_model_config = runtime_model_config
    app.state.runtime_policy_repo = runtime_policy_repo
    app.state.policy_resolver = policy_resolver
    app.state.request_status_repo = request_status_repo
    app.state.conversation_meta_repo = conversation_meta_repo
    app.state.memory_service = memory_service
    app.state.graph_store = graph_store
    app.state.chat_responder = chat_responder
    app.state.rule_retrieval_mode_selector = rule_selector
    app.state.agent_retrieval_mode_selector = agent_selector

    app.include_router(ui_router)
    app.include_router(health_router)
    app.include_router(memory_router)
    app.include_router(chat_router)
    app.include_router(graph_router)
    app.include_router(model_config_router)
    app.include_router(conversation_meta_router)
    app.include_router(policy_router)
    app.include_router(status_router)
    return app



