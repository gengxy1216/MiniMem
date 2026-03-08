from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from flockmem.api.routes.config_raw import router as config_raw_router
from flockmem.api.routes.chat import router as chat_router
from flockmem.api.routes.conversation_meta import router as conversation_meta_router
from flockmem.api.routes.health import router as health_router
from flockmem.api.routes.graph import router as graph_router
from flockmem.api.routes.ingest import router as ingest_router
from flockmem.api.routes.memory import router as memory_router
from flockmem.api.routes.model_config import router as model_config_router
from flockmem.api.routes.policy import router as policy_router
from flockmem.api.routes.status import router as status_router
from flockmem.api.routes.ui import router as ui_router
from flockmem.config.config_json import JsonConfigRepository
from flockmem.config.profiles import PROFILE_PRESETS
from flockmem.config.settings import LiteSettings
from flockmem.infra.graph.kuzu_store import KuzuGraphStore
from flockmem.infra.sqlite.conversation_meta_repository import ConversationMetaRepository
from flockmem.infra.sqlite.request_status_repository import RequestStatusRepository
from flockmem.infra.runtime_policy.repository import RuntimePolicyRepository
from flockmem.infra.sqlite.db import SQLiteEngine
from flockmem.infra.sqlite.init_schema import init_schema
from flockmem.infra.vector.lancedb_store import LanceVectorStore
from flockmem.service.chat_responder import ChatResponder
from flockmem.service.embedding_factory import build_embedding_provider
from flockmem.service.extractor_factory import build_memory_extractor
from flockmem.service.foresight_extractor import ChatModelForesightExtractor
from flockmem.service.formation_enhancer import ChatModelFormationEnhancer
from flockmem.service.memory_service import MemoryService
from flockmem.service.policy_resolver import PolicyResolver
from flockmem.service.query_rewriter import ChatModelQueryRewriter
from flockmem.service.rerank_factory import build_rerank_provider
from flockmem.service.retrieval_verifier import ChatModelRetrievalVerifier
from flockmem.service.retrieval_mode_selector import (
    OpenAIRetrievalModeSelector,
    RuleRetrievalModeSelector,
)


def create_app(settings: LiteSettings) -> FastAPI:
    config_repo = JsonConfigRepository(settings.config_path)
    settings = config_repo.get_effective_settings(settings)
    runtime_model_config = config_repo.get_runtime_model_config(settings)
    engine = SQLiteEngine(settings.db_path)
    init_schema(engine)
    runtime_policy_repo = RuntimePolicyRepository(engine=engine)
    policy_resolver = PolicyResolver(runtime_policy_repo)
    request_status_repo = RequestStatusRepository(engine)
    conversation_meta_repo = ConversationMetaRepository(engine)
    profile_name = str(settings.retrieval_profile or "").strip().lower()
    if settings.recall_mode and profile_name in {"", "agentic", "balanced", "hybrid"}:
        profile_name = "recall"
    fallback_profile = "recall" if settings.recall_mode else "agentic"
    profile = PROFILE_PRESETS.get(profile_name, PROFILE_PRESETS[fallback_profile])
    embedding_provider = build_embedding_provider(
        settings=settings, runtime_model_config=runtime_model_config
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
    foresight_extractor = ChatModelForesightExtractor(
        base_url=phase1_base_url,
        api_key=phase1_api_key,
        model=phase1_model,
    )
    rerank_provider = build_rerank_provider(
        settings=settings, runtime_model_config=runtime_model_config
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
    event_log_vector_store = LanceVectorStore(
        settings.lancedb_dir / "eventlog",
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
        rerank_provider=rerank_provider,
        rerank_trigger_k=settings.rerank_trigger_k,
        rerank_top_n=settings.rerank_top_n,
        rerank_timeout_ms=settings.rerank_timeout_ms,
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
        event_log_vector_store=event_log_vector_store,
        foresight_extractor=foresight_extractor,
        extract_max_retries=settings.extract_max_retries,
        recall_mode=settings.recall_mode,
        agentic_round_min_k=settings.agentic_round_min_k,
        agentic_round_max_k=settings.agentic_round_max_k,
        agentic_force_second_round=settings.agentic_force_second_round,
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
    app.state.config_repo = config_repo
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
    app.include_router(ingest_router)
    app.include_router(memory_router)
    app.include_router(config_raw_router)
    app.include_router(chat_router)
    app.include_router(graph_router)
    app.include_router(model_config_router)
    app.include_router(conversation_meta_router)
    app.include_router(policy_router)
    app.include_router(status_router)
    return app




