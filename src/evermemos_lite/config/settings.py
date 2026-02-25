from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_data_dir() -> Path:
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA")
        if base:
            return Path(base) / "MiniMem"
        return Path.home() / "AppData" / "Local" / "MiniMem"
    xdg = os.getenv("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "minimem"
    return Path.home() / ".local" / "share" / "minimem"


@dataclass(frozen=True)
class LiteSettings:
    app_name: str
    host: str
    port: int
    data_dir: Path
    db_path: Path
    lancedb_dir: Path
    graph_dir: Path
    retrieval_profile: str
    request_status_ttl_sec: int
    nonce_ttl_sec: int
    openai_base_url: str
    openai_api_key: str
    chat_base_url: str
    chat_api_key: str
    chat_model: str
    chat_provider: str
    embedding_base_url: str
    embedding_api_key: str
    embedding_provider: str
    embedding_model: str
    extractor_provider: str
    extractor_base_url: str
    extractor_api_key: str
    extractor_model: str
    phase1_enabled: bool
    phase1_semantic_boundary_enabled: bool
    phase1_narrative_enabled: bool
    phase1_boundary_min_confidence: float
    phase2_agentic_verifier_enabled: bool
    phase2_agentic_verifier_min_confidence: float
    phase3_query_rewriter_enabled: bool
    phase3_query_rewriter_min_confidence: float
    phase4_reasoning_enabled: bool
    phase4_temporal_rerank_weight: float
    phase4_multi_hop_max_queries: int
    agent_policy_provider: str
    agent_policy_model: str
    agent_policy_enabled: bool
    graph_enabled: bool
    graph_top_k: int
    graph_write_min_importance: float
    key_memory_importance_threshold: float
    vector_lancedb_enabled: bool
    vector_lancedb_min_importance: float
    vector_write_min_importance: float
    vector_index_type: str
    vector_index_metric: str
    vector_index_m: int
    vector_index_ef_construction: int
    vector_search_nprobes: int
    vector_search_ef: int
    vector_search_refine_factor: int
    vector_embed_chunk_chars: int
    vector_embed_max_chunks: int
    search_budget_factor: int
    search_min_probe_k: int
    keyword_confident_best_score: float
    keyword_confident_kth_score: float
    semantic_vector_budget_cap: int
    semantic_keyword_budget_cap: int
    query_embed_cache_size: int
    query_embed_cache_ttl_sec: int
    search_trace_enabled: bool
    search_trace_slow_ms: int

    @classmethod
    def from_env(cls) -> "LiteSettings":
        data_dir_raw = os.getenv("LITE_DATA_DIR")
        if not data_dir_raw:
            data_dir_raw = str(_default_data_dir())
        data_dir = Path(data_dir_raw).resolve()
        db_path = Path(os.getenv("LITE_DB_PATH", str(data_dir / "lite.db"))).resolve()
        lancedb_dir = Path(
            os.getenv("LITE_LANCEDB_DIR", str(data_dir / "lancedb"))
        ).resolve()
        graph_dir = Path(os.getenv("LITE_GRAPH_DIR", str(data_dir / "kuzu"))).resolve()
        return cls(
            app_name=os.getenv("LITE_APP_NAME", "MiniMem"),
            host=os.getenv("LITE_HOST", "127.0.0.1"),
            port=int(os.getenv("LITE_PORT", "20195")),
            data_dir=data_dir,
            db_path=db_path,
            lancedb_dir=lancedb_dir,
            graph_dir=graph_dir,
            retrieval_profile=os.getenv("LITE_RETRIEVAL_PROFILE", "agentic"),
            request_status_ttl_sec=int(
                os.getenv("LITE_REQUEST_STATUS_TTL_SEC", "3600")
            ),
            nonce_ttl_sec=int(os.getenv("LITE_NONCE_TTL_SEC", "600")),
            openai_base_url=os.getenv(
                "LITE_OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"
            ),
            openai_api_key=os.getenv(
                "LITE_OPENAI_API_KEY", ""
            ),
            chat_base_url=os.getenv(
                "LITE_CHAT_BASE_URL", "https://api.siliconflow.cn/v1"
            ),
            chat_api_key=os.getenv(
                "LITE_CHAT_API_KEY",
                "",
            ),
            chat_model=os.getenv(
                "LITE_CHAT_MODEL", "Qwen/Qwen3-8B"
            ),
            chat_provider=os.getenv("LITE_CHAT_PROVIDER", "siliconflow"),
            embedding_base_url=os.getenv(
                "LITE_EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1/embeddings"
            ),
            embedding_api_key=os.getenv(
                "LITE_EMBEDDING_API_KEY",
                "",
            ),
            embedding_provider=os.getenv("LITE_EMBEDDING_PROVIDER", "openai"),
            embedding_model=os.getenv(
                "LITE_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"
            ),
            extractor_provider=os.getenv("LITE_EXTRACTOR_PROVIDER", "chat_model"),
            extractor_base_url=os.getenv(
                "LITE_EXTRACTOR_BASE_URL",
                os.getenv("LITE_CHAT_BASE_URL", "https://api.siliconflow.cn/v1"),
            ),
            extractor_api_key=os.getenv(
                "LITE_EXTRACTOR_API_KEY",
                os.getenv("LITE_CHAT_API_KEY", ""),
            ),
            extractor_model=os.getenv(
                "LITE_EXTRACTOR_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507"
            ),
            phase1_enabled=_env_bool("LITE_PHASE1_ENABLED", True),
            phase1_semantic_boundary_enabled=_env_bool(
                "LITE_PHASE1_SEMANTIC_BOUNDARY_ENABLED", True
            ),
            phase1_narrative_enabled=_env_bool("LITE_PHASE1_NARRATIVE_ENABLED", True),
            phase1_boundary_min_confidence=max(
                0.0,
                min(
                    1.0,
                    float(os.getenv("LITE_PHASE1_BOUNDARY_MIN_CONFIDENCE", "0.68")),
                ),
            ),
            phase2_agentic_verifier_enabled=_env_bool(
                "LITE_PHASE2_AGENTIC_VERIFIER_ENABLED", True
            ),
            phase2_agentic_verifier_min_confidence=max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "LITE_PHASE2_AGENTIC_VERIFIER_MIN_CONFIDENCE", "0.66"
                        )
                    ),
                ),
            ),
            phase3_query_rewriter_enabled=_env_bool(
                "LITE_PHASE3_QUERY_REWRITER_ENABLED", True
            ),
            phase3_query_rewriter_min_confidence=max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "LITE_PHASE3_QUERY_REWRITER_MIN_CONFIDENCE", "0.62"
                        )
                    ),
                ),
            ),
            phase4_reasoning_enabled=_env_bool("LITE_PHASE4_REASONING_ENABLED", True),
            phase4_temporal_rerank_weight=max(
                0.0,
                min(1.0, float(os.getenv("LITE_PHASE4_TEMPORAL_RERANK_WEIGHT", "0.35"))),
            ),
            phase4_multi_hop_max_queries=max(
                1, int(os.getenv("LITE_PHASE4_MULTI_HOP_MAX_QUERIES", "3"))
            ),
            agent_policy_provider=os.getenv("LITE_AGENT_POLICY_PROVIDER", "rule"),
            agent_policy_model=os.getenv(
                "LITE_AGENT_POLICY_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507"
            ),
            agent_policy_enabled=_env_bool("LITE_AGENT_POLICY_ENABLED", True),
            graph_enabled=_env_bool("LITE_GRAPH_ENABLED", True),
            graph_top_k=int(os.getenv("LITE_GRAPH_TOP_K", "3")),
            graph_write_min_importance=float(
                os.getenv("LITE_GRAPH_WRITE_MIN_IMPORTANCE", "0.6")
            ),
            key_memory_importance_threshold=float(
                os.getenv("LITE_KEY_MEMORY_IMPORTANCE_THRESHOLD", "0.5")
            ),
            vector_lancedb_enabled=_env_bool("LITE_VECTOR_LANCEDB_ENABLED", True),
            vector_lancedb_min_importance=float(
                os.getenv("LITE_VECTOR_LANCEDB_MIN_IMPORTANCE", "0.72")
            ),
            vector_write_min_importance=float(
                os.getenv("LITE_VECTOR_WRITE_MIN_IMPORTANCE", "0.30")
            ),
            vector_index_type=str(
                os.getenv("LITE_VECTOR_INDEX_TYPE", "IVF_HNSW_PQ")
            ).strip()
            or "IVF_HNSW_PQ",
            vector_index_metric=str(
                os.getenv("LITE_VECTOR_INDEX_METRIC", "cosine")
            ).strip()
            or "cosine",
            vector_index_m=max(4, int(os.getenv("LITE_VECTOR_INDEX_M", "20"))),
            vector_index_ef_construction=max(
                32, int(os.getenv("LITE_VECTOR_INDEX_EF_CONSTRUCTION", "300"))
            ),
            vector_search_nprobes=max(
                1, int(os.getenv("LITE_VECTOR_SEARCH_NPROBES", "20"))
            ),
            vector_search_ef=max(
                8, int(os.getenv("LITE_VECTOR_SEARCH_EF", "80"))
            ),
            vector_search_refine_factor=max(
                0, int(os.getenv("LITE_VECTOR_SEARCH_REFINE_FACTOR", "2"))
            ),
            vector_embed_chunk_chars=max(
                120, int(os.getenv("LITE_VECTOR_EMBED_CHUNK_CHARS", "600"))
            ),
            vector_embed_max_chunks=max(
                1, int(os.getenv("LITE_VECTOR_EMBED_MAX_CHUNKS", "8"))
            ),
            search_budget_factor=max(
                2, int(os.getenv("LITE_SEARCH_BUDGET_FACTOR", "4"))
            ),
            search_min_probe_k=max(
                4, int(os.getenv("LITE_SEARCH_MIN_PROBE_K", "12"))
            ),
            keyword_confident_best_score=float(
                os.getenv("LITE_KEYWORD_CONFIDENT_BEST_SCORE", "9.0")
            ),
            keyword_confident_kth_score=float(
                os.getenv("LITE_KEYWORD_CONFIDENT_KTH_SCORE", "2.8")
            ),
            semantic_vector_budget_cap=max(
                8, int(os.getenv("LITE_SEMANTIC_VECTOR_BUDGET_CAP", "32"))
            ),
            semantic_keyword_budget_cap=max(
                4, int(os.getenv("LITE_SEMANTIC_KEYWORD_BUDGET_CAP", "16"))
            ),
            query_embed_cache_size=max(
                32, int(os.getenv("LITE_QUERY_EMBED_CACHE_SIZE", "256"))
            ),
            query_embed_cache_ttl_sec=max(
                30, int(os.getenv("LITE_QUERY_EMBED_CACHE_TTL_SEC", "900"))
            ),
            search_trace_enabled=_env_bool("LITE_SEARCH_TRACE_ENABLED", False),
            search_trace_slow_ms=max(
                0, int(os.getenv("LITE_SEARCH_TRACE_SLOW_MS", "0"))
            ),
        )

