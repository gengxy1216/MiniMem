from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    agent_policy_provider: str
    agent_policy_model: str
    agent_policy_enabled: bool
    graph_enabled: bool
    graph_top_k: int
    graph_write_min_importance: float
    key_memory_importance_threshold: float

    @classmethod
    def from_env(cls) -> "LiteSettings":
        data_dir = Path(os.getenv("LITE_DATA_DIR", "MiniMem_data")).resolve()
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
            retrieval_profile=os.getenv("LITE_RETRIEVAL_PROFILE", "medium"),
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
        )

