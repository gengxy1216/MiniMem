from __future__ import annotations

from dataclasses import dataclass

from evermemos_lite.domain.policy import RuntimePolicy


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    vector_dim: int
    vector_enabled: bool
    keyword_enabled: bool
    agentic_enabled: bool
    importance_threshold: float
    keyword_top_k: int
    vector_top_k: int
    rrf_k: int

    def to_policy(self) -> RuntimePolicy:
        return RuntimePolicy(
            vector_enabled=self.vector_enabled,
            keyword_enabled=self.keyword_enabled,
            agentic_enabled=self.agentic_enabled,
            importance_threshold=self.importance_threshold,
            keyword_top_k=self.keyword_top_k,
            vector_top_k=self.vector_top_k,
            rrf_k=self.rrf_k,
            profile=self.name,
            reason=f"profile:{self.name}",
        )


PROFILE_PRESETS: dict[str, RetrievalProfile] = {
    "keyword": RetrievalProfile(
        name="keyword",
        vector_dim=256,
        vector_enabled=False,
        keyword_enabled=True,
        agentic_enabled=False,
        importance_threshold=0.45,
        keyword_top_k=24,
        vector_top_k=0,
        rrf_k=60,
    ),
    "hybrid": RetrievalProfile(
        name="hybrid",
        vector_dim=384,
        vector_enabled=True,
        keyword_enabled=True,
        agentic_enabled=False,
        importance_threshold=0.25,
        keyword_top_k=30,
        vector_top_k=30,
        rrf_k=60,
    ),
    "agentic": RetrievalProfile(
        name="agentic",
        vector_dim=768,
        vector_enabled=True,
        keyword_enabled=True,
        agentic_enabled=True,
        importance_threshold=0.15,
        keyword_top_k=50,
        vector_top_k=50,
        rrf_k=80,
    ),
}
