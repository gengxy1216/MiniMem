from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class RuntimePolicy:
    vector_enabled: bool | None = None
    keyword_enabled: bool | None = None
    agentic_enabled: bool | None = None
    importance_threshold: float | None = None
    keyword_top_k: int | None = None
    vector_top_k: int | None = None
    rrf_k: int | None = None
    profile: str | None = None
    reason: str | None = None

    def merged_with(self, override: "RuntimePolicy | None") -> "RuntimePolicy":
        if override is None:
            return RuntimePolicy(**self.to_dict())
        data = self.to_dict()
        for key, value in override.to_dict().items():
            if value is not None:
                data[key] = value
        return RuntimePolicy(**data)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "RuntimePolicy":
        if not value:
            return cls()
        allowed = {f.name for f in fields(cls)}
        data = {k: v for k, v in value.items() if k in allowed}
        return cls(**data)


@dataclass(frozen=True)
class EffectivePolicy:
    vector_enabled: bool
    keyword_enabled: bool
    agentic_enabled: bool
    importance_threshold: float
    keyword_top_k: int
    vector_top_k: int
    rrf_k: int
    profile: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector_enabled": self.vector_enabled,
            "keyword_enabled": self.keyword_enabled,
            "agentic_enabled": self.agentic_enabled,
            "importance_threshold": self.importance_threshold,
            "keyword_top_k": self.keyword_top_k,
            "vector_top_k": self.vector_top_k,
            "rrf_k": self.rrf_k,
            "profile": self.profile,
            "reason": self.reason,
        }
