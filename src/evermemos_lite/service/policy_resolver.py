from __future__ import annotations

from dataclasses import dataclass

from evermemos_lite.config.profiles import PROFILE_PRESETS
from evermemos_lite.domain.policy import EffectivePolicy, RuntimePolicy


@dataclass(frozen=True)
class ResolveInput:
    default_profile: str
    tenant_id: str
    request_override: RuntimePolicy | None = None


class PolicyResolver:
    def __init__(self, repo) -> None:
        self.repo = repo

    def resolve(self, payload: ResolveInput) -> EffectivePolicy:
        default_profile = payload.default_profile or "agentic"
        if default_profile not in PROFILE_PRESETS:
            default_profile = "agentic"

        base = PROFILE_PRESETS[default_profile].to_policy()
        tenant_patch = self.repo.get(payload.tenant_id)
        merged = base.merged_with(tenant_patch).merged_with(payload.request_override)

        return EffectivePolicy(
            vector_enabled=bool(
                merged.vector_enabled if merged.vector_enabled is not None else True
            ),
            keyword_enabled=bool(
                merged.keyword_enabled if merged.keyword_enabled is not None else True
            ),
            agentic_enabled=bool(
                merged.agentic_enabled if merged.agentic_enabled is not None else False
            ),
            importance_threshold=float(
                merged.importance_threshold
                if merged.importance_threshold is not None
                else 0.2
            ),
            keyword_top_k=int(merged.keyword_top_k if merged.keyword_top_k else 30),
            vector_top_k=int(merged.vector_top_k if merged.vector_top_k else 30),
            rrf_k=int(merged.rrf_k if merged.rrf_k else 60),
            profile=merged.profile or default_profile,
            reason=merged.reason or f"resolved:{default_profile}",
        )
