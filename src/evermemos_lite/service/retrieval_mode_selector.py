from __future__ import annotations

from dataclasses import dataclass

from evermemos_lite.domain.policy import RuntimePolicy


@dataclass(frozen=True)
class SelectionInput:
    query: str
    top_k: int
    user_id: str | None = None
    group_id: str | None = None


@dataclass(frozen=True)
class SelectionOutput:
    policy: RuntimePolicy
    reason: str


class RuleRetrievalModeSelector:
    def select(self, payload: SelectionInput) -> SelectionOutput:
        q = payload.query.strip().lower()
        is_precise = any(token in q for token in ("谁", "what", "where", "when", "哪"))
        is_long = len(q) > 80

        if is_precise and not is_long:
            policy = RuntimePolicy(
                keyword_enabled=True,
                vector_enabled=False,
                agentic_enabled=False,
                reason="rule:keyword_for_precise_query",
            )
        elif is_long:
            policy = RuntimePolicy(
                keyword_enabled=True,
                vector_enabled=True,
                agentic_enabled=True,
                reason="rule:agentic_for_long_query",
            )
        else:
            policy = RuntimePolicy(
                keyword_enabled=True,
                vector_enabled=True,
                agentic_enabled=False,
                reason="rule:hybrid_default",
            )
        return SelectionOutput(policy=policy, reason=policy.reason or "rule")


class OpenAIRetrievalModeSelector:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._fallback = RuleRetrievalModeSelector()

    def select(self, payload: SelectionInput) -> SelectionOutput:
        # Keep stable in local/offline environments; this can be replaced with
        # actual LLM policy selection when credentials are configured.
        return self._fallback.select(payload)
