from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib import error, request

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


_PROFILE_SET = {"keyword", "hybrid", "agentic"}


def _policy_from_profile(profile: str, reason: str) -> RuntimePolicy:
    name = str(profile or "").strip().lower()
    if name == "keyword":
        return RuntimePolicy(
            profile="keyword",
            keyword_enabled=True,
            vector_enabled=False,
            agentic_enabled=False,
            reason=reason,
        )
    if name == "hybrid":
        return RuntimePolicy(
            profile="hybrid",
            keyword_enabled=True,
            vector_enabled=True,
            agentic_enabled=False,
            reason=reason,
        )
    return RuntimePolicy(
        profile="agentic",
        keyword_enabled=True,
        vector_enabled=True,
        agentic_enabled=True,
        reason=reason,
    )


class RuleRetrievalModeSelector:
    def select(self, payload: SelectionInput) -> SelectionOutput:
        q = payload.query.strip().lower()
        is_precise = any(token in q for token in ("谁", "什么", "where", "when", "哪", "who"))
        is_short = len(q) <= 24
        if is_precise and is_short:
            policy = _policy_from_profile("keyword", "rule:keyword_for_precise_query")
        elif len(q) <= 12:
            policy = _policy_from_profile("hybrid", "rule:hybrid_for_short_query")
        else:
            policy = _policy_from_profile("agentic", "rule:agentic_default")
        return SelectionOutput(policy=policy, reason=policy.reason or "rule")


class OpenAIRetrievalModeSelector:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._fallback = RuleRetrievalModeSelector()

    def select(self, payload: SelectionInput) -> SelectionOutput:
        if not self.base_url or not self.api_key or not self.model:
            return self._fallback.select(payload)
        try:
            raw = self._chat_completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a retrieval policy selector.\n"
                            "Return strict JSON only with keys: profile, reason.\n"
                            "profile must be one of: keyword, hybrid, agentic."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"query={payload.query}\n"
                            f"top_k={payload.top_k}\n"
                            "Pick the minimal profile that keeps recall quality."
                        ),
                    },
                ],
            )
            profile, reason = self._parse_profile(raw)
            if profile not in _PROFILE_SET:
                return self._fallback.select(payload)
            policy = _policy_from_profile(profile, reason or f"agent:{profile}")
            return SelectionOutput(policy=policy, reason=policy.reason or "agent")
        except Exception:
            return self._fallback.select(payload)

    def _chat_completion(self, *, model: str, messages: list[dict[str, str]]) -> str:
        url = self._build_completion_url(self.base_url)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 120,
            "response_format": {"type": "json_object"},
        }
        req = request.Request(
            url=url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=18) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"selector HTTP {exc.code}: {detail[:200]}") from exc
        except Exception as exc:
            raise RuntimeError(f"selector request failed: {exc}") from exc
        body = json.loads(raw)
        content = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
            ).strip()
        else:
            text = str(content).strip()
        if not text:
            raise RuntimeError("selector returned empty content")
        return text

    @staticmethod
    def _parse_profile(raw: str) -> tuple[str, str]:
        text = str(raw or "").strip()
        if not text:
            return "", ""
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        obj = {}
        try:
            obj = json.loads(text)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except Exception:
                    obj = {}
        if isinstance(obj, dict):
            profile = str(obj.get("profile", "")).strip().lower()
            reason = str(obj.get("reason", "")).strip()
            if profile in _PROFILE_SET:
                return profile, reason
        profile = text.strip().lower()
        if profile in _PROFILE_SET:
            return profile, ""
        return "", ""

    @staticmethod
    def _build_completion_url(base_url: str) -> str:
        base = base_url.strip().rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/chat/completions"
