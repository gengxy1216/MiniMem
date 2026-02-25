from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from urllib import error, request


@dataclass(frozen=True)
class RewriteDecision:
    query: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class QueryExpansionDecision:
    queries: list[str]
    confidence: float
    reason: str


class QueryRewriterProtocol(Protocol):
    def rewrite(self, *, query: str, hits: list[dict], insufficiency_reason: str) -> RewriteDecision | None:
        ...


class ChatModelQueryRewriter:
    def __init__(self, *, base_url: str, api_key: str, model: str, enabled: bool) -> None:
        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        self.enabled = bool(enabled)

    def rewrite(self, *, query: str, hits: list[dict], insufficiency_reason: str) -> RewriteDecision | None:
        if not self._enabled():
            return None
        compact_hits = []
        for row in list(hits or [])[:5]:
            if not isinstance(row, dict):
                continue
            compact_hits.append(
                {
                    "summary": str(row.get("summary", "")).strip()[:160],
                    "subject": str(row.get("subject", "")).strip()[:80],
                    "source": str(row.get("source", "")).strip()[:48],
                    "timestamp": int(_to_float(row.get("timestamp"), default=0.0)),
                }
            )
        payload = {
            "query": str(query or "").strip()[:300],
            "insufficiency_reason": str(insufficiency_reason or "").strip()[:120],
            "hits": compact_hits,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite retrieval queries to improve evidence recall.\n"
                    "Return strict JSON only with keys: query, confidence, reason.\n"
                    "Rules: keep user intent unchanged; add missing entities/time constraints."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            raw = self._chat_completion(messages=messages)
            data = _safe_json(raw)
            rewritten = str(data.get("query") or "").strip()
            if not rewritten:
                return None
            return RewriteDecision(
                query=rewritten[:400],
                confidence=max(0.0, min(1.0, _to_float(data.get("confidence"), default=0.0))),
                reason=str(data.get("reason") or "").strip()[:80] or "llm_rewrite",
            )
        except Exception:
            return None

    def expand_queries(
        self,
        *,
        query: str,
        hits: list[dict],
        insufficiency_reason: str,
        max_queries: int = 3,
    ) -> QueryExpansionDecision | None:
        if not self._enabled():
            return None
        cap = max(1, min(4, int(max_queries)))
        compact_hits = []
        for row in list(hits or [])[:6]:
            if not isinstance(row, dict):
                continue
            compact_hits.append(
                {
                    "summary": str(row.get("summary", "")).strip()[:140],
                    "subject": str(row.get("subject", "")).strip()[:80],
                    "source": str(row.get("source", "")).strip()[:40],
                    "timestamp": int(_to_float(row.get("timestamp"), default=0.0)),
                }
            )
        payload = {
            "query": str(query or "").strip()[:300],
            "insufficiency_reason": str(insufficiency_reason or "").strip()[:120],
            "hits": compact_hits,
            "max_queries": cap,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You generate complementary retrieval queries.\n"
                    "Return strict JSON only with keys: queries, confidence, reason.\n"
                    "Rules: keep intent unchanged; each query should cover a different missing clue."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            raw = self._chat_completion(messages=messages)
            data = _safe_json(raw)
            raw_queries = data.get("queries")
            if not isinstance(raw_queries, list):
                return None
            queries: list[str] = []
            seen: set[str] = set()
            base = str(query or "").strip().lower()
            for item in raw_queries:
                q = " ".join(str(item or "").strip().split())[:360]
                if not q:
                    continue
                key = q.lower()
                if key == base or key in seen:
                    continue
                seen.add(key)
                queries.append(q)
                if len(queries) >= cap:
                    break
            if not queries:
                return None
            return QueryExpansionDecision(
                queries=queries,
                confidence=max(0.0, min(1.0, _to_float(data.get("confidence"), default=0.0))),
                reason=str(data.get("reason") or "").strip()[:80] or "llm_query_expand",
            )
        except Exception:
            return None

    def _enabled(self) -> bool:
        return self.enabled and bool(self.base_url and self.api_key and self.model)

    def _chat_completion(self, *, messages: list[dict[str, str]]) -> str:
        url = self._build_completion_url(self.base_url)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 220,
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
            raise RuntimeError(f"rewriter HTTP {exc.code}: {detail[:220]}") from exc
        except Exception as exc:
            raise RuntimeError(f"rewriter request failed: {exc}") from exc
        body = json.loads(raw)
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = "".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict)
            ).strip()
        else:
            text = str(content).strip()
        if not text:
            raise RuntimeError("rewriter returned empty content")
        return text

    @staticmethod
    def _build_completion_url(base_url: str) -> str:
        base = str(base_url or "").strip().rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/chat/completions"


def _safe_json(text: str) -> dict:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _to_float(value, *, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)
