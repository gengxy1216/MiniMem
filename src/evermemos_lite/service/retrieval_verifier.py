from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from urllib import error, request


@dataclass(frozen=True)
class SufficiencyDecision:
    sufficient: bool
    confidence: float
    reason: str


class RetrievalVerifierProtocol(Protocol):
    def judge_sufficiency(
        self, *, query: str, hits: list[dict], top_k: int
    ) -> SufficiencyDecision | None:
        ...


class ChatModelRetrievalVerifier:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        enabled: bool,
    ) -> None:
        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        self.enabled = bool(enabled)

    def judge_sufficiency(
        self, *, query: str, hits: list[dict], top_k: int
    ) -> SufficiencyDecision | None:
        if not self._enabled():
            return None
        compact_hits = []
        for row in list(hits or [])[: max(1, min(int(top_k), 6))]:
            if not isinstance(row, dict):
                continue
            compact_hits.append(
                {
                    "id": str(row.get("id", ""))[:64],
                    "summary": str(row.get("summary", "")).strip()[:160],
                    "subject": str(row.get("subject", "")).strip()[:80],
                    "score": _to_float(row.get("score"), default=0.0),
                    "source": str(row.get("source", "")).strip()[:48],
                    "timestamp": int(_to_float(row.get("timestamp"), default=0.0)),
                }
            )
        payload = {
            "query": str(query or "").strip()[:300],
            "top_k": max(1, int(top_k)),
            "hits": compact_hits,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a retrieval sufficiency checker.\n"
                    "Return strict JSON only with keys:\n"
                    "sufficient (bool), confidence (0~1), reason (short string <= 40 chars).\n"
                    "Judge whether current retrieved evidence is enough to answer the query."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            raw = self._chat_completion(messages=messages)
            data = _safe_json(raw)
            return SufficiencyDecision(
                sufficient=_to_bool(data.get("sufficient")),
                confidence=max(
                    0.0, min(1.0, _to_float(data.get("confidence"), default=0.0))
                ),
                reason=str(data.get("reason") or "").strip()[:80] or "llm_verifier",
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
            "max_tokens": 180,
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
            raise RuntimeError(f"verifier HTTP {exc.code}: {detail[:220]}") from exc
        except Exception as exc:
            raise RuntimeError(f"verifier request failed: {exc}") from exc
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
            raise RuntimeError("verifier returned empty content")
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


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
