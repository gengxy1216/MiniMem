from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from urllib import error, request


@dataclass(frozen=True)
class BoundaryDecision:
    should_cut: bool
    confidence: float
    reason: str


class FormationEnhancerProtocol(Protocol):
    def detect_boundary(
        self,
        *,
        query: str,
        recent_user_queries: list[str],
        turn_count: int,
        idle_seconds: int,
    ) -> BoundaryDecision | None:
        ...

    def synthesize_narrative(
        self, *, turns_markdown: str, user_id: str, group_id: str | None
    ) -> str | None:
        ...


class ChatModelFormationEnhancer:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        enabled: bool,
        boundary_enabled: bool,
        narrative_enabled: bool,
    ) -> None:
        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        self.enabled = bool(enabled)
        self.boundary_enabled = bool(boundary_enabled)
        self.narrative_enabled = bool(narrative_enabled)

    def detect_boundary(
        self,
        *,
        query: str,
        recent_user_queries: list[str],
        turn_count: int,
        idle_seconds: int,
    ) -> BoundaryDecision | None:
        if not self._enabled_for_boundary():
            return None
        payload = {
            "query": str(query or "").strip()[:360],
            "recent_user_queries": [
                str(x or "").strip()[:180]
                for x in recent_user_queries[-5:]
                if str(x or "").strip()
            ],
            "turn_count": max(0, int(turn_count)),
            "idle_seconds": max(0, int(idle_seconds)),
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a semantic topic-boundary detector for chat memory segmentation.\n"
                    "Return strict JSON only with keys:\n"
                    "should_cut (bool), confidence (0~1), reason (short string <= 40 chars).\n"
                    "Judge by semantic topic shift, not only lexical overlap."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]
        try:
            raw = self._chat_completion(messages=messages, max_tokens=180)
            data = _safe_json(raw)
            confidence = max(
                0.0, min(1.0, _to_float(data.get("confidence"), default=0.0))
            )
            should_cut = _to_bool(data.get("should_cut"))
            reason = str(data.get("reason") or "").strip()[:80] or "llm_semantic"
            return BoundaryDecision(
                should_cut=bool(should_cut), confidence=float(confidence), reason=reason
            )
        except Exception:
            return None

    def synthesize_narrative(
        self, *, turns_markdown: str, user_id: str, group_id: str | None
    ) -> str | None:
        if not self._enabled_for_narrative():
            return None
        compact_turns = str(turns_markdown or "").strip()
        if not compact_turns:
            return None
        if len(compact_turns) > 2600:
            compact_turns = compact_turns[-2600:]
        payload = {
            "user_id": str(user_id or "").strip(),
            "group_id": str(group_id or "").strip(),
            "turns_markdown": compact_turns,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite chat turns into third-person narrative memory for retrieval.\n"
                    "Return strict JSON only with key: narrative.\n"
                    "Requirements: resolve pronouns, keep key entities/time/events, keep concise."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]
        try:
            raw = self._chat_completion(messages=messages, max_tokens=360)
            data = _safe_json(raw)
            narrative = str(
                data.get("narrative")
                or data.get("episode")
                or data.get("summary")
                or ""
            ).strip()
            if not narrative:
                return None
            narrative = " ".join(narrative.split())
            return narrative[:1200]
        except Exception:
            return None

    def _enabled_for_boundary(self) -> bool:
        return (
            self.enabled
            and self.boundary_enabled
            and bool(self.base_url and self.api_key and self.model)
        )

    def _enabled_for_narrative(self) -> bool:
        return (
            self.enabled
            and self.narrative_enabled
            and bool(self.base_url and self.api_key and self.model)
        )

    def _chat_completion(
        self, *, messages: list[dict[str, str]], max_tokens: int
    ) -> str:
        url = self._build_completion_url(self.base_url)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max(80, int(max_tokens)),
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
            with request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"formation HTTP {exc.code}: {detail[:220]}") from exc
        except Exception as exc:
            raise RuntimeError(f"formation request failed: {exc}") from exc
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
            raise RuntimeError("formation model returned empty content")
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


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def _to_float(value, *, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)
