from __future__ import annotations

import json
import re
from typing import Any, Protocol
from urllib import error, request

from flockmem.service.http_auth import build_auth_headers


class ForesightExtractorProtocol(Protocol):
    def extract(
        self,
        *,
        episode: str,
        atomic_facts: list[str],
        existing: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        ...


class RuleForesightExtractor:
    def __init__(self, *, max_items: int = 8) -> None:
        self.max_items = max(1, int(max_items))

    def extract(
        self,
        *,
        episode: str,
        atomic_facts: list[str],
        existing: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        normalized_existing = _normalize_foresights(existing, max_items=self.max_items)
        if normalized_existing:
            return normalized_existing
        text = " ".join(str(episode or "").split()).strip()
        if not text:
            return []
        clues = list(atomic_facts or [])
        if not clues:
            clues = [text]
        out: list[dict[str, Any]] = []
        lower = text.lower()
        temporal_hint = any(
            token in lower
            for token in (
                "tomorrow",
                "next week",
                "next month",
                "deadline",
                "plan",
                "schedule",
                "明天",
                "下周",
                "下个月",
                "截止",
                "计划",
                "安排",
                "提醒",
            )
        )
        if temporal_hint:
            for clue in clues:
                content = " ".join(str(clue or "").split()).strip()
                if not content:
                    continue
                out.append(
                    {
                        "content": content[:280],
                        "start_time": None,
                        "end_time": None,
                        "confidence": 0.62,
                    }
                )
                if len(out) >= self.max_items:
                    break
        return _normalize_foresights(out, max_items=self.max_items)


class ChatModelForesightExtractor:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_items: int = 8,
    ) -> None:
        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        self.max_items = max(1, int(max_items))
        self._fallback = RuleForesightExtractor(max_items=max_items)

    def extract(
        self,
        *,
        episode: str,
        atomic_facts: list[str],
        existing: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if not (self.base_url and self.api_key and self.model):
            return self._fallback.extract(
                episode=episode,
                atomic_facts=atomic_facts,
                existing=existing,
            )
        compact_episode = str(episode or "").strip()[:3000]
        payload = {
            "episode": compact_episode,
            "atomic_facts": list(atomic_facts or [])[:28],
            "existing": _normalize_foresights(existing, max_items=self.max_items),
            "max_items": self.max_items,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract foresight items for memory retrieval.\n"
                    "Return strict JSON with key: foresights.\n"
                    "Each foresight item: content,start_time,end_time,confidence."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            raw = self._chat_completion(messages=messages)
            data = _safe_json(raw)
            foresights = _normalize_foresights(data.get("foresights"), max_items=self.max_items)
            if foresights:
                return foresights
        except Exception:
            pass
        return self._fallback.extract(
            episode=episode,
            atomic_facts=atomic_facts,
            existing=existing,
        )

    def _chat_completion(self, *, messages: list[dict[str, str]]) -> str:
        url = self._build_completion_url(self.base_url)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 260,
            "response_format": {"type": "json_object"},
        }
        req = request.Request(
            url=url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=build_auth_headers(self.api_key),
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=18) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"foresight extractor HTTP {exc.code}: {detail[:220]}") from exc
        except Exception as exc:
            raise RuntimeError(f"foresight extractor request failed: {exc}") from exc
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
            raise RuntimeError("foresight extractor returned empty content")
        return text

    @staticmethod
    def _build_completion_url(base_url: str) -> str:
        base = str(base_url or "").strip().rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/chat/completions"


def _safe_json(text: str) -> dict[str, Any]:
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


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _normalize_foresights(value: Any, *, max_items: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        content = " ".join(str(item.get("content", "")).split()).strip()
        if not content:
            continue
        key = re.sub(r"\s+", " ", content.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        try:
            confidence = float(item.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        out.append(
            {
                "content": content[:300],
                "start_time": _to_int_or_none(item.get("start_time")),
                "end_time": _to_int_or_none(item.get("end_time")),
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )
        if len(out) >= max_items:
            break
    return out


