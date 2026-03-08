from __future__ import annotations

import json
from urllib import error, request

from evermemos_lite.service.http_auth import build_auth_headers


class ChatModelRerankProvider:
    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()

    def rerank(self, *, query: str, documents: list[str]) -> list[float]:
        base_url = self.base_url.strip()
        api_key = self.api_key.strip()
        model = self.model.strip()
        if not base_url:
            raise ValueError("chat rerank base_url is required")
        if not api_key:
            raise ValueError("chat rerank api_key is required")
        if not model:
            raise ValueError("chat rerank model is required")
        docs = [str(x or "").strip() for x in documents if str(x or "").strip()]
        if not docs:
            return []
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a reranker.\n"
                        "Return strict JSON with key `scores` as an array of floats.\n"
                        "Array length must equal document count; higher score means more relevant."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": str(query or "").strip(),
                            "documents": docs,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 220,
            "response_format": {"type": "json_object"},
        }
        req = request.Request(
            url=self._build_completion_url(base_url),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._build_headers(api_key),
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=45) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"chat rerank HTTP {exc.code}: {detail[:260]}") from exc
        except Exception as exc:
            raise RuntimeError(f"chat rerank request failed: {exc}") from exc

        try:
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
            data = self._safe_json(text)
            scores_raw = data.get("scores")
            if not isinstance(scores_raw, list):
                raise RuntimeError("chat rerank response missing scores")
            scores: list[float] = []
            for idx in range(len(docs)):
                value = scores_raw[idx] if idx < len(scores_raw) else 0.0
                try:
                    scores.append(float(value))
                except Exception:
                    scores.append(0.0)
            return scores
        except Exception as exc:
            raise RuntimeError(f"invalid chat rerank response: {raw[:260]}") from exc

    @staticmethod
    def _build_completion_url(base_url: str) -> str:
        base = str(base_url or "").strip().rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/chat/completions"

    @staticmethod
    def _build_headers(api_key: str) -> dict[str, str]:
        return build_auth_headers(api_key)

    @staticmethod
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
