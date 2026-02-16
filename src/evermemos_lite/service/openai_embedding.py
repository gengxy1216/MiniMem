from __future__ import annotations

import json
from urllib import error, request


class OpenAIEmbeddingProvider:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def embed(self, text: str) -> list[float]:
        base_url = self.base_url.strip()
        api_key = self.api_key.strip()
        model = self.model.strip()
        if not base_url:
            raise ValueError("embedding base_url is required")
        if not api_key:
            raise ValueError("embedding api_key is required")
        if not model:
            raise ValueError("embedding model is required")
        payload = {
            "model": model,
            "input": text,
            "encoding_format": "float",
        }
        req = request.Request(
            url=self._build_embeddings_url(base_url),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=45) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"embedding HTTP {exc.code}: {detail[:260]}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"embedding request failed: {exc}") from exc
        try:
            body = json.loads(raw)
            vec = body.get("data", [{}])[0].get("embedding")
            if not isinstance(vec, list) or not vec:
                raise RuntimeError("missing embedding in response")
            out: list[float] = []
            for item in vec:
                out.append(float(item))
            return out
        except Exception as exc:
            raise RuntimeError(f"invalid embedding response: {raw[:260]}") from exc

    def _build_embeddings_url(self, base_url: str) -> str:
        base = base_url.strip().rstrip("/")
        if base.endswith("/embeddings"):
            return base
        if base.endswith("/v1"):
            return f"{base}/embeddings"
        return f"{base}/embeddings"
