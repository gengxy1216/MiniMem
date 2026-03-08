from __future__ import annotations

import json
from urllib import error, request

from flockmem.service.http_auth import build_auth_headers


class OpenAIRerankProvider:
    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()

    def rerank(self, *, query: str, documents: list[str]) -> list[float]:
        base_url = self.base_url.strip()
        api_key = self.api_key.strip()
        model = self.model.strip()
        if not base_url:
            raise ValueError("rerank base_url is required")
        if not api_key:
            raise ValueError("rerank api_key is required")
        if not model:
            raise ValueError("rerank model is required")
        docs = [str(x or "").strip() for x in documents if str(x or "").strip()]
        if not docs:
            return []
        payload = {
            "model": model,
            "query": str(query or "").strip(),
            "documents": docs,
            "top_n": len(docs),
        }
        req = request.Request(
            url=self._build_rerank_url(base_url),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._build_headers(api_key),
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=45) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"rerank HTTP {exc.code}: {detail[:260]}") from exc
        except Exception as exc:
            raise RuntimeError(f"rerank request failed: {exc}") from exc

        try:
            body = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"invalid rerank response: {raw[:260]}") from exc

        scores = [0.0] * len(docs)
        parsed_any = False
        results = body.get("results")
        if isinstance(results, list):
            for row in results:
                if not isinstance(row, dict):
                    continue
                idx = row.get("index")
                score = row.get("relevance_score", row.get("score"))
                if isinstance(idx, int) and 0 <= idx < len(scores):
                    try:
                        scores[idx] = float(score)
                        parsed_any = True
                    except Exception:
                        continue
        data = body.get("data")
        if isinstance(data, list):
            for idx, row in enumerate(data):
                if not isinstance(row, dict):
                    continue
                score = row.get("relevance_score", row.get("score"))
                try:
                    if idx < len(scores):
                        scores[idx] = float(score)
                        parsed_any = True
                except Exception:
                    continue
        if not parsed_any:
            raise RuntimeError("rerank response missing scores")
        return scores

    def _build_rerank_url(self, base_url: str) -> str:
        base = str(base_url or "").strip().rstrip("/")
        if base.endswith("/rerank"):
            return base
        if base.endswith("/v1") or base.endswith("/v2"):
            return f"{base}/rerank"
        return f"{base}/rerank"

    def _build_headers(self, api_key: str) -> dict[str, str]:
        return build_auth_headers(api_key)

