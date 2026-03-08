from __future__ import annotations

import hashlib
import math
import re


class LocalHashEmbeddingProvider:
    """Deterministic local embedding with no external model dependency."""

    def __init__(
        self,
        *,
        model: str,
        device: str = "cpu",
        batch_size: int = 16,
        max_concurrency: int = 2,
        vector_dim: int = 384,
    ) -> None:
        self.model = str(model or "").strip() or "local-hash-384"
        self.device = str(device or "").strip() or "cpu"
        self.batch_size = max(1, int(batch_size))
        self.max_concurrency = max(1, int(max_concurrency))
        self.vector_dim = max(64, int(vector_dim))

    def embed(self, text: str) -> list[float]:
        raw = str(text or "").strip()
        if not raw:
            raise ValueError("embedding text is required")
        tokens = self._tokenize(raw)
        if not tokens:
            tokens = [raw]
        vec = [0.0] * self.vector_dim
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx_a = int.from_bytes(digest[:4], "big") % self.vector_dim
            idx_b = int.from_bytes(digest[4:], "big") % self.vector_dim
            sign_a = -1.0 if digest[0] & 1 else 1.0
            sign_b = -1.0 if digest[1] & 1 else 1.0
            weight = 1.0 + float((digest[2] % 6)) / 10.0
            vec[idx_a] += sign_a * weight
            vec[idx_b] += sign_b * (weight * 0.7)
        return self._l2_normalize(vec)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        lowered = str(text or "").lower()
        base = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered)
        if not base:
            return []
        tokens = list(base)
        zh = [ch for ch in base if re.match(r"[\u4e00-\u9fff]", ch)]
        for i in range(len(zh) - 1):
            tokens.append(zh[i] + zh[i + 1])
        return tokens

    @staticmethod
    def _l2_normalize(values: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in values))
        if norm <= 1e-12:
            out = [0.0] * len(values)
            if out:
                out[0] = 1.0
            return out
        return [v / norm for v in values]
