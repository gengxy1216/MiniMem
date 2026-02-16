from __future__ import annotations

import hashlib
import math


class HashEmbeddingProvider:
    def __init__(self, dim: int = 384) -> None:
        self.dim = max(8, int(dim))

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        normalized = text.strip().lower()
        if not normalized:
            return vector
        for token in normalized.split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            return vector
        return [x / norm for x in vector]
