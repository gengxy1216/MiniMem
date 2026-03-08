from __future__ import annotations

import re
from typing import Any


class LocalHeuristicRerankProvider:
    """Local lightweight rerank provider based on lexical overlap."""

    def __init__(
        self,
        *,
        model: str,
        device: str = "cpu",
        batch_size: int = 16,
        max_concurrency: int = 2,
    ) -> None:
        self.model = str(model or "").strip() or "local-rerank-lexical-v1"
        self.device = str(device or "").strip() or "cpu"
        self.batch_size = max(1, int(batch_size))
        self.max_concurrency = max(1, int(max_concurrency))

    def rerank(self, *, query: str, documents: list[str]) -> list[float]:
        q_tokens = self._tokenize(query)
        if not documents:
            return []
        out: list[float] = []
        for doc in documents:
            d_tokens = self._tokenize(doc)
            if not q_tokens or not d_tokens:
                out.append(0.0)
                continue
            overlap = len(q_tokens & d_tokens) / float(max(1, len(q_tokens | d_tokens)))
            out.append(float(overlap))
        return out

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        raw = str(text or "").lower().strip()
        if not raw:
            return set()
        tokens = {t for t in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,4}", raw) if t}
        chars = re.findall(r"[\u4e00-\u9fff]", raw)
        for idx in range(len(chars) - 1):
            tokens.add(chars[idx] + chars[idx + 1])
        if chars:
            tokens.add("".join(chars))
        stop = {"我", "你", "他", "她", "它", "吗", "呢", "啊", "呀", "的", "了"}
        return {x for x in tokens if x and x not in stop}


def normalize_rerank_scores(scores: list[Any], expected_len: int) -> list[float]:
    out: list[float] = []
    for value in scores[:expected_len]:
        try:
            out.append(float(value))
        except Exception:
            out.append(0.0)
    while len(out) < expected_len:
        out.append(0.0)
    return out
