from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EventLogItem:
    fact_order: int
    fact: str
    fact_norm: str


class RuleEventLogExtractor:
    """Build event-log items from atomic facts with stable dedup keys."""

    def __init__(self, *, max_items: int = 32, max_fact_chars: int = 220) -> None:
        self.max_items = max(1, int(max_items))
        self.max_fact_chars = max(32, int(max_fact_chars))

    def extract(self, *, atomic_facts: list[str], episode: str) -> list[EventLogItem]:
        facts = self._normalize_facts(atomic_facts)
        if not facts:
            facts = self._facts_from_episode(episode)
        out: list[EventLogItem] = []
        seen: set[str] = set()
        for fact in facts:
            clean = str(fact or "").strip()
            if not clean:
                continue
            norm = self._norm_key(clean)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append(
                EventLogItem(
                    fact_order=len(out) + 1,
                    fact=clean[: self.max_fact_chars],
                    fact_norm=norm[:128],
                )
            )
            if len(out) >= self.max_items:
                break
        return out

    def _normalize_facts(self, facts: list[str]) -> list[str]:
        out: list[str] = []
        for item in list(facts or []):
            text = " ".join(str(item or "").split()).strip(" ，,。;；")
            if not text:
                continue
            out.append(text)
            if len(out) >= self.max_items:
                break
        return out

    def _facts_from_episode(self, episode: str) -> list[str]:
        raw = " ".join(str(episode or "").split())
        if not raw:
            return []
        parts = [
            x.strip(" ，,。;；")
            for x in re.split(r"[。！？!?；;]|(?<!\d)\.(?!\d)", raw)
            if x.strip()
        ]
        if not parts:
            return [raw[: self.max_fact_chars]]
        return parts[: self.max_items]

    @staticmethod
    def _norm_key(text: str) -> str:
        value = str(text or "").lower()
        value = re.sub(r"[\s\W_]+", "", value, flags=re.UNICODE)
        return value.strip()
