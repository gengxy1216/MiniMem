from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


def _resolve_kuzu_db_path(path: Path) -> Path:
    if path.suffix.lower() == ".kuzu":
        return path
    return path / "graph.kuzu"


def _is_noise_triple(subject: str, relation: str, obj: str) -> bool:
    bad_subject = {"谁", "什么", "哪", "哪里", "哪儿", "哪个", "哪位"}
    if subject.strip() in bad_subject:
        return True
    text = f"{subject}{relation}{obj}"
    if "?" in text or "？" in text:
        return True
    return False


@dataclass(frozen=True)
class GraphRow:
    subject: str
    relation: str
    obj: str
    confidence: float
    event_id: str
    timestamp: int
    user_id: str | None
    group_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "obj": self.obj,
            "confidence": self.confidence,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "group_id": self.group_id,
        }


class KuzuGraphStore:
    def __init__(self, db_dir: Path, enabled: bool = True) -> None:
        self.db_dir = Path(db_dir)
        self.db_path = _resolve_kuzu_db_path(self.db_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.enabled = bool(enabled)
        self._rows: list[GraphRow] = []

    def upsert_triples(
        self,
        triples: list[Any],
        *,
        event_id: str,
        timestamp: int,
        user_id: str | None,
        group_id: str | None,
    ) -> int:
        if not self.enabled:
            return 0
        inserted = 0
        for t in triples:
            subject = str(getattr(t, "subject", "")).strip()
            relation = str(getattr(t, "relation", "")).strip()
            obj = str(getattr(t, "obj", "")).strip()
            confidence = float(getattr(t, "confidence", 0.5))
            if not subject or not relation or not obj:
                continue
            if _is_noise_triple(subject, relation, obj):
                continue
            self._rows.append(
                GraphRow(
                    subject=subject,
                    relation=relation,
                    obj=obj,
                    confidence=max(0.0, min(1.0, confidence)),
                    event_id=event_id,
                    timestamp=int(timestamp),
                    user_id=user_id,
                    group_id=group_id,
                )
            )
            inserted += 1
        return inserted

    def search(
        self, query: str, *, user_id: str | None, group_id: str | None, limit: int
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        q = query.strip().lower()
        q_tokens = _tokenize_text(q)
        out: list[tuple[float, dict[str, Any]]] = []
        for row in reversed(self._rows):
            if user_id and row.user_id != user_id:
                continue
            if group_id and row.group_id != group_id:
                continue
            text = f"{row.subject} {row.relation} {row.obj}".lower()
            text_tokens = _tokenize_text(text)
            lexical = _token_overlap(q_tokens, text_tokens)
            if q and q in text:
                lexical = max(lexical, 0.65)
            relation_boost = _relation_intent_boost(q, row.relation)
            score = lexical + relation_boost + 0.25 * float(row.confidence)
            if score < 0.18:
                continue
            item = row.to_dict()
            item["match_score"] = float(score)
            out.append((score, item))
        out.sort(key=lambda item: (item[0], item[1].get("timestamp", 0)), reverse=True)
        return [row for _, row in out[: max(1, int(limit))]]

    def neighbors(
        self, entity_name: str, *, user_id: str | None, group_id: str | None, limit: int
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        name = entity_name.strip().lower()
        out: list[dict[str, Any]] = []
        for row in reversed(self._rows):
            if user_id and row.user_id != user_id:
                continue
            if group_id and row.group_id != group_id:
                continue
            if row.subject.lower() != name and row.obj.lower() != name:
                continue
            out.append(row.to_dict())
            if len(out) >= max(1, int(limit)):
                break
        return out


def _tokenize_text(text: str) -> set[str]:
    raw = str(text or "").strip().lower()
    if not raw:
        return set()
    tokens = {t for t in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,4}", raw) if t}
    chars = re.findall(r"[\u4e00-\u9fff]", raw)
    for i in range(len(chars) - 1):
        tokens.add(chars[i] + chars[i + 1])
    if chars:
        tokens.add("".join(chars))
    stop = {"我", "你", "他", "她", "它", "吗", "呢", "啊", "呀", "的", "了"}
    return {t for t in tokens if t and t not in stop}


def _token_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b)) / float(max(1, len(a | b)))


def _relation_intent_boost(query: str, relation: str) -> float:
    q = str(query or "").lower()
    r = str(relation or "").lower()
    if not q or not r:
        return 0.0
    if _is_identity_query(q) and r == "name_is":
        return 1.2
    is_age_query = any(k in q for k in ("几岁", "多大", "年龄", "age", "old"))
    if is_age_query and r == "age_is":
        if any(k in q for k in ("儿子", "女儿", "孩子", "son", "daughter", "child")):
            return 1.15
        return 0.9
    if any(k in q for k in ("儿子", "孩子", "son")) and r in {"has_son", "has_child"}:
        return 0.8
    if any(k in q for k in ("女儿", "daughter")) and r == "has_daughter":
        return 0.8
    if any(k in q for k in ("喜欢", "爱好", "like")) and r == "likes":
        return 0.6
    return 0.0


def _is_identity_query(query: str) -> bool:
    q = str(query or "").lower()
    if not q:
        return False
    patterns = (
        r"我叫.*什么",
        r"我的名字",
        r"我是谁",
        r"who am i",
        r"what(?:'s| is) my name",
    )
    return any(re.search(p, q, flags=re.IGNORECASE) for p in patterns)
