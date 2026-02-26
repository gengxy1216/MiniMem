from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from threading import Lock
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
        self._rows_path = self.db_path.parent / "graph_rows.jsonl"
        self.enabled = bool(enabled)
        self._lock = Lock()
        self._rows: list[GraphRow] = []
        self._load_rows()

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
        append_rows: list[GraphRow] = []
        for t in triples:
            row = self._to_graph_row(
                triple=t,
                event_id=event_id,
                timestamp=timestamp,
                user_id=user_id,
                group_id=group_id,
            )
            if row is None:
                continue
            append_rows.append(row)
        if not append_rows:
            return 0
        with self._lock:
            self._rows.extend(append_rows)
            for row in append_rows:
                self._append_row(row)
                inserted += 1
        return inserted

    def search(
        self, query: str, *, user_id: str | None, group_id: str | None, limit: int
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        q = query.strip().lower()
        q_tokens = _tokenize_text(q)
        with self._lock:
            rows_snapshot = list(self._rows)
        out: list[tuple[float, dict[str, Any]]] = []
        for row in reversed(rows_snapshot):
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
        with self._lock:
            rows_snapshot = list(self._rows)
        out: list[dict[str, Any]] = []
        for row in reversed(rows_snapshot):
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

    def _to_graph_row(
        self,
        *,
        triple: Any,
        event_id: str,
        timestamp: int,
        user_id: str | None,
        group_id: str | None,
    ) -> GraphRow | None:
        subject = str(getattr(triple, "subject", "")).strip()
        relation = str(getattr(triple, "relation", "")).strip()
        obj = str(getattr(triple, "obj", "")).strip()
        confidence = float(getattr(triple, "confidence", 0.5))
        if not subject or not relation or not obj:
            return None
        if _is_noise_triple(subject, relation, obj):
            return None
        return GraphRow(
            subject=subject,
            relation=relation,
            obj=obj,
            confidence=max(0.0, min(1.0, confidence)),
            event_id=event_id,
            timestamp=int(timestamp),
            user_id=user_id,
            group_id=group_id,
        )

    def _append_row(self, row: GraphRow) -> None:
        self._rows_path.parent.mkdir(parents=True, exist_ok=True)
        with self._rows_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False))
            fh.write("\n")

    def _load_rows(self) -> None:
        if not self._rows_path.exists():
            return
        restored: list[GraphRow] = []
        with self._rows_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except Exception:
                    continue
                row = self._row_from_dict(payload)
                if row is None:
                    continue
                restored.append(row)
        self._rows = restored

    @staticmethod
    def _row_from_dict(value: Any) -> GraphRow | None:
        if not isinstance(value, dict):
            return None
        subject = str(value.get("subject", "")).strip()
        relation = str(value.get("relation", "")).strip()
        obj = str(value.get("obj", "")).strip()
        event_id = str(value.get("event_id", "")).strip()
        if not subject or not relation or not obj or not event_id:
            return None
        try:
            confidence = float(value.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        try:
            timestamp = int(value.get("timestamp", 0))
        except Exception:
            timestamp = 0
        user_id_raw = value.get("user_id")
        group_id_raw = value.get("group_id")
        user_id = str(user_id_raw).strip() if user_id_raw is not None else None
        group_id = str(group_id_raw).strip() if group_id_raw is not None else None
        return GraphRow(
            subject=subject,
            relation=relation,
            obj=obj,
            confidence=max(0.0, min(1.0, confidence)),
            event_id=event_id,
            timestamp=timestamp,
            user_id=user_id or None,
            group_id=group_id or None,
        )


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
