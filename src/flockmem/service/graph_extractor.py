from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphTriple:
    subject: str
    relation: str
    obj: str
    confidence: float


_QUESTION_PREFIX_RE = re.compile(
    r"^(?:请问)?(?:谁|什么|啥|哪|哪个|哪位|哪里|哪儿|几|怎么|如何|为何|为啥|是不是|能不能|可不可以)"
)
_QUESTION_END_RE = re.compile(r"(吗|么|呢)$")


def _is_question(text: str) -> bool:
    t = text.strip()
    return bool(_QUESTION_PREFIX_RE.search(t) or _QUESTION_END_RE.search(t) or "?" in t or "？" in t)


def _normalize_entity(value: str) -> str:
    return value.strip(" ，。；;！？!?")


def extract_graph_triples(facts: list[str], user_id: str) -> list[GraphTriple]:
    triples: list[GraphTriple] = []
    for raw in facts:
        text = raw.strip()
        if not text or _is_question(text):
            continue

        m = re.search(r"我叫([^\s，。；;！？!?]{1,32})", text)
        if m:
            triples.append(GraphTriple(user_id, "name_is", _normalize_entity(m.group(1)), 0.95))
            continue

        m = re.search(r"我的儿子是([^\s，。；;！？!?]{1,32})", text)
        if m:
            triples.append(GraphTriple(user_id, "has_son", _normalize_entity(m.group(1)), 0.9))
            continue

        m = re.search(r"([^\s，。；;！？!?]{1,32})是我的儿子", text)
        if m:
            triples.append(GraphTriple(user_id, "has_son", _normalize_entity(m.group(1)), 0.9))
            continue

        m = re.search(r"我的女儿是([^\s，。；;！？!?]{1,32})", text)
        if m:
            triples.append(GraphTriple(user_id, "has_daughter", _normalize_entity(m.group(1)), 0.9))
            continue

        m = re.search(r"([^\s，。；;！？!?]{1,32})是我的女儿", text)
        if m:
            triples.append(GraphTriple(user_id, "has_daughter", _normalize_entity(m.group(1)), 0.9))
            continue

        m = re.search(r"我喜欢([^\s，。；;！？!?]{1,32})", text)
        if m:
            triples.append(GraphTriple(user_id, "likes", _normalize_entity(m.group(1)), 0.85))
            continue

        m = re.search(r"(?:我|本人)(?:现在|今年)?\s*([0-9]{1,3})岁", text)
        if m:
            triples.append(GraphTriple(user_id, "age_is", f"{m.group(1)}岁", 0.9))
            continue

        m = re.search(r"^([^\s，。；;！？!?]{1,32})\s*([0-9]{1,3})岁$", text)
        if m:
            subject = _normalize_entity(m.group(1))
            if subject in {"我", "本人"}:
                subject = user_id
            triples.append(GraphTriple(subject, "age_is", f"{m.group(2)}岁", 0.88))
            continue

        m = re.search(r"([^\s，。；;！？!?]{1,32})是我的([^\s，。；;！？!?]{1,32})", text)
        if m:
            person = _normalize_entity(m.group(1))
            relation = _normalize_entity(m.group(2))
            triples.append(GraphTriple(user_id, f"has_{relation}", person, 0.8))
            continue

    unique = {}
    for t in triples:
        unique[(t.subject, t.relation, t.obj)] = t
    return list(unique.values())
