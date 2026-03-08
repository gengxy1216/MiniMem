from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol
from urllib import error, request

from evermemos_lite.service.http_auth import build_auth_headers, normalize_api_key_token

EXTRACT_INPUT_CHAR_CAP = 4000
EXTRACT_SUMMARY_CHAR_CAP = 220
EXTRACT_SUMMARY_HARD_CAP = 360
EXTRACT_ATOMIC_FACT_MAX_ITEMS = 24
EXTRACT_ATOMIC_FACT_MAX_CHARS = 160
EXTRACT_MAX_TOKENS = 900
EXTRACT_MAX_RETRIES = 2


@dataclass(frozen=True)
class ExtractedMemory:
    episode: str
    summary: str
    subject: str
    importance_score: float
    atomic_facts: list[str]
    foresights: list[dict]
    profile_patch: dict[str, str]


class MemoryExtractor(Protocol):
    def extract(self, content: str, sender: str, group_id: str | None) -> ExtractedMemory:
        ...


class RuleMemoryExtractor:
    def extract(self, content: str, sender: str, group_id: str | None) -> ExtractedMemory:
        return _build_rule_extracted(content=content, sender=sender)


class OpenAIMemoryExtractor:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.model = model
        self.client = None
        try:
            from openai import OpenAI  # type: ignore

            self.client = OpenAI(
                base_url=base_url,
                api_key=normalize_api_key_token(api_key),
            )
        except Exception:
            self.client = None
        self._fallback = RuleMemoryExtractor()

    def extract(self, content: str, sender: str, group_id: str | None) -> ExtractedMemory:
        if self.client is None:
            return self._fallback.extract(content=content, sender=sender, group_id=group_id)
        episode = _markdown_to_plain(content).strip()
        compact_input = episode[:EXTRACT_INPUT_CHAR_CAP]
        prompt = (
            "You are a memory extractor. Return strict JSON only, no markdown.\n"
            f"Fields: episode (string), summary (<={EXTRACT_SUMMARY_CHAR_CAP} chars), subject (short), "
            "importance_score (0~1), "
            f"atomic_facts (1~{EXTRACT_ATOMIC_FACT_MAX_ITEMS} items; each <={EXTRACT_ATOMIC_FACT_MAX_CHARS} chars), "
            "foresights (array), profile_patch (object string->string).\n"
            "Preserve temporal order and named entities. Keep facts retrieval-oriented."
        )
        for _ in range(EXTRACT_MAX_RETRIES):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": compact_input},
                    ],
                    temperature=0.0,
                    max_tokens=EXTRACT_MAX_TOKENS,
                )
                raw = str(completion.choices[0].message.content or "").strip()
                data = _safe_json(raw)
                return _build_extracted_from_dict(data=data, episode=episode, sender=sender)
            except Exception:
                continue
        return self._fallback.extract(content=content, sender=sender, group_id=group_id)


class ChatModelMemoryExtractor:
    """Use runtime chat model as semantic extractor with retrieval-oriented output."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        self._fallback = RuleMemoryExtractor()

    def extract(self, content: str, sender: str, group_id: str | None) -> ExtractedMemory:
        if not self.base_url or not self.api_key or not self.model:
            return self._fallback.extract(content=content, sender=sender, group_id=group_id)
        episode = _markdown_to_plain(content).strip()
        if not episode:
            return self._fallback.extract(content=content, sender=sender, group_id=group_id)
        compact_input = episode[:EXTRACT_INPUT_CHAR_CAP]
        sys_prompt = (
            "You extract memory into strict JSON only.\n"
            "Required keys: episode, summary, subject, importance_score, atomic_facts, foresights, profile_patch.\n"
            f"Rules: summary <= {EXTRACT_SUMMARY_CHAR_CAP} chars; "
            f"atomic_facts 1~{EXTRACT_ATOMIC_FACT_MAX_ITEMS} items; "
            f"each fact <= {EXTRACT_ATOMIC_FACT_MAX_CHARS} chars; no explanation text.\n"
            "Keep temporal and relation clues in atomic_facts."
        )
        user_prompt = (
            "Split this memory into fine-grained facts for retrieval.\n"
            f"Input:\n{compact_input}"
        )
        for _ in range(EXTRACT_MAX_RETRIES):
            try:
                raw = self._chat_completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                data = _safe_json(raw)
                return _build_extracted_from_dict(data=data, episode=episode, sender=sender)
            except Exception:
                continue
        return self._fallback.extract(content=content, sender=sender, group_id=group_id)

    def _chat_completion(self, *, model: str, messages: list[dict[str, str]]) -> str:
        url = self._build_completion_url(self.base_url)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": EXTRACT_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }
        req = request.Request(
            url=url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=build_auth_headers(self.api_key),
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=18) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"extractor HTTP {exc.code}: {detail[:220]}") from exc
        except Exception as exc:
            raise RuntimeError(f"extractor request failed: {exc}") from exc
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
        if not text:
            raise RuntimeError("extractor returned empty content")
        return text

    @staticmethod
    def _build_completion_url(base_url: str) -> str:
        base = base_url.strip().rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/chat/completions"


def _build_rule_extracted(content: str, sender: str) -> ExtractedMemory:
    episode = _markdown_to_plain(content).strip()
    atomic_facts = _extract_atomic_facts(episode)
    profile_patch = _extract_profile_patch(episode)
    summary = _build_compact_summary(episode=episode, atomic_facts=atomic_facts)
    importance = _estimate_importance(
        episode=episode, atomic_facts=atomic_facts, profile_patch=profile_patch
    )
    foresights = _extract_foresights(episode)
    return ExtractedMemory(
        episode=episode,
        summary=summary,
        subject=sender.strip() or "user",
        importance_score=importance,
        atomic_facts=atomic_facts,
        foresights=foresights,
        profile_patch=profile_patch,
    )


def _build_extracted_from_dict(
    *, data: dict, episode: str, sender: str
) -> ExtractedMemory:
    normalized_episode = str(data.get("episode") or episode).strip()
    atomic_facts = _normalize_atomic_facts(data.get("atomic_facts"), normalized_episode)
    profile_patch = _normalize_profile_patch(data.get("profile_patch"))
    summary = str(data.get("summary") or _build_compact_summary(normalized_episode, atomic_facts)).strip()
    if len(summary) > EXTRACT_SUMMARY_HARD_CAP:
        summary = summary[:EXTRACT_SUMMARY_HARD_CAP]
    subject = str(data.get("subject") or sender or "user").strip()[:60] or "user"
    try:
        imp = float(data.get("importance_score", -1.0))
    except Exception:
        imp = -1.0
    if imp < 0.0 or imp > 1.0:
        imp = _estimate_importance(
            episode=normalized_episode,
            atomic_facts=atomic_facts,
            profile_patch=profile_patch,
        )
    foresights = _normalize_foresights(data.get("foresights"))
    return ExtractedMemory(
        episode=normalized_episode,
        summary=summary,
        subject=subject,
        importance_score=max(0.0, min(1.0, imp)),
        atomic_facts=atomic_facts,
        foresights=foresights,
        profile_patch=profile_patch,
    )


def _safe_json(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                return {}
    return {}


def _estimate_importance(
    *, episode: str, atomic_facts: list[str], profile_patch: dict[str, str]
) -> float:
    score = min(1.0, len(episode.strip()) / 400.0 + 0.2)
    lower = episode.lower()
    if len(atomic_facts) >= 3:
        score += 0.08
    if profile_patch:
        score += 0.08
    if any(k in lower for k in ("儿子", "女儿", "孩子", "我叫", "岁")):
        score += 0.08
    if any(k in lower for k in ("紧急", "important", "deadline", "明天", "马上")):
        score += 0.2
    return round(max(0.0, min(1.0, score)), 3)


def _build_compact_summary(episode: str, atomic_facts: list[str]) -> str:
    if atomic_facts:
        text = "；".join(atomic_facts[:8])
        return text[:EXTRACT_SUMMARY_HARD_CAP]
    return episode[:EXTRACT_SUMMARY_HARD_CAP]


def _extract_atomic_facts(content: str) -> list[str]:
    text = str(content or "").strip()
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()

    for fact in _extract_structured_atomic_facts(text):
        key = fact.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key[:EXTRACT_ATOMIC_FACT_MAX_CHARS])
        if len(out) >= EXTRACT_ATOMIC_FACT_MAX_ITEMS:
            return out

    rough_parts = [
        p.strip()
        for p in re.split(r"[。！？!?；;，,\n]+|(?<!\d)\.(?!\d)", text)
        if p.strip()
    ]
    for raw in rough_parts:
        cleaned = re.sub(r"^\d+\.\s*", "", raw).strip()
        cleaned = re.sub(r"^(?:并且|而且|然后|另外)\s*", "", cleaned)
        if re.fullmatch(r"\d{1,3}岁(?:了)?", cleaned):
            continue
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned[:EXTRACT_ATOMIC_FACT_MAX_CHARS])
        if len(out) >= EXTRACT_ATOMIC_FACT_MAX_ITEMS:
            break
    return out[:EXTRACT_ATOMIC_FACT_MAX_ITEMS]


def _extract_structured_atomic_facts(text: str) -> list[str]:
    out: list[str] = []
    child_names: list[str] = []

    for m in re.finditer(
        r"(?:我的儿子是|我儿子是|([^\s，。；;！？!?]{1,32})是我的儿子)([^\s，。；;！？!?]{0,32})?",
        text,
    ):
        if m.group(1):
            name = m.group(1)
        else:
            name = _normalize_name_tail(str(m.group(2) or ""))
        if not name:
            m2 = re.search(r"(?:我的儿子是|我儿子是)([^\s，。；;！？!?]{1,32})", m.group(0))
            name = _normalize_name_tail(m2.group(1)) if m2 else ""
        if not name:
            continue
        out.append(f"我的儿子是{name}")
        child_names.append(name)

    for m in re.finditer(
        r"(?:我的女儿是|我女儿是|([^\s，。；;！？!?]{1,32})是我的女儿)([^\s，。；;！？!?]{0,32})?",
        text,
    ):
        if m.group(1):
            name = m.group(1)
        else:
            name = _normalize_name_tail(str(m.group(2) or ""))
        if not name:
            m2 = re.search(r"(?:我的女儿是|我女儿是)([^\s，。；;！？!?]{1,32})", m.group(0))
            name = _normalize_name_tail(m2.group(1)) if m2 else ""
        if not name:
            continue
        out.append(f"我的女儿是{name}")
        child_names.append(name)

    for name in list(dict.fromkeys(child_names)):
        m_age = re.search(rf"{re.escape(name)}[，,\s]*([0-9]{{1,2}})岁", text)
        if m_age:
            out.append(f"{name}{m_age.group(1)}岁")

    m_son_age = re.search(
        r"(?:我的儿子是|我儿子是)([^\s，。；;！？!?]{1,32})[，,\s]*([0-9]{1,2})岁",
        text,
    )
    if m_son_age:
        out.append(f"{_normalize_name_tail(m_son_age.group(1))}{m_son_age.group(2)}岁")

    m_daughter_age = re.search(
        r"(?:我的女儿是|我女儿是)([^\s，。；;！？!?]{1,32})[，,\s]*([0-9]{1,2})岁",
        text,
    )
    if m_daughter_age:
        out.append(f"{_normalize_name_tail(m_daughter_age.group(1))}{m_daughter_age.group(2)}岁")

    m_self_age = re.search(r"(?:我|本人)(?:现在|今年)?\s*([0-9]{1,3})岁", text)
    if m_self_age:
        out.append(f"我{m_self_age.group(1)}岁")

    m_name = re.search(r"我叫([\u4e00-\u9fa5A-Za-z0-9_]{1,32})", text)
    if m_name:
        out.append(f"我叫{m_name.group(1)}")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in out:
        fact = item.strip(" ，。；;！？!?")
        if not fact or fact in seen:
            continue
        seen.add(fact)
        normalized.append(fact)
        if len(normalized) >= EXTRACT_ATOMIC_FACT_MAX_ITEMS:
            break
    return normalized


def _normalize_name_tail(value: str) -> str:
    return str(value or "").strip(" ，。；;！？!?")


def _markdown_to_plain(content: str) -> str:
    lines = content.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    in_code = False
    for line in lines:
        t = line.rstrip()
        if t.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        t = t.strip()
        if not t:
            continue
        if t.startswith("#"):
            t = t.lstrip("#").strip()
        if t.startswith(">"):
            t = t.lstrip(">").strip()
        t = re.sub(r"^[-*+]\s+", "", t)
        t = re.sub(r"^\d+\.\s+", "", t)
        t = re.sub(r"`([^`]+)`", r"\1", t)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)
        t = re.sub(r"[*_~]{1,3}", "", t)
        if t:
            out.append(t)
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_foresights(content: str) -> list[dict]:
    lower = content.lower()
    if not any(k in lower for k in ("明天", "下周", "deadline", "截止", "提醒", "todo", "计划")):
        return []
    snippet = content.strip()[:EXTRACT_SUMMARY_HARD_CAP]
    return [{"content": snippet, "start_time": None, "end_time": None, "confidence": 0.62}]


def _extract_profile_patch(content: str) -> dict[str, str]:
    out: dict[str, str] = {}
    match = re.search(r"我叫([\u4e00-\u9fa5A-Za-z0-9_]{1,32})", content)
    if match:
        out["name"] = match.group(1)
    m_age = re.search(r"(?:我|本人)(?:现在|今年)?\s*([0-9]{1,3})岁", content)
    if m_age:
        out["age"] = m_age.group(1)
    m_son = re.search(r"(?:我的儿子是|我儿子是)([^\s，。；;！？!?]{1,32})", content)
    if m_son:
        out["son_name"] = _normalize_name_tail(m_son.group(1))
        m_son_age = re.search(rf"{re.escape(out['son_name'])}[，,\s]*([0-9]{{1,2}})岁", content)
        if m_son_age:
            out["son_age"] = m_son_age.group(1)
    m_daughter = re.search(r"(?:我的女儿是|我女儿是)([^\s，。；;！？!?]{1,32})", content)
    if m_daughter:
        out["daughter_name"] = _normalize_name_tail(m_daughter.group(1))
    return out


def _normalize_atomic_facts(value, episode: str) -> list[str]:
    if isinstance(value, list):
        out = []
        seen: set[str] = set()
        for x in value:
            fact = str(x).strip()
            if not fact:
                continue
            fact = fact[:EXTRACT_ATOMIC_FACT_MAX_CHARS]
            if fact in seen:
                continue
            seen.add(fact)
            out.append(fact)
            if len(out) >= EXTRACT_ATOMIC_FACT_MAX_ITEMS:
                break
        if out:
            return out
    return _extract_atomic_facts(episode)


def _normalize_foresights(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    out: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        try:
            confidence = float(item.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        out.append(
            {
                "content": content[:300],
                "confidence": max(0.0, min(1.0, confidence)),
                "start_time": _to_int_or_none(item.get("start_time")),
                "end_time": _to_int_or_none(item.get("end_time")),
            }
        )
        if len(out) >= 6:
            break
    return out


def _normalize_profile_patch(value) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in value.items():
        key = str(k).strip()[:64]
        val = str(v).strip()[:200]
        if key and val:
            out[key] = val
    return out


def _to_int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None
