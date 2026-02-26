from __future__ import annotations

import json
import re
from datetime import datetime
from urllib import error, request
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,4}")
_STOP_TOKENS = {"我", "你", "他", "她", "它", "吗", "呢", "啊", "呀", "的", "了"}
_SLICE_LEN = 240
_SNIPPET_LEN = 220


def _normalize_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_text(value: str) -> set[str]:
    raw = _normalize_text(value).lower()
    if not raw:
        return set()
    tokens = {t for t in _TOKEN_RE.findall(raw) if t}
    chars = re.findall(r"[\u4e00-\u9fff]", raw)
    for idx in range(len(chars) - 1):
        tokens.add(chars[idx] + chars[idx + 1])
    if chars:
        tokens.add("".join(chars))
    return {t for t in tokens if t and t not in _STOP_TOKENS}


def _token_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b)) / float(max(1, len(a | b)))


def _slice_text(value: str, slice_len: int) -> list[str]:
    text = _normalize_text(value)
    if not text:
        return []
    out: list[str] = []
    for idx in range(0, len(text), max(1, int(slice_len))):
        out.append(text[idx : idx + slice_len])
    return out


def _extract_query_snippet(query: str, text: str) -> tuple[str, int | None, float]:
    clean = _normalize_text(text)
    if not clean:
        return "", None, 0.0
    q = _normalize_text(query)
    if q:
        hit_idx = clean.lower().find(q.lower())
        if hit_idx >= 0:
            slice_no = hit_idx // _SLICE_LEN + 1
            start = max(0, hit_idx - 40)
            snippet = clean[start : start + _SNIPPET_LEN]
            return snippet, slice_no, 1.0
    chunks = _slice_text(clean, _SLICE_LEN)
    if not chunks:
        return clean[:_SNIPPET_LEN], 1, 0.0
    query_tokens = _tokenize_text(q)
    if not query_tokens:
        return chunks[0][:_SNIPPET_LEN], 1, 0.0
    best_idx = 0
    best_score = -1.0
    for idx, chunk in enumerate(chunks):
        score = _token_overlap(query_tokens, _tokenize_text(chunk))
        if score > best_score:
            best_idx = idx
            best_score = score
    return chunks[best_idx][:_SNIPPET_LEN], best_idx + 1, max(0.0, best_score)


class ChatResponder:
    def __init__(
        self, base_url: str, api_key: str, model: str, provider: str = "openai"
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.provider = provider

    def respond(
        self,
        *,
        query: str,
        memories: list[dict[str, Any]],
        system_time: datetime | None = None,
        provider: str | None = None,
        provider_options: dict[str, dict[str, str]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        active_provider = (provider or self.provider or "openai").strip() or "openai"
        provider_cfg = (provider_options or {}).get(active_provider, {})
        active_base_url = (
            str(provider_cfg.get("base_url") or "").strip() or self.base_url
        )
        active_api_key = str(provider_cfg.get("api_key") or "").strip() or self.api_key
        active_model = (
            str(provider_cfg.get("model") or "").strip()
            or (model or self.model)
            or self.model
        )
        if not active_base_url:
            raise ValueError(f"chat provider '{active_provider}' missing base_url")
        if not active_api_key:
            raise ValueError(f"chat provider '{active_provider}' missing api_key")
        if not active_model:
            raise ValueError(f"chat provider '{active_provider}' missing model")

        snippets: list[str] = []
        citations: list[dict[str, Any]] = []
        for row in memories[:5]:
            base_text = str(row.get("episode") or row.get("summary") or "")
            fallback_text = str(row.get("summary") or row.get("episode") or "").strip()
            snippet, slice_no, match_score = _extract_query_snippet(query, base_text)
            if not snippet:
                snippet = fallback_text
            if snippet:
                snippets.append(f"- {snippet}")
            citation = dict(row)
            citation["citation_snippet"] = snippet
            citation["citation_match_score"] = float(match_score)
            if slice_no is not None:
                citation["citation_slice"] = int(slice_no)
            citations.append(citation)
        memory_context = "\n".join(snippets) if snippets else "- 当前没有检索到可用记忆。"
        now = system_time or datetime.now().astimezone()
        system_prompt = (
            "你是 MiniMem 助手。"
            f"当前系统时间：{now.isoformat(timespec='seconds')}。"
            "请仅在记忆片段与用户问题明显相关时使用记忆；"
            "若问题是寒暄或无关问题，不要硬套记忆。"
            "若记忆不足，请明确说明并给出下一步建议。"
        )
        user_prompt = (
            "用户问题：\n"
            f"{query}\n\n"
            "可用记忆片段：\n"
            f"{memory_context}"
        )
        answer = self._chat_completion(
            base_url=active_base_url,
            api_key=active_api_key,
            model=active_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return {
            "answer": answer,
            "citations": citations,
            "model": active_model,
            "provider": active_provider,
        }

    def _chat_completion(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        url = self._build_completion_url(base_url)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
        }
        req = request.Request(
            url=url,
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
                f"chat completion HTTP {exc.code}: {detail[:260]}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"chat completion request failed: {exc}") from exc
        try:
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
                raise RuntimeError("chat completion returned empty content")
            return text
        except Exception as exc:
            raise RuntimeError(f"invalid chat completion response: {raw[:260]}") from exc

    def _build_completion_url(self, base_url: str) -> str:
        base = base_url.strip().rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/chat/completions"
