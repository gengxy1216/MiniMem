from __future__ import annotations

import json
from datetime import datetime
from urllib import error, request
from typing import Any


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
        for row in memories[:5]:
            summary = str(row.get("summary") or row.get("episode") or "").strip()
            if summary:
                snippets.append(f"- {summary}")
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
            "citations": memories[:5],
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
