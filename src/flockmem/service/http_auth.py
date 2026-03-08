from __future__ import annotations


def normalize_api_key_token(api_key: str) -> str:
    token = str(api_key or "").strip()
    if not token:
        return ""
    lower = token.lower()
    if lower.startswith("authorization:"):
        token = token.split(":", 1)[1].strip()
        lower = token.lower()
    for prefix in ("bearer ", "token ", "apikey "):
        if lower.startswith(prefix):
            return token.split(" ", 1)[1].strip()
    return token


def build_auth_headers(api_key: str) -> dict[str, str]:
    """Build OpenAI-compatible auth headers with best-effort scheme detection."""
    token = str(api_key or "").strip()
    headers = {"Content-Type": "application/json"}
    if not token:
        return headers
    lower = token.lower()
    if lower.startswith("authorization:"):
        auth_value = token.split(":", 1)[1].strip()
        if auth_value:
            headers["Authorization"] = auth_value
        return headers
    if lower.startswith(("bearer ", "basic ", "token ", "apikey ")):
        headers["Authorization"] = token
        return headers
    headers["Authorization"] = f"Bearer {token}"
    return headers
