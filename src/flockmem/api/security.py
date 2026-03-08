from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Request

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def _extract_token(authorization: str | None, x_api_key: str | None) -> str:
    if x_api_key and str(x_api_key).strip():
        return str(x_api_key).strip()
    raw = str(authorization or "").strip()
    if not raw:
        return ""
    lower = raw.lower()
    if lower.startswith("authorization:"):
        return raw.split(":", 1)[1].strip()
    if lower.startswith(("bearer ", "token ")):
        return raw.split(" ", 1)[1].strip()
    return raw


def _is_local_request(request: Request) -> bool:
    client = request.client
    host = str(client.host).strip().lower() if client and client.host else ""
    return host in _LOCAL_HOSTS


async def require_admin_access(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    settings = request.app.state.settings
    expected = str(getattr(settings, "admin_token", "") or "").strip()
    allow_localhost = bool(getattr(settings, "admin_allow_localhost", True))

    if expected:
        provided = _extract_token(authorization, x_api_key)
        if provided and secrets.compare_digest(provided, expected):
            return
        raise HTTPException(status_code=401, detail="admin authorization required")

    if allow_localhost and _is_local_request(request):
        return
    raise HTTPException(
        status_code=401,
        detail="admin token not configured; set LITE_ADMIN_TOKEN or enable localhost access",
    )
