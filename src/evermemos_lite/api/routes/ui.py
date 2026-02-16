from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse, Response

router = APIRouter(tags=["ui"])


@router.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@router.get("/console", include_in_schema=False)
async def console() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@router.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@router.get("/ui", response_class=HTMLResponse)
async def ui_index() -> str:
    html_path = Path(__file__).resolve().parents[2] / "ui" / "index.html"
    if html_path.exists():
        text = html_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return "<html><body><h1>MiniMem</h1><p>UI is ready.</p></body></html>"
