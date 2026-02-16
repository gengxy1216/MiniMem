from __future__ import annotations

import os
import socket
import subprocess
import time

import uvicorn

from evermemos_lite.bootstrap.app_factory import create_app
from evermemos_lite.config.settings import LiteSettings


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _find_pids_on_port(port: int) -> list[int]:
    if os.name != "nt":
        return []
    proc = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: set[int] = set()
    needle = f":{port}"
    for line in proc.stdout.splitlines():
        t = line.strip()
        if "LISTENING" not in t:
            continue
        if needle not in t:
            continue
        parts = t.split()
        if not parts:
            continue
        try:
            pid = int(parts[-1])
        except Exception:
            continue
        if pid != os.getpid():
            pids.add(pid)
    return sorted(pids)


def _kill_pid(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)


def _preflight_port(host: str, port: int) -> None:
    if not _is_port_open(host, port):
        return
    auto_kill = os.getenv("LITE_AUTO_KILL_PORT", "true").strip().lower() in {
        "1",
        "on",
        "true",
        "yes",
    }
    if not auto_kill:
        raise RuntimeError(
            f"Port {port} is already in use. Set LITE_AUTO_KILL_PORT=true or free it manually."
        )
    pids = _find_pids_on_port(port)
    for pid in pids:
        _kill_pid(pid)
    deadline = time.time() + 5
    while time.time() < deadline:
        if not _is_port_open(host, port):
            return
        time.sleep(0.2)
    raise RuntimeError(f"Port {port} is still in use after cleanup attempt.")


def main() -> None:
    settings = LiteSettings.from_env()
    _preflight_port(settings.host, settings.port)
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)
