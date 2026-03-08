from __future__ import annotations

import shutil
import uuid
from pathlib import Path


class WritableTempDir:
    """Temporary directory helper that avoids tempfile ACL issues in this sandbox."""

    def __init__(self, prefix: str = "tmp", ignore_cleanup_errors: bool = True) -> None:
        base = Path.cwd() / "tmp_test_runtime"
        base.mkdir(parents=True, exist_ok=True)
        name = f"{prefix}_{uuid.uuid4().hex}"
        self.path = base / name
        self.path.mkdir(parents=True, exist_ok=False)
        self.name = str(self.path)
        self._ignore_cleanup_errors = bool(ignore_cleanup_errors)

    def cleanup(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=self._ignore_cleanup_errors)

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

