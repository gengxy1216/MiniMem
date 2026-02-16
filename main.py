from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evermemos_lite.cli import main as run_minimem  # noqa: E402


def main() -> None:
    run_minimem()


if __name__ == "__main__":
    main()
