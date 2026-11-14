"""本地 CI：ruff check + ruff format --check + pytest（与 .github/workflows/ci.yml 同源）。

用法（仓库根目录）：python scripts/ci.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS: list[tuple[str, list[str]]] = [
    ("ruff check", [sys.executable, "-m", "ruff", "check", "."]),
    ("ruff format --check", [sys.executable, "-m", "ruff", "format", "--check", "."]),
    ("pytest", [sys.executable, "-m", "pytest", "-q"]),
]


def main() -> int:
    """依次跑三步，第一个失败就返回它的退出码。"""
    for name, cmd in STEPS:
        print(f"== {name}", flush=True)
        code = subprocess.call(cmd, cwd=ROOT)
        if code != 0:
            print(f"FAILED: {name}", flush=True)
            return code
    print("CI OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
