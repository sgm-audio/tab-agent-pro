#!/usr/bin/env python3
"""
Verify that every variable advertised in .env.example is actually read.

Background (docs-audit P2-1): .env.example used to document HOST, PORT,
CACHE_DIR, DEVICE and HF_TOKEN while only HOST was read by the code, and no
module loaded .env at all.

Exits non-zero if a documented variable has no consumer in the source tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_GLOBS = ("*.py", "reaper/*.lua")
ENV_FILE = ROOT / ".env.example"


def documented_variables() -> list[str]:
    names: list[str] = []
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name = line.split("=", 1)[0].strip()
        if name:
            names.append(name)
    return names


def sources() -> str:
    chunks = []
    for pattern in SOURCE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            chunks.append(path.read_text())
    return "\n".join(chunks)


def main() -> int:
    text = sources()
    missing = [name for name in documented_variables() if name not in text]

    if missing:
        print("Unused variables documented in .env.example:")
        for name in missing:
            print(f"  ✗ {name} (no reference in {', '.join(SOURCE_GLOBS)})")
        return 1

    print(f".env.example OK — {len(documented_variables())} variables all referenced in code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
