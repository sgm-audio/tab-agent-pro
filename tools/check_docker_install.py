#!/usr/bin/env python3
"""
Guard the Docker install line for Basic Pitch.

Background (docs-audit P0-2):
  * ``pip install --no-deps basic-pitch>=0.4.0`` is parsed by the shell as
    ``pip install --no-deps basic-pitch > =0.4.0`` — the version pin is lost.
  * With ``--no-deps``, the packages that ``basic_pitch.note_creation`` imports
    (``mir_eval`` and ``resampy``) must be installed explicitly, otherwise
    ``import basic_pitch`` fails inside the image and the app silently
    degrades to "no transcription models available".

This script fails if either regresses.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ("mir_eval", "resampy")

failures: list[str] = []


def _unquoted_specifier(line: str) -> str | None:
    """Return the offending token if a version specifier is not shell-quoted."""
    for match in re.finditer(r"basic-pitch[><=]", line):
        start = match.start()
        if start > 0 and line[start - 1] in "\"'":
            continue  # quoted, the shell will not treat ">" as a redirection
        return line[max(0, start - 20) : match.end() + 20].strip()
    return None


def check_dockerfile() -> None:
    text = (ROOT / "Dockerfile").read_text()

    for line in text.splitlines():
        stripped = line.strip()
        if "pip install" not in stripped or "basic-pitch" not in stripped:
            continue

        # Unquoted "basic-pitch>=0.4.0" is a shell redirection: the pin is lost.
        offending = _unquoted_specifier(stripped)
        if offending:
            failures.append(
                f"Unquoted basic-pitch version specifier (shell redirection): {offending}"
            )

        # A --no-deps install must add mir_eval/resampy back explicitly.
        if "--no-deps" in stripped:
            for dep in REQUIRED:
                if dep not in text:
                    failures.append(
                        f"'{dep}' is not installed anywhere in the Dockerfile, but "
                        "--no-deps basic-pitch needs it at import time"
                    )


def check_requirements() -> None:
    reqs = (ROOT / "requirements.txt").read_text()
    for dep in REQUIRED:
        if dep.replace("_", "-") not in reqs and dep not in reqs:
            failures.append(f"'{dep}' missing from requirements.txt")


def check_app_binds_externally() -> None:
    app = (ROOT / "app.py").read_text()
    if 'os.getenv("HOST", "127.0.0.1")' in app:
        failures.append(
            "app.py still defaults HOST to 127.0.0.1; Docker/Hugging Face Spaces "
            "cannot reach it (use 0.0.0.0)"
        )


def main() -> int:
    check_dockerfile()
    check_requirements()
    check_app_binds_externally()

    if failures:
        print("Docker install checks FAILED:")
        for f in failures:
            print(f"  ✗ {f}")
        return 1

    print("Docker install checks passed (basic-pitch deps present, host binds externally).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
