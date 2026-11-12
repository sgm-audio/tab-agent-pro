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

    found_basic_pitch = False
    for line in text.splitlines():
        stripped = line.strip()
        if "pip install" not in stripped or "basic-pitch" not in stripped:
            continue

        found_basic_pitch = True
        if "--no-deps" not in stripped:
            failures.append(
                f"basic-pitch must be installed with --no-deps in Dockerfile: {stripped}"
            )

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

    if not found_basic_pitch:
        failures.append("Dockerfile does not install basic-pitch in any pip install line")


def check_requirements() -> None:
    reqs = (ROOT / "requirements.txt").read_text()
    for dep in REQUIRED:
        if dep.replace("_", "-") not in reqs and dep not in reqs:
            failures.append(f"'{dep}' missing from requirements.txt")


def check_app_binds_externally() -> None:
    app = (ROOT / "app.py").read_text()
    if 'os.getenv("HOST", "127.0.0.1")' in app:
        failures.append(
            "app.py still defaults HOST to 127.0.0.1; Docker containers cannot "
            "reach it (use 0.0.0.0)"
        )


def check_hardening() -> None:
    """Production-hardening invariants added by the phase-5 pass."""
    text = (ROOT / "Dockerfile").read_text()

    if "HEALTHCHECK" not in text:
        failures.append("Dockerfile has no HEALTHCHECK")
    elif "/ready" not in text:
        failures.append(
            "HEALTHCHECK must probe /ready, not /health: /health stays 200 when "
            "every transcription backend failed to import"
        )

    if "USER tabagent" not in text:
        failures.append("Dockerfile does not drop privileges with USER")

    if 'CMD ["python", "app.py"]' not in text:
        failures.append(
            "Dockerfile CMD must use exec form so SIGTERM reaches the server " "(graceful shutdown)"
        )

    if "--index-url https://download.pytorch.org/whl/cpu" not in text:
        failures.append(
            "torch must install from the CPU-only index; the PyPI wheel pulls "
            "~2.5 GB of unusable NVIDIA CUDA packages"
        )

    # The build must (a) re-apply the lock so the installed set matches what was
    # tested, and (b) fail when the installed set and the lock disagree.
    if "pip install --no-deps -r requirements.lock" not in text:
        failures.append(
            "Dockerfile must re-apply requirements.lock after the range-based "
            "install, otherwise the shipped versions can drift from the lock"
        )
    if "check_lockfile.py --interpreter" not in text:
        failures.append(
            "Dockerfile must verify that every locked pin installs on the image's "
            "own interpreter before installing them"
        )
    if "check_lockfile.py" not in text:
        failures.append(
            "Dockerfile must fail the build when the installed set and "
            "requirements.lock disagree"
        )


def main() -> int:
    check_dockerfile()
    check_requirements()
    check_app_binds_externally()
    check_hardening()

    if failures:
        print("Docker install checks FAILED:")
        for f in failures:
            print(f"  ✗ {f}")
        return 1

    print(
        "Docker checks passed (basic-pitch deps, external bind, non-root, "
        "readiness probe, CPU torch, lockfile enforce+verify)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
