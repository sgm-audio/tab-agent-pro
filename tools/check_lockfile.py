#!/usr/bin/env python3
"""
Keep requirements.lock in sync with the environment that actually ships.

Why this exists
---------------
`requirements.txt` is a range-based, resolver-facing list. Because basic-pitch
declares TensorFlow on Linux (which this project deliberately does not install)
a vulnerable-looking TensorFlow/keras/pillow tree shows up in any `pip-audit -r
requirements.txt` run even though none of it is in the image. A lock of the
real, installed set is what lets the security scan answer "is anything we
actually ship vulnerable?" instead of "what would the resolver do?".

Usage:
    python tools/check_lockfile.py           # verify (exit 1 on drift)
    python tools/check_lockfile.py --write   # regenerate
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "requirements.lock"

# Packages that are part of the *input* to the build rather than the shipped
# runtime (build backends, the installer itself). Everything else that is
# installed must appear in the lock.
EXCLUDED = {
    "pip",
    "setuptools",  # pinned <81 in requirements.txt; install-time only
    "wheel",
    "build",
    "pkg-resources",
}

# Names whose version is dictated by the CPU-only torch index rather than PyPI;
# they are still recorded so drift is visible.
PINNED_ELSEWHERE = ("torch", "torchaudio")

NAME_RE = re.compile(r"^(?P<name>[A-Za-z0-9_.\-]+)")


def canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def installed() -> dict[str, str]:
    """name -> version for every installed distribution we ship."""
    out: dict[str, str] = {}
    for dist in metadata.distributions():
        raw_name = dist.metadata["Name"]
        if not raw_name:
            continue
        name = canonical(raw_name)
        if name in EXCLUDED:
            continue
        # Phase 4 tooling is not shipped in the image.
        out[name] = dist.version
    return out


def lock_entries(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        if "==" not in line:
            continue
        name, _, version = line.partition("==")
        entries[canonical(name.strip())] = version.strip()
    return entries


def render(packages: dict[str, str]) -> str:
    header = LOCK.read_text().split("--index-url")[0]
    body = [f"{name}=={ver}" for name, ver in sorted(packages.items())]
    return header + "--index-url https://pypi.org/simple\n" + "\n".join(body) + "\n"


def declared_requirements() -> dict[str, str | None]:
    """
    Names required by requirements.txt / pyproject.toml, mapped to a pinned
    version when the declaration uses '==' (or None for a range).

    Parsed textually on purpose: running a resolver here would drag in the
    TensorFlow tree that basic-pitch declares but this project does not install.
    """
    declared: dict[str, str | None] = {}

    req = ROOT / "requirements.txt"
    if req.exists():
        for raw in req.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith(("#", "-")):
                continue
            name_part = line.split(";")[0].strip()
            match = NAME_RE.match(name_part)
            if not match:
                continue
            name = canonical(match.group("name"))
            version = name_part.split("==")[1].strip() if "==" in name_part else None
            declared[name] = version

    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        import re as _re

        text = pyproject.read_text()
        block = _re.search(r"^dependencies = \[(.*?)^\]", text, _re.S | _re.M)
        if block:
            for item in _re.findall(r'"([^"]+)"', block.group(1)):
                match = NAME_RE.match(item.strip())
                if not match:
                    continue
                name = canonical(match.group("name"))
                declared.setdefault(name, item.split("==")[1].strip() if "==" in item else None)
    return declared


def check_static() -> int:
    """
    Pre-install check: every declared dependency must be locked.

    Runs in CI before the heavy install, so a newly added dependency that was
    not locked fails in seconds instead of after a multi-minute install.
    """
    if not LOCK.exists():
        print(
            f"{LOCK.name} is missing. Run: python tools/check_lockfile.py --write", file=sys.stderr
        )
        return 1

    locked = lock_entries(LOCK.read_text())
    declared = declared_requirements()
    # Build/install-time only: excluded from the runtime lock by design, and
    # documented in SECURITY.md (setuptools<81 is a load-bearing pin).
    declared = {n: v for n, v in declared.items() if n not in EXCLUDED}
    missing = sorted(name for name in declared if name not in locked)

    if missing:
        print(f"{LOCK.name} does not cover these declared dependencies:", file=sys.stderr)
        for name in missing:
            print(f"  ✗ {name}", file=sys.stderr)
        print("\nRegenerate with: python tools/check_lockfile.py --write", file=sys.stderr)
        return 1

    pinned = [n for n, v in declared.items() if v]
    mismatched = sorted(
        f"{name} declared=={version} but locked=={locked[name]}"
        for name, version in declared.items()
        if version and name in locked and locked[name] != version
    )
    if mismatched:
        print(f"{LOCK.name} disagrees with pinned declarations:", file=sys.stderr)
        for line in mismatched:
            print(f"  ~ {line}", file=sys.stderr)
        return 1

    print(
        f"{LOCK.name} covers all {len(declared)} declared dependencies "
        f"({len(pinned)} exact pins verified)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    write = "--write" in argv

    if "--static" in argv:
        return check_static()

    current = installed()

    if write:
        LOCK.write_text(render(current))
        print(f"Wrote {LOCK.name} with {len(current)} pinned packages.")
        return 0

    if not LOCK.exists():
        print(
            f"{LOCK.name} is missing. Run: python tools/check_lockfile.py --write", file=sys.stderr
        )
        return 1

    locked = lock_entries(LOCK.read_text())

    missing = sorted(set(current) - set(locked))
    stale = sorted(set(locked) - set(current))
    changed = sorted(
        f"{name} {locked[name]} -> {current[name]}"
        for name in set(locked) & set(current)
        if locked[name] != current[name]
    )

    if missing or stale or changed:
        print(f"{LOCK.name} is out of date with the installed environment:")
        for name in missing:
            print(f"  + {name}=={current[name]} is installed but not locked")
        for name in stale:
            print(f"  - {name}=={locked[name]} is locked but not installed")
        for line in changed:
            print(f"  ~ {line}")
        print("\nRegenerate with: python tools/check_lockfile.py --write")
        return 1

    print(f"{LOCK.name} OK — {len(locked)} packages match the installed environment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
