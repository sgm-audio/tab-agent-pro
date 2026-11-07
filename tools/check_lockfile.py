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
    python tools/check_lockfile.py           # exact: lock == installed env
    python tools/check_lockfile.py --verify  # lock == closure of declared deps
    python tools/check_lockfile.py --write   # regenerate from this environment
    python tools/check_lockfile.py --interpreter [--offline]
                                             # every pin installs on *this* Python

A development virtualenv holds lint/test/audit tooling that is never shipped, so
the exact mode is only meaningful in a clean install of the shipped set. CI uses
--verify, which walks the dependency closure of requirements.txt/pyproject.toml
and compares that to the lock: it exercises the pinning contract without
dragging TensorFlow in, and without failing on tooling it deliberately does not
install.

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


def _public_version(name: str, version: str) -> str:
    """Drop the PEP 440 local tag (such as +cpu) for index-pinned packages."""
    if name in PINNED_ELSEWHERE:
        return version.split("+", 1)[0]
    return version


# The PyPI torch wheel declares the NVIDIA CUDA runtime as a conditional
# requirement on Linux; the CPU-only wheel this project ships does not. The image
# and CI both install from https://download.pytorch.org/whl/cpu precisely to
# avoid ~2.5 GB of unusable redistributables, so they are not part of the
# shipped set and must not be locked - locking them would describe an image we
# never build (and made `pip install -r requirements.lock` pull CUDA back in).
CUDA_PREFIXES = ("nvidia-", "triton")


def is_cuda_redistributable(name: str) -> bool:
    return canonical(name).startswith(CUDA_PREFIXES)


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
        if name in EXCLUDED or is_cuda_redistributable(name):
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
        name = canonical(name.strip())
        if is_cuda_redistributable(name):
            continue
        entries[name] = version.strip()
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


PYPI_TIMEOUT = 12


def _requires_python(name: str, version: str) -> str | None:
    """PEP 503 `requires_python` for one pin, or None when offline/unknown."""
    import json as _json
    import urllib.request

    try:
        url = f"https://pypi.org/pypi/{name}/{version}/json"
        with urllib.request.urlopen(url, timeout=PYPI_TIMEOUT) as resp:  # nosec B310 - fixed host
            return _json.load(resp)["info"].get("requires_python") or ""
    except Exception:
        return None


def check_interpreter(skip_network: bool = False) -> int:
    """
    Every pin must install on the interpreter the image actually runs.

    This exists because a lock generated on one Python can be silently
    uninstallable on another: 16 of this lock's pins require >=3.11, which made
    `pip install -r requirements.lock` fail on a 3.10 CI runner while the image
    (3.12) was fine. Without this check the mismatch only shows up as a red
    build after a multi-minute install.
    """
    import re as _re

    if not LOCK.exists():
        print(f"{LOCK.name} is missing.", file=sys.stderr)
        return 1

    locked = lock_entries(LOCK.read_text())
    target = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"checking {len(locked)} pins against Python {target}")

    unknown = 0
    incompatible: list[str] = []
    for name, version in sorted(locked.items()):
        spec = None if skip_network else _requires_python(name, version)
        if spec is None:
            unknown += 1
            continue
        if not spec:
            continue
        match = _re.search(r">=\s*([0-9]+)\.([0-9]+)", spec)
        if match:
            floor = (int(match.group(1)), int(match.group(2)))
            if floor > sys.version_info[:2]:
                incompatible.append(f"{name}=={version} requires Python {spec}")

    if incompatible:
        print("These pins cannot install on the running interpreter:", file=sys.stderr)
        for line in incompatible:
            print(f"  ✗ {line}", file=sys.stderr)
        print(
            "\nAlign the CI runner with the image (Dockerfile) or regenerate the "
            "lock on the shipped interpreter.",
            file=sys.stderr,
        )
        return 1

    note = f" ({unknown} unverified: no PyPI metadata / offline)" if unknown else ""
    print(f"All {len(locked) - unknown} verifiable pins install on Python {target}{note}.")
    return 0


def _walk_closure(roots: set[str]) -> set[str]:
    """Transitive dependency closure of `roots` over the installed metadata."""
    import re as _re

    dists = {}
    for dist in metadata.distributions():
        raw = dist.metadata["Name"]
        if raw:
            dists[canonical(raw)] = dist

    seen: set[str] = set()
    todo = [name for name in roots if name in dists]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for req in dists[name].requires or []:
            if "extra ==" in req:  # optional feature, not part of the base install
                continue
            match = _re.match(r"\s*([A-Za-z0-9_.\-]+)", req)
            if not match:
                continue
            dep = canonical(match.group(1))
            if is_cuda_redistributable(dep):
                continue  # CPU-only torch: never installed in the image
            if dep not in seen and dep in dists:
                todo.append(dep)
    return seen


def check_verify(extra: list[str] | None = None) -> int:
    """
    Verify the lock describes the closure of the *declared* dependencies.

    Every declared dependency and every transitive dependency the resolver would
    install must be locked at a version that is actually installed; and the lock
    must not carry entries outside that closure unless they are named on the
    command line via --extra (tooling installed for the check itself).
    """
    if not LOCK.exists():
        print(
            f"{LOCK.name} is missing. Run: python tools/check_lockfile.py --write", file=sys.stderr
        )
        return 1

    locked = lock_entries(LOCK.read_text())
    declared = declared_requirements()
    # Exact pins in requirements.txt (e.g. `IPython>=8.0,<8.30`) are walk roots
    # too: they describe installed packages whose own extra requirements are
    # part of the shipped set even when nothing else declares them.
    roots = {name for name in declared if name not in EXCLUDED}
    exact = {name for name, version in declared.items() if version and name not in EXCLUDED}
    # setuptools is declared (pinned <81) but excluded from the runtime lock: it
    # is install-time only. It is still part of the closure, so re-add it as a
    # walk root and drop it from the comparison instead of hiding it silently.
    walk_roots = set(roots) | {"setuptools"}

    current = installed()
    closure = _walk_closure(walk_roots | exact) - EXCLUDED
    tolerated = {canonical(name) for name in (extra or [])}

    unlocked = sorted(name for name in closure if name not in locked)
    drifted = sorted(
        f"{name} locked=={locked[name]} but installed=={current[name]}"
        for name in closure & set(locked) & set(current)
        if _public_version(name, locked[name]) != _public_version(name, current[name])
    )
    absent = sorted(
        f"{name} locked=={locked[name]} but not installed" for name in closure - set(current)
    )
    extra_entries = sorted(set(locked) - closure - EXCLUDED - tolerated)

    problems = False
    if unlocked:
        problems = True
        print(f"{LOCK.name} is missing {len(unlocked)} shipped dependencies:", file=sys.stderr)
        for name in unlocked:
            print(f"  + {name}=={current.get(name, '?')} (in the install closure)", file=sys.stderr)
    if drifted:
        problems = True
        print(f"{LOCK.name} pins versions that are not installed:", file=sys.stderr)
        for line in drifted:
            print(f"  ~ {line}", file=sys.stderr)
    if absent:
        problems = True
        print(f"{LOCK.name} pins packages that are absent:", file=sys.stderr)
        for line in absent:
            print(f"  - {line}", file=sys.stderr)
    if extra_entries:
        problems = True
        print(
            f"{LOCK.name} contains entries outside the shipped closure "
            f"(not tooling declared with --extra):",
            file=sys.stderr,
        )
        for name in extra_entries:
            print(f"  ? {name}=={locked[name]}", file=sys.stderr)

    if problems:
        print(
            "\nRegenerate with: python tools/check_lockfile.py --write "
            "(from a clean install of the shipped set)",
            file=sys.stderr,
        )
        return 1

    print(
        f"{LOCK.name} OK — closure of {len(roots)} declared dependencies "
        f"({len(closure)} packages) is fully and exactly pinned"
        + (f"; {len(tolerated)} tooling package(s) tolerated" if tolerated else "")
        + "."
    )
    return 0


def main(argv: list[str] | None = None) -> int:

    argv = sys.argv[1:] if argv is None else argv
    write = "--write" in argv

    if "--static" in argv:
        return check_static()

    extras = [argv[i + 1] for i, a in enumerate(argv) if a == "--extra" and i + 1 < len(argv)]
    if "--verify" in argv:
        return check_verify(extras)

    if "--interpreter" in argv:
        return check_interpreter(skip_network="--offline" in argv)

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
        if _public_version(name, locked[name]) != _public_version(name, current[name])
    )

    if missing or stale or changed:
        print(
            f"{LOCK.name} is out of date with the installed environment.\n"
            "(If this environment holds lint/test tooling that is not shipped, use "
            "--verify instead of exact mode.)"
        )
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
