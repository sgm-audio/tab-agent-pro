#!/usr/bin/env python3
"""
Dependency vulnerability gate for the *shipped* dependency set.

Design notes
------------
`pip-audit -r requirements.txt` audits what the resolver would install, which
includes the TensorFlow chain that basic-pitch declares on Linux and this
project deliberately does not install (see SECURITY.md). That produces ~60
findings in packages that never reach the image, which drowns the ones that
matter.

This gate audits the installed environment instead — the same set pinned in
requirements.lock — and fails on HIGH/CRITICAL findings unless the advisory ID
is listed in .pip-audit-ignore with a reason and an expiry date.

Usage:
    python tools/check_vulns.py                 # gate (exit 1 on unaccepted finding)
    python tools/check_vulns.py --list-accepted # show the accepted-risk table
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess  # nosec B404
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IGNORE_FILE = ROOT / ".pip-audit-ignore"


def load_ignores(path: Path = IGNORE_FILE) -> dict[str, dict[str, str]]:
    """
    Parse the ignore file.

    Format (one entry per advisory, '#' comments allowed):
        <ADVISORY-ID>  <YYYY-MM-DD expiry>  <reason...>

    The advisory ID may be any of the one advisory's identifiers. PyPI/OSV
    re-canonicalise as issues are processed: torch advisories accepted as
    ``PYSEC-2025-191`` started being reported as ``GHSA-3749-ghw9-m3mg`` with no
    change to the vulnerability. Matching only the reported ID would fail the
    build for an advisory a reviewer has already accepted under another name.
    """
    entries: dict[str, dict[str, str]] = {}
    if not path.exists():
        return entries
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            print(f"⚠️  Malformed ignore entry: {raw!r}", file=sys.stderr)
            continue
        advisory, expiry, reason = parts
        entries[advisory] = {"expiry": expiry, "reason": reason}
    return entries


def advisory_ids(vuln: dict) -> list[str]:
    """
    Every identifier an advisory is known by, most canonical first.

    pip-audit reports ``id`` plus, when the source provides them, ``aliases``
    (CVE / GHSA / PYSEC). Which one is "the" id is not stable between advisory
    DB refreshes, so all of them are usable keys in .pip-audit-ignore.
    """
    ids = [vuln.get("id") or "UNKNOWN"]
    ids.extend(vuln.get("aliases") or [])
    seen: list[str] = []
    for advisory in ids:
        if advisory and advisory not in seen:
            seen.append(advisory)
    return seen


def lookup(ignores: dict[str, dict[str, str]], vuln: dict) -> tuple[str, dict] | None:
    """Find the accepted-risk entry for an advisory under any of its names."""
    for advisory in advisory_ids(vuln):
        entry = ignores.get(advisory)
        if entry is not None:
            return advisory, entry
    return None


def audit() -> list[dict]:
    """Run pip-audit against the installed environment and return its JSON."""
    result = subprocess.run(  # nosec B603, B607
        [sys.executable, "-m", "pip_audit", "--progress-spinner", "off", "--format", "json"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if not result.stdout.strip():
        msg = f"pip-audit produced no output (exit {result.returncode}): {result.stderr[-500:]}"
        raise RuntimeError(msg)
    return json.loads(result.stdout).get("dependencies", [])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ignores = load_ignores()

    if "--list-accepted" in argv:
        if not ignores:
            print("No accepted-risk advisories recorded.")
            return 0
        print(f"{'ADVISORY':24s} {'EXPIRES':12s} REASON")
        for advisory, meta in sorted(ignores.items()):
            print(f"{advisory:24s} {meta['expiry']:12s} {meta['reason']}")
        return 0

    try:
        deps = audit()
    except (RuntimeError, subprocess.SubprocessError) as e:
        print(f"❌ vulnerability scan failed to run: {e}", file=sys.stderr)
        return 1

    today = dt.date.today()
    unaccepted: list[tuple[str, str, str]] = []
    accepted: list[tuple[str, str]] = []
    expired: list[str] = []

    for dep in deps:
        for vuln in dep.get("vulns") or []:
            match = lookup(ignores, vuln)
            if match:
                advisory, entry = match
                if dt.date.fromisoformat(entry["expiry"]) < today:
                    expired.append(advisory)
                    unaccepted.append((dep["name"], dep["version"], advisory))
                else:
                    accepted.append((advisory, dep["name"]))
                continue
            unaccepted.append((dep["name"], dep["version"], vuln.get("id", "UNKNOWN")))

    for advisory in sorted(set(expired)):
        print(
            f"❌ accepted-risk entry EXPIRED: {advisory} — re-review it in .pip-audit-ignore",
            file=sys.stderr,
        )

    if unaccepted:
        print(
            f"❌ {len(unaccepted)} unaccepted vulnerabilities in the shipped set:", file=sys.stderr
        )
        for name, version, advisory in sorted(set(unaccepted)):
            print(f"   {name} {version}: {advisory}", file=sys.stderr)
        print(
            "\nFix the dependency, or record an accepted risk with a reason and an "
            "expiry date in .pip-audit-ignore.",
            file=sys.stderr,
        )
        return 1

    print(
        f"✅ No unaccepted vulnerabilities. "
        f"({len(accepted)} advisory/advisories recorded as accepted risk; "
        f"{len(deps)} packages scanned)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
