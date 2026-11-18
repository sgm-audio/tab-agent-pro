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
requirements.lock — and fails on any reported advisory unless it is listed in
.pip-audit-ignore with a reason and an expiry date.

Usage:
    python tools/check_vulns.py                 # gate (exit 1 on unaccepted finding)
    python tools/check_vulns.py --list-accepted # show the accepted-risk table
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IGNORE_FILE = ROOT / ".pip-audit-ignore"


def load_ignores(path: Path = IGNORE_FILE) -> dict[str, dict[str, str]]:
    """
    Parse the ignore file.

    Format (one entry per advisory, '#' comments allowed):
        <ADVISORY-ID>  <YYYY-MM-DD expiry>  <reason...>
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
    Every identifier an advisory is known by, reported id first.

    pip-audit collapses results that share an alias set and keeps whichever
    record it saw first (pip_audit/_audit.py, _service/interface.py), so the
    reported `id` flips between PYSEC-*, GHSA-* and CVE-* as the advisory
    databases are refreshed. The vulnerability is unchanged by the rename, so
    any of its identifiers may be the key in .pip-audit-ignore.
    """
    ids: list[str] = []
    for candidate in [vuln.get("id") or "UNKNOWN", *(vuln.get("aliases") or [])]:
        if candidate and candidate not in ids:
            ids.append(candidate)
    return ids


def lookup(ignores: dict[str, dict[str, str]], vuln: dict) -> tuple[str, dict[str, str]] | None:
    """Return the accepted-risk entry for an advisory, under whatever name it has."""
    for advisory in advisory_ids(vuln):
        entry = ignores.get(advisory)
        if entry is not None:
            return advisory, entry
    return None


def describe(vuln: dict) -> str:
    """Render an unaccepted advisory with its other names, so a rename is obvious."""
    ids = advisory_ids(vuln)
    if len(ids) == 1:
        return ids[0]
    return f"{ids[0]} (aliases: {', '.join(ids[1:])})"


# pip-audit resolves every installed package against PyPI, so one slow response
# can abort an otherwise clean scan with a ReadTimeout and no JSON on stdout.
# That is a transport failure, not a finding, and it reddened CI while the
# advisory set was unchanged. Retrying is safe because pip-audit is read-only,
# and a real finding still exits 1 with valid JSON, which is never retried.
AUDIT_ATTEMPTS = 3
AUDIT_BACKOFF_SECONDS = 5.0
# pip-audit defaults to a 15 s per-request timeout and issues one request per
# installed package (~120). A loaded PyPI exceeds 15 s often enough that the
# common case -- a slow index -- should be a slow success, not a red build.
AUDIT_TIMEOUT_SECONDS = 90


def audit(attempts: int = AUDIT_ATTEMPTS) -> list[dict]:
    """Run pip-audit against the installed environment and return its JSON.

    Transient failures are retried with linear backoff. A response that parses
    is returned immediately even when it reports findings: findings are the
    caller's business, and retrying them would only delay a failing gate.
    """
    last = ""
    for attempt in range(1, attempts + 1):
        result = subprocess.run(  # nosec B603, B607
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--progress-spinner",
                "off",
                "--format",
                "json",
                "--timeout",
                str(AUDIT_TIMEOUT_SECONDS),
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.stdout.strip():
            try:
                return json.loads(result.stdout).get("dependencies", [])
            except json.JSONDecodeError:
                last = f"unparseable JSON (exit {result.returncode}): {result.stdout[:200]}"
        else:
            last = f"no output (exit {result.returncode}): {result.stderr[-300:]}"
        if attempt < attempts:
            print(
                f"⚠️  pip-audit attempt {attempt}/{attempts} failed "
                f"({last.splitlines()[-1][:120]}); "
                f"retrying in {AUDIT_BACKOFF_SECONDS * attempt:.0f}s",
                file=sys.stderr,
            )
            time.sleep(AUDIT_BACKOFF_SECONDS * attempt)
    msg = f"pip-audit produced no usable output after {attempts} attempts: {last}"
    raise RuntimeError(msg)


def main(argv: list[str] | None = None, ignore_path: Path | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ignores = load_ignores(ignore_path or IGNORE_FILE)

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
            if match is None:
                unaccepted.append((dep["name"], dep["version"], describe(vuln)))
                continue
            advisory, entry = match
            if dt.date.fromisoformat(entry["expiry"]) < today:
                expired.append(advisory)
                unaccepted.append((dep["name"], dep["version"], advisory))
            else:
                accepted.append((advisory, dep["name"]))

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
