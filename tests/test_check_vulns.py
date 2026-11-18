"""The vulnerability gate must accept an advisory under any of its identifiers.

PyPI/OSV re-canonicalise advisory IDs as issues are processed: torch advisories
recorded as PYSEC-2025-191 began being reported as GHSA-3749-ghw9-m3mg with no
change to the underlying vulnerability. Matching only the reported ID failed the
build for advisories a reviewer had already accepted under another name.
"""

import pytest

from tools.check_vulns import advisory_ids, lookup

IGNORES = {"PYSEC-2025-191": {"expiry": "2027-01-31", "reason": "accepted"}}


def test_advisory_ids_prefers_reported_id_then_aliases():
    vuln = {
        "id": "GHSA-3749-ghw9-m3mg",
        "aliases": ["CVE-2025-2953", "PYSEC-2025-191"],
    }
    ids = advisory_ids(vuln)
    assert ids[0] == "GHSA-3749-ghw9-m3mg"
    assert "PYSEC-2025-191" in ids


def test_advisory_ids_deduplicates_and_tolerates_missing_aliases():
    assert advisory_ids({"id": "CVE-1"}) == ["CVE-1"]
    assert advisory_ids({"id": "CVE-1", "aliases": ["CVE-1", "GHSA-x"]}) == [
        "CVE-1",
        "GHSA-x",
    ]


def test_lookup_matches_an_alias_when_the_id_is_renamed():
    # The regression: reported id is not in the file, but an alias is.
    vuln = {"id": "GHSA-3749-ghw9-m3mg", "aliases": ["PYSEC-2025-191"]}
    matched = lookup(IGNORES, vuln)
    assert matched is not None
    assert matched[0] == "PYSEC-2025-191"


def test_lookup_returns_none_for_a_genuinely_new_advisory():
    vuln = {"id": "GHSA-new", "aliases": ["CVE-2099-1"]}
    assert lookup(IGNORES, vuln) is None


@pytest.mark.parametrize("vuln", [{"id": ""}, {"id": None}, {"aliases": ["X"]}])
def test_advisory_ids_always_yields_a_usable_key(vuln):
    assert advisory_ids(vuln)[0]
