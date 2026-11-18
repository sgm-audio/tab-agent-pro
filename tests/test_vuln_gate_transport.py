"""The vulnerability gate must survive a slow index without inventing a finding.

pip-audit issues one PyPI request per installed package (~120) at a 15 s default
timeout, so a loaded index regularly produces a ReadTimeout with no JSON on
stdout. The gate treated that as "audit failed" and reddened CI while the
advisory set was unchanged. These tests pin the distinction between a transport
failure, which is retried, and a report of findings, which is not.
"""

import importlib.util

import pytest

from tests.test_tools import ROOT

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _guard():
    path = ROOT / "tools" / "check_vulns.py"
    spec = importlib.util.spec_from_file_location("check_vulns_transport", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Result:
    """The subset of subprocess.CompletedProcess that audit() reads."""

    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _no_sleep(monkeypatch, guard) -> None:
    monkeypatch.setattr(guard, "AUDIT_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(guard.time, "sleep", lambda _seconds: None)


def test_transient_read_timeout_is_retried(monkeypatch):
    guard = _guard()
    _no_sleep(monkeypatch, guard)
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Result(stdout="", returncode=1, stderr="requests.exceptions.ReadTimeout")
        return _Result(stdout='{"dependencies": [{"name": "x", "version": "1"}]}')

    monkeypatch.setattr(guard.subprocess, "run", fake_run)

    assert guard.audit(attempts=2) == [{"name": "x", "version": "1"}]
    assert calls["n"] == 2


def test_findings_are_returned_without_being_retried(monkeypatch):
    # A scan that reports findings exits 1 with valid JSON. Retrying it would only
    # delay a gate that is doing its job.
    guard = _guard()
    _no_sleep(monkeypatch, guard)
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return _Result(stdout='{"dependencies": [{"name": "y", "version": "2"}]}', returncode=1)

    monkeypatch.setattr(guard.subprocess, "run", fake_run)

    assert guard.audit(attempts=3) == [{"name": "y", "version": "2"}]
    assert calls["n"] == 1


def test_exhausted_attempts_raise_rather_than_pass(monkeypatch):
    # A gate that cannot verify must fail closed: never report "no vulnerabilities"
    # on the strength of an audit that never completed.
    guard = _guard()
    _no_sleep(monkeypatch, guard)

    monkeypatch.setattr(
        guard.subprocess,
        "run",
        lambda *a, **k: _Result(stdout="", returncode=1, stderr="ReadTimeout"),
    )

    with pytest.raises(RuntimeError, match="no usable output after 2 attempts"):
        guard.audit(attempts=2)


def test_unparseable_output_is_retried_then_raises(monkeypatch):
    guard = _guard()
    _no_sleep(monkeypatch, guard)

    monkeypatch.setattr(
        guard.subprocess,
        "run",
        lambda *a, **k: _Result(stdout="<html>proxy error</html>", returncode=0),
    )

    with pytest.raises(RuntimeError, match="unparseable JSON"):
        guard.audit(attempts=2)


def test_audit_raises_the_long_timeout_to_pip_audit(monkeypatch):
    # The whole point: do not let pip-audit keep its 15 s default.
    guard = _guard()
    _no_sleep(monkeypatch, guard)
    seen = {}

    def fake_run(cmd, *args, **kwargs):
        seen["cmd"] = cmd
        return _Result(stdout='{"dependencies": []}')

    monkeypatch.setattr(guard.subprocess, "run", fake_run)
    guard.audit()

    assert "--timeout" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--timeout") + 1] == str(guard.AUDIT_TIMEOUT_SECONDS)
