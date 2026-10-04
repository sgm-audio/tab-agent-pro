"""
Tests for the guard scripts in tools/ and for index.xml (docs-audit P0-2,
P1-4, P2-1).

These guard scripts run in CI; the tests here make sure the guards themselves
still catch the regressions they were written for.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(module_name: str):
    path = ROOT / "tools" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# tools/check_docker_install.py  (P0-2)
# ---------------------------------------------------------------------------


class TestDockerInstallGuard:
    def test_detects_unquoted_specifier(self) -> None:
        guard = _load("check_docker_install")
        offending = guard._unquoted_specifier("RUN pip install --no-deps basic-pitch>=0.4.0 && \\")
        assert offending, "unquoted version specifier must be flagged"

    def test_accepts_quoted_specifier(self) -> None:
        guard = _load("check_docker_install")
        line = 'RUN pip install --no-deps "basic-pitch>=0.4.0,<0.5" "mir_eval>=0.6" && \\'
        assert guard._unquoted_specifier(line) is None

    def test_repo_state_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "check_docker_install.py")],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_dockerfile_quotes_and_installs_dependencies(self) -> None:
        text = (ROOT / "Dockerfile").read_text()
        assert '"basic-pitch' in text
        assert "basic-pitch>=" in text
        assert "mir_eval" in text and "resampy" in text
        assert "HOST=0.0.0.0" in text


# ---------------------------------------------------------------------------
# tools/check_env_example.py  (P2-1)
# ---------------------------------------------------------------------------


class TestEnvExampleGuard:
    def test_repo_state_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "check_env_example.py")],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_documented_variables_are_read(self) -> None:
        guard = _load("check_env_example")
        sources = guard.sources()
        missing = [name for name in guard.documented_variables() if name not in sources]
        assert missing == []


# ---------------------------------------------------------------------------
# index.xml  (P1-4)
# ---------------------------------------------------------------------------


class TestReaPackIndex:
    def test_index_is_valid(self) -> None:
        guard = _load("check_index")
        problems = guard.check_index(ROOT / "index.xml")
        assert problems == [], "index.xml problems:\n" + "\n".join(problems)

    def test_guard_detects_old_format(self, tmp_path) -> None:
        """The guard must reject the pre-fix index (no commit, version='1.0')."""
        guard = _load("check_index")
        legacy = tmp_path / "index.xml"
        legacy.write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<index name="Tab Agent Pro" version="1.0" '
            'xmlns="http://www.cockos.com/reapack/schema/1.0">\n'
            '  <category name="Transcription"><description>x</description></category>\n'
            '  <reaper name="Tab Agent" type="script">\n'
            '    <version name="1.0.0" author="Scott Mills" type="main">\n'
            '      <source file="reaper/TabAgent.lua">TabAgent.lua</source>\n'
            "    </version>\n"
            "  </reaper>\n"
            "</index>\n"
        )
        problems = guard.check_index(legacy)
        joined = " ".join(problems)
        assert any("version" in p for p in problems)
        assert "commit" in joined
        assert "<reaper>" in joined or "no <reapack> packages" in joined


# ---------------------------------------------------------------------------
# tools/check_reaper_settings.py  (P1-3)
# ---------------------------------------------------------------------------


class TestReaperSettingsGuard:
    def test_repo_state_passes(self) -> None:
        pytest.importorskip("lupa")
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "check_reaper_settings.py")],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_tab_agent_lua_forwards_settings(self) -> None:
        lua = (ROOT / "reaper" / "TabAgent.lua").read_text()
        for flag in (
            "--profile",
            "--instrument",
            "--onset",
            "--frame",
            "--no-midi",
            "--output-dir",
        ):
            # The Lua builds these with string concatenation, e.g. "--onset "
            token = flag.rstrip("0123456789")
            assert token in lua, f"TabAgent.lua never passes {flag}"

    def test_settings_lua_exposes_thresholds_and_output_dir(self) -> None:
        lua = (ROOT / "reaper" / "Settings.lua").read_text()
        for key in ("onset", "frame", "output_dir"):
            assert f'"{key}"' in lua, f"Settings.lua does not persist {key}"
