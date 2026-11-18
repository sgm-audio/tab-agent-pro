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

    def test_torch_index_requires_cpu_default_and_actual_install_use(self) -> None:
        guard = _load("check_docker_install")
        cpu = "https://download.pytorch.org/whl/cpu"
        assert guard._has_cpu_torch_install(
            f'ARG TORCH_INDEX_URL={cpu}\nRUN pip install torch==2.5.1 --index-url "$TORCH_INDEX_URL"'
        )
        assert guard._has_cpu_torch_install(f"RUN pip install torch==2.5.1 --index-url {cpu}")
        assert not guard._has_cpu_torch_install(
            f"ARG TORCH_INDEX_URL={cpu}\nRUN pip install torch==2.5.1"
        )
        assert not guard._has_cpu_torch_install(
            'ARG TORCH_INDEX_URL=https://example.com\nRUN pip install torch==2.5.1 --index-url "$TORCH_INDEX_URL"'
        )
        assert not guard._has_cpu_torch_install(f"# RUN pip install torch --index-url {cpu}")

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

    def test_cpu_index_check_resolves_docker_build_arg_default(self) -> None:
        guard = _load("check_docker_install")
        text = (ROOT / "Dockerfile").read_text()
        assert guard._has_cpu_torch_install(text)
        assert not guard._has_cpu_torch_install(
            text.replace(
                "ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu",
                "ARG TORCH_INDEX_URL=https://pypi.org/simple",
            )
        )
        assert not guard._has_cpu_torch_install(
            text.replace('--index-url "$TORCH_INDEX_URL"', "--index-url https://pypi.org/simple")
        )

    def test_in_process_main_and_failure_paths(self, tmp_path, monkeypatch) -> None:
        guard = _load("check_docker_install")
        assert guard.main() == 0

        # Simulate broken Dockerfile / requirements / app host binding in tmp_path
        (tmp_path / "Dockerfile").write_text("RUN pip install basic-pitch>=0.4.0\n")
        (tmp_path / "requirements.txt").write_text("numpy\n")
        (tmp_path / "app.py").write_text('os.getenv("HOST", "127.0.0.1")\n')
        monkeypatch.setattr(guard, "ROOT", tmp_path)
        monkeypatch.setattr(guard, "failures", [])
        assert guard.main() == 1
        assert len(guard.failures) >= 3


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
        vars_found = guard.documented_variables()
        sources = guard.sources()
        missing = [name for name in vars_found if name not in sources]
        assert missing == []
        assert guard.main() == 0

    def test_detects_unused_env_var(self, tmp_path, monkeypatch) -> None:
        guard = _load("check_env_example")
        env_file = tmp_path / ".env.example"
        env_file.write_text("NEVER_USED_VAR_XYZ=1\n")
        monkeypatch.setattr(guard, "ENV_FILE", env_file)
        assert guard.main() == 1


class TestLockDependencyClosure:
    @pytest.mark.parametrize("mode", [[], ["--verify"]], ids=["exact", "verify"])
    def test_rejects_an_absent_required_transitive_dependency(
        self, tmp_path, monkeypatch, capsys, mode
    ) -> None:
        guard = _load("check_lockfile")
        root = tmp_path / "audit_root-1.0.dist-info"
        root.mkdir()
        (root / "METADATA").write_text(
            "Name: audit-root\nVersion: 1.0\nRequires-Dist: audit-missing>=1\n"
        )
        lock = tmp_path / "requirements.lock"
        lock.write_text("audit-root==1.0\n")
        monkeypatch.setattr(guard, "LOCK", lock)
        monkeypatch.setattr(guard, "declared_requirements", lambda: {"audit-root": "1.0"})
        monkeypatch.setattr(
            guard.metadata, "distributions", lambda: [guard.metadata.Distribution.at(root)]
        )
        assert guard.main(mode) == 1
        assert "audit-missing is required but not installed" in capsys.readouterr().err

        # Inactive markers, including unselected extras, must not become required.
        (root / "METADATA").write_text(
            "Name: audit-root\nVersion: 1.0\n"
            'Requires-Dist: audit-optional; extra == "example"\n'
            'Requires-Dist: audit-inactive; python_version < "3.0"\n'
        )
        assert guard.main(mode) == 0


class TestValidationSuite:
    def test_validate_quick_passes(self) -> None:
        import validate

        rc = validate.main(["--quick"])
        assert rc == 0, "\n".join(r for r in validate.RESULTS if "❌" in r)

    def test_failure_simulations_complete_without_stale_backend_assumptions(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "simulate_failures.py")],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "malformed Basic Pitch output raises" in result.stdout
        assert "demucs unavailable -> raw-audio passthrough" in result.stdout


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
        assert guard.main([str(ROOT / "index.xml")]) == 0
        assert guard.main([str(legacy)]) == 1
