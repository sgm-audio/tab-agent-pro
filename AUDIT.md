# Tab Agent Pro — Full Codebase Audit & Production Readiness Report

**Date:** 2026-07-01
**Project:** tab-agent-pro (scottmills306/tab-agent-pro)
**Commit:** `7a25561`
**Python:** 3.14.6 | **Tools:** ruff 0.15.17, mypy 2.1.0, black 26.5.1, bandit 1.9.4, pytest 9.0.3

---

## 1. Static Analysis Summary

| Tool | Result |
|------|--------|
| **ruff** | **71 errors** (46 auto-fixable) |
| **mypy** | **30+ type errors** (mostly missing stubs, some real issues) |
| **black** | **13 files need reformatting** |
| **bandit** | **0 CRITICAL/HIGH** (12 MEDIUM, 23 LOW in project code) |
| **isort** | Not configured |

---

## 2. Tests Summary

```
46 collected → 41 passed, 2 failed, 3 skipped
Coverage: 47% overall (1941 stmts, 1020 missed)
```

### Failing Tests

| Test | Issue | Root Cause |
|------|-------|------------|
| `test_ear_agent_transcribes_synthetic` | `RuntimeError: No transcription models available` | Basic Pitch not installed in environment |
| `test_export_tab_to_txt` | `'0s' not found in ASCII tab` | Output format mismatch — string spacing produces `0  ` instead of `0s` |

### Coverage Per File

| File | Coverage | Notes |
|------|----------|-------|
| `suno_postprocessor.py` | **93%** | ✅ Well-tested |
| `test_pipeline.py` | 93% | Self-testing |
| `tests/test_tab.py` | 99% | ✅ |
| `tests/test_suno.py` | 99% | ✅ |
| `tests/test_splitter.py` | 98% | ✅ |
| `tests/test_ear.py` | 99% | ✅ |
| `agents.py` | **45%** | ❌ Most YourMT3+ paths never exercised |
| `monitoring.py` | 72% | Acceptable |
| `main.py` | **23%** | ❌ CLI pipeline mostly untested |
| `app.py` | **0%** | ❌ Gradio UI completely untested |
| `init_memory.py` | **0%** | ❌ Untested |
| `validate.py` | **0%** | ❌ Self-validation script, but not run in CI |

---

## 3. Issues Found

### CRITICAL (0)
None found.

### HIGH (5)

| # | File | Line | Issue | Fix |
|---|------|------|-------|-----|
| H1 | `agents.py` | 491 | **Unsafe PyTorch load**: `weights_only=False` allows arbitrary code execution from untrusted checkpoints | Use `weights_only=True` or validate checkpoint source |
| H2 | `agents.py` | 368 | **Unpinned HuggingFace Hub download**: `snapshot_download` without `revision=` pins to `main` branch — supply-chain risk | Add `revision="<commit-sha>"` parameter |
| H3 | `app.py` | 400 | **Binds to all interfaces**: `host="0.0.0.0"` with no auth layer — any network peer can access the Gradio API | Add auth, or document "internal network only" |
| H4 | `agents.py` | 242-257 | **YourMT3+ model reference broken**: `mimbres/YourMT3-cpu` returns HTTP 401 — core transcription model unavailable | Remove dead code path or fix model ID |
| H5 | `agents.py` | 1265 | **Function too long**: `generate_tab` is 107 lines with nested loops and DP logic — hard to test/verify | Extract cost computation and technique annotation into separate methods |

### MEDIUM (12)

| # | File | Line | Issue | Fix |
|---|------|------|-------|-----|
| M1 | `agents.py` | 15 | **`subprocess` import** (bandit B404) — potential command injection surface | Validate audio_path before passing to demucs CLI. Currently no path sanitization |
| M2 | `agents.py` | 394-412 | **Git subprocess calls use partial paths** — `git` resolution depends on `$PATH` | Use full path or validate binary |
| M3 | `agents.py` | 30 | **Unused import**: `ICASSP_2022_MODEL_PATH` imported but never used | Remove |
| M4 | `agents.py` | 58 | **Unused import**: `torchaudio` imported but never used | Remove |
| M5 | `app.py` | 9-10, 17-18, 21 | **6 unused imports**: `os`, `sys`, `numpy`, `json`, `get_logger`, `WSGIMiddleware`, `threading` | Remove |
| M6 | `monitoring.py` | 27-28, 30 | **4 unused imports**: `os`, `functools`, `List`, `Callable` | Remove |
| M7 | `validate.py` | 104-149 | **Many unused imports** with bare `try/except/pass` swallowing errors | Remove or use `importlib.util.find_spec` |
| M8 | Multiple | various | **13 files need black formatting** — inconsistent style | Run `black .` |
| M9 | Multiple | various | **46 f-strings without placeholders** (ruff F541) — e.g. `f"✅ Guitar processing complete"` | Use plain strings |
| M10 | `agents.py` | 172 | **`subprocess.run` result never checked** in CLI fallback path | Validate return code |
| M11 | `suno_postprocessor.py` | 57 | **Unused variable** `S_db` computed but never used | Remove |
| M12 | `validate.py` | 330-332 | **Ambiguous variable name `l`** (ruff E741) + unused `untracked` var | Rename to `line` |

### LOW / NIT (many)

| # | File | Issue |
|---|------|-------|
| L1 | `agents.py`, `main.py`, `init_memory.py` | ~30 f-strings without placeholders — use plain `print("msg")` instead of `print(f"msg")` |
| L2 | `agents.py:876,1023,1243` | Missing type annotations for `notes`, `seen_pitches`, `final_tab` (mypy `var-annotated`) |
| L3 | `validate.py:139-142` | `assert` in test code — removed under `-O` flag |
| L4 | `validate.py:104-120` | Bare `try/except/pass` swallows all import errors |
| L5 | `main.py` | Heavy use of `print()` instead of structured logging — no log levels, timestamps, or JSON output for CLI mode |
| L6 | `agents.py:317,378,525,632,909,911` | F-strings without placeholders in print statements |

---

## 4. Test Gap Analysis

| Area | Coverage | What's Missing |
|------|----------|----------------|
| **Gradio UI** (`app.py`) | 0% | No tests for UI rendering, event handlers, file upload, ZIP generation |
| **CLI pipeline** (`main.py`) | 23% | No tests for argument parsing, file I/O, user memory, session logging |
| **YourMT3+ integration** | 0% | All YourMT3+ codepaths untested (model unreachable) |
| **Tab export** (ASCII/JSON) | Partial | ASCII tab format subtly broken (fret spacing, technique markers) |
| **Error handling** | Low | No tests for Demucs failures, missing models, corrupt audio |
| **Security** | 0% | No tests for path traversal, model tampering, subprocess injection |

---

## 5. Infrastructure Gaps (Production Readiness)

| Area | Status | What's Needed |
|------|--------|---------------|
| **`pyproject.toml`** | ❌ Missing | No project metadata, no tool configs for ruff/black/mypy/isort |
| **Lock file** | ❌ Missing | No `poetry.lock` or `pip freeze` — builds are non-reproducible |
| **CI/CD** | ❌ Missing | No GitHub Actions workflow for lint, type-check, test, build |
| **Pre-commit hooks** | ❌ Missing | No automated formatting/linting before commits |
| **Docker optimization** | ⚠️ Partial | Multi-stage build? Layer caching improved? Missing step #6 comment |
| **Health endpoint auth** | ❌ Missing | `/health` and `/health/metrics` are public with no auth |
| **Structured logging (CLI)** | ❌ Missing | `main.py` uses `print()` — monitoring module exists but unused in CLI path |
| **YourMT3+ model** | ❌ Broken | Model ID returns 401 — dead code path with ~200 lines of complex loading logic |
| **Basic Pitch install** | ⚠️ Partial | Listed in `requirements.txt` but fails to load (missing deps or version conflict) |
| **Type checking in CI** | ❌ Missing | mypy errors won't block merges |
| **Security scanning** | ❌ Missing | No bandit or safety check in pipeline |
| **Environment variables** | ⚠️ Partial | No `.env.example` for API keys or config overrides |

---

## 6. Production Readiness Plan

### Phase 1 — Fix the Blockers (1-2 hours)

1. **Fix the 2 failing tests**
   - `test_export_tab_to_txt`: Fix ASCII tab format — technique markers (`0s`) getting lost in spacing logic (`main.py:74-89`)
   - `test_ear_agent_transcribes_synthetic`: Install `basic-pitch` or update test to skip gracefully

2. **Fix HIGH security issues**
   - `agents.py:491`: Change `weights_only=False` → `weights_only=True` (or document why unsafe load is necessary)
   - `agents.py:368`: Add `revision=` pin to YourMT3+ download

3. **Auto-fix 46 ruff issues**
   ```bash
   ruff check --fix .
   ```

4. **Format with black**
   ```bash
   black .
   ```

### Phase 2 — Build Infrastructure (2-3 hours)

5. **Create `pyproject.toml`** with:
   - Project metadata (name, version, python requires)
   - Tool configs: `[tool.ruff]`, `[tool.black]`, `[tool.mypy]`, `[tool.pytest]`
   - Dependency groups (main vs dev)

6. **Create GitHub Actions CI** (`.github/workflows/ci.yml`)
   ```yaml
   - ruff check
   - black --check
   - mypy .
   - pytest --cov --cov-fail-under=50
   - bandit -r agents.py main.py app.py
   ```

7. **Add pre-commit config** (`.pre-commit-config.yaml`)

### Phase 3 — Increase Coverage (3-4 hours)

8. **Add unit tests for `main.py`** (CLI pipeline)
   - Test `export_tab_to_txt` with various tab data
   - Test `export_tab_to_json` output format
   - Test `load_user_memory` with/without file

9. **Add unit tests for `init_memory.py`**
   - Test profile loading, `save_profile`, `list_profiles`

10. **Add smoke test for `app.py`**
    - Test Gradio UI creation (no need to render)
    - Test health endpoint response format

### Phase 4 — Polish (1-2 hours)

11. **Remove dead code**
    - YourMT3+ model loading (~200 lines) if model is permanently broken
    - Unused imports across all files

12. **Switch CLI to structured logging**
    - Replace `print()` in `main.py` with `PipelineLogger` from `monitoring.py`

13. **Fix ASCII tab output format** — ensure technique markers render correctly

14. **Add `.env.example`** and document all configuration knobs

### Phase 5 — Documentation & Release (1 hour)

15. **Update README** with badges (CI, coverage, Python version)
16. **Tag v1.0.0 release** with changelog

---

## 7. Recommendations

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| 🔴 P0 | Fix 2 failing tests | 30 min | Unblock CI |
| 🔴 P0 | Fix torch.load `weights_only=False` | 5 min | Security |
| 🟡 P1 | `ruff check --fix .` + `black .` | 5 min | 46 issues gone |
| 🟡 P1 | Create `pyproject.toml` | 15 min | Foundation |
| 🟡 P1 | Add CI workflow | 30 min | Automated gates |
| 🟢 P2 | Test main.py + init_memory.py | 1-2 hr | Coverage to 60%+ |
| 🟢 P2 | Remove dead YourMT3+ code or fix model ID | 30 min | -200 LOC |
| 🔵 P3 | Switch CLI to structured logging | 30 min | Observability |
| 🔵 P3 | Add pre-commit hooks | 15 min | Consistency |
| ⚪ P4 | Tag release, update README | 15 min | Polish |

---

## 8. Quick Wins (Do First)

```bash
# 1. Auto-fix lint issues
ruff check --fix .

# 2. Format everything
black .

# 3. Remove unused imports manually for non-auto-fixable ones
# (ruff --fix handles F401 but not F541 f-strings)

# 4. Install Basic Pitch so tests pass
pip install basic-pitch

# 5. Run tests to verify
pytest tests/ test_pipeline.py -v
```
