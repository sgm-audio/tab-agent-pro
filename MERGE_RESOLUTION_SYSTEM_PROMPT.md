# System Prompt — Resolve Severe Merge Conflicts in tab-agent-pro

## Mission

Merge `origin/main` into branch `arena/ebfd10d9-tab-agent-pro` (or vice versa — produce a clean tree that contains BOTH sides' work). The two branches diverged at merge-base `4061696bc2438002dbb3b5f1689385d87665e8b1` and both heavily modified the SAME 31 files. Expect **severe, file-level conflicts** in core pipeline files (`agents.py`, `app.py`, `main.py`, `Dockerfile`, `requirements*.txt`, `monitoring.py`, `pyproject.toml`) plus infrastructure files (`tools/check_lockfile.py`, `tools/check_vulns.py`, `tests/test_hardening.py`, `.github/workflows/ci.yml`).

## Working directory & starting state

```
cd "C:\Users\scott\Dev\tab-agent-pro"
git status            # must be clean
git branch --show-current  # arena/ebfd10d9-tab-agent-pro
git log --oneline -3   # HEAD = 019c174 (prompt commit) / 29d8814 (big refactor)
```

Merge-base: `4061696bc2438002dbb3b5f1689385d87665e8b1`

## Commit-by-commit divergence — read these first

Before touching any conflict, read both sides of history so you understand intent.

### Branch side (arena/ebfd10d9-tab-agent-pro) — the "refactor" direction

| Commit | What it did |
|--------|-------------|
| `29d8814` | **THE BIG ONE.** Removed Hugging Face entirely (YourMT3+/HF Hub, spaces.GPU/ZeroGPU, HF_TOKEN, `tools/deploy_space.sh`). Basic Pitch = sole transcription backend. Added `device.py` (local GPU: auto/cuda/rocm/vulkan/mps/directml/cpu). `Dockerfile` gained `TORCH_INDEX_URL` arg. Web UI: device selector + backend chip + inline tablature HTML. `reaper/TabAgent.lua`: `io.popen` + `reaper.defer` console streaming instead of blocking `os.execute`. Scrubbed HF from all docs. |
| `e09b8b9` / `d0af86d` / `4b9f52e` / `2373d28` | CI / lockfile gating tweaks (CPU torch pin, interpreter compatibility, lockfile gate fixes). |
| `1977671` | Merge of origin/main into a production-readiness branch (already contains some main-side changes). |
| `b1f2f55` | "Production readiness 1.1.0" — an earlier version of what main's `82077bb` also is. Likely overlapping content with main's commit. |

### Main side (origin/main) — the "production readiness 1.1.0" direction

| Commit | What it did |
|--------|-------------|
| `82077bb` | **THE BIG ONE on main.** Production readiness 1.1.0 (#9). Added: CI workflow (`.github/workflows/ci.yml`), `.pip-audit-ignore`, `AUDIT.md`, `CHANGELOG.md`, `SECURITY.md`, `THIRD_PARTY_NOTICES.md`, `docs/RUNBOOK.md` (234 lines), `tools/check_lockfile.py` (447 lines — THE lockfile gate), `tools/check_vulns.py`, `tools/simulate_failures.py`, `tests/test_hardening.py` (363 lines). Touched `agents.py`, `app.py`, `main.py`, `monitoring.py`, `Dockerfile`, `requirements*`, `index.xml`, `pyproject.toml`, `tests/*`, `.env.example`, `.github/workflows/ci.yml`. |

### The core conflict

BOTH sides rewrote `agents.py`, `app.py`, `main.py`, `monitoring.py`, `Dockerfile`, `requirements*.txt`, and `index.xml`. Main added production gating (lockfile, vuln scan, CI, hardening tests) while the branch removed HF and added local GPU. You must **keep both**: the branch's HF-removal + GPU changes AND main's production gating infrastructure.

## Step-by-step procedure

### 1. Start the merge (do this ONCE)

```bash
cd "C:\Users\scott\Dev\tab-agent-pro"
git fetch origin
git merge origin/main
```

This will stop with conflicts in ~31 files. Do NOT abort unless something goes catastrophically wrong.

### 2. List the conflicted files

```bash
git diff --name-only --diff-filter=U
```

Work through each one. For each file:

1. `git show origin/main:path/to/file` — see main's version
2. `git show HEAD:path/to/file` — see branch's version
3. Open the file with conflict markers `<<<<<<<`, `=======`, `>>>>>>>`
4. Decide the merged content using the per-file guidance below
5. Remove all conflict markers
6. `git add path/to/file`

### 3. Per-file resolution guidance

#### Highest-priority core files (do these first)

**`agents.py`**
- Branch: removed YourMT3+/HF Hub entirely; Basic Pitch sole backend; removed `_convert_to_noteseq`, `_parse_mt3_tokens`, `_build_yourmt3_args`, `prefer_yourmt3` param, `model_id` param.
- Main: added production hardening, health checks, metrics, `REQUIRED_FOR_TRANSCRIPTION`, `DEGRADABLE_COMPONENTS`, `OPTIONAL_COMPONENTS`.
- **Resolution:** Start from the branch version (no YourMT3). Re-add main's health/metrics component registration and production hardening that does NOT reference YourMT3. Keep `BASIC_PITCH_AVAILABLE`, `DEMUCS_API_AVAILABLE`, `health.set_component(...)`, `health.set_status("up")`. Remove any `yourmt3` references. Keep branch's `SplitterAgent`, `EarAgent` (device-aware), `TabAgent`.

**`app.py`**
- Branch: removed `spaces`/`SPACES_ZERO_GPU`/`GPU_AVAILABLE`/`@spaces.GPU`; added `device.py` import, `DEVICE_BACKENDS`, `_active_device_config()`, `_save_device_preference()`, device dropdown + backend chip UI, inline tablature HTML (`render_tab_text` import, `tab_output` gr.HTML).
- Main: added production UI hardening, `/health`/`/ready`/`/health/metrics` endpoints, queue bounds, error handling, signal handling.
- **Resolution:** Start from branch version. Re-add main's FastAPI health endpoints (`/health`, `/ready`, `/health/metrics`), uvicorn serve, queue config (`demo.queue(...)`), bounded concurrency. Keep branch's device selector UI + backend chip + inline tab. Ensure `ui_entry = process_audio` (no spaces.GPU).

**`main.py`**
- Branch: added `_progress()` helper + stage log lines; `EarAgent(device=args.device)`; `SplitterAgent(... device=args.device)`.
- Main: added production CLI hardening, threshold validation, profile application, output dir logic, session logging to memory.
- **Resolution:** Start from branch version. Re-add main's `resolve_thresholds`, `validate_thresholds`, `load_user_memory`/`apply_profile`, session logging block at end of `main()`. Keep `_progress()` calls. Keep `--device` choices expanded (auto/cpu/cuda/rocm/vulkan/mps/directml).

**`Dockerfile`**
- Branch: removed `HF_HOME`; added `ARG TORCH_INDEX_URL` + CPU default + GPU build instructions; `onnxruntime` base.
- Main: added `HEALTHCHECK`, non-root `tabagent` user, read-only code, `EXPOSE 7860`, `CMD ["python", "app.py"]`, CPU-only torch pin `--index-url https://download.pytorch.org/whl/cpu`.
- **Resolution:** Combine: keep branch's `TORCH_INDEX_URL` ARG + GPU build comments; keep main's `HEALTHCHECK`, non-root user, read-only code, `EXPOSE`, `CMD`, healthcheck `RUN python -c "import agents..."`. Keep `HF_HOME` removed. Use CPU index as default, override via `TORCH_INDEX_URL`.

**`requirements.txt` / `pyproject.toml`**
- Branch: removed `spaces`, `huggingface_hub[cli]`; kept everything else.
- Main: may have version bumps, new prod deps (bandit, etc. in dev).
- **Resolution:** Start from branch (no spaces/huggingface_hub). Apply main's version bumps and any new deps that don't require HF. Keep `torch>=2.1`, `torchaudio>=2.1`, `onnxruntime>=1.17.0`, `basic-pitch>=0.4.0,<0.5`, `demucs>=4.0.0`, `gradio>=4.44,<7`, `fastapi`, `uvicorn`, `setuptools<81`.

**`requirements.lock`**
- **Regenerate after deps settle:** `python tools/check_lockfile.py --write`. Branch removed `huggingface-hub`/`spaces` entries. Main added many pinned prod deps. Do NOT hand-edit; regenerate from a clean install.

**`index.xml`**
- Branch: removed "YourMT3+" from changelog text.
- Main: may have version/time updates.
- **Resolution:** Keep branch's "Basic Pitch transcription" changelog text; apply main's version/time if different.

**`monitoring.py`**
- Branch: removed `yourmt3` from `OPTIONAL_COMPONENTS` and `_component_status`.
- Main: added full health/metrics infrastructure (`STATUS_*`, `REQUIRED_FOR_TRANSCRIPTION`, `DEGRADABLE_COMPONENTS`, `OPTIONAL_COMPONENTS`, component status dict, `set_component`, `set_status`, `is_ready`, `effective_status`, `component_health`, `as_dict`, `record_request`, `default_metrics`).
- **Resolution:** Start from main's full monitoring infrastructure (it's the production version). Remove any `yourmt3` references. Keep `basic_pitch`, `demucs`, `suno_detector` components.

**`reaper/TabAgent.lua`**
- **Keep branch version** (io.popen + reaper.defer streaming). Main's version uses blocking `os.execute`. Do NOT revert. Only `reaper.*` official API + `io.popen`.

**`tools/check_reaper_settings.py`**
- **Keep branch version** (io.popen stub + `reaper.defer` no-op + `_lua_str` path escaping). Main's version uses `os.execute` stub.

**`tools/deploy_space.sh`**
- **Deleted on branch** (HF-specific, intentionally removed). If main modified it, accept deletion.

**`.env.example`**
- Branch: removed `HF_TOKEN`, `SPACES_ZERO_GPU`, `YOURMT3_REVISION*`; added `DEVICE` doc.
- Main: may have production env vars.
- **Resolution:** Keep branch's `DEVICE` doc; apply main's additional production env vars (MAX_CONCURRENT_REQUESTS, QUEUE_MAX_SIZE, UPLOAD_TTL_SEC, HOST, PORT, CACHE_DIR). No HF_TOKEN/SPACES_ZERO_GPU/YourMT3.

**Docs** (`README.md`, `docs/RUNBOOK.md`, `docs/SECURITY.md`, `docs/ACCEPTED_RISKS.md`, `docs/RELEASE_NOTES_1.1.0.md`, `index.xml`, `THIRD_PARTY_NOTICES.md`, `AUDIT.md`, `CHANGELOG.md`):
- Branch: scrubbed HF references everywhere.
- Main: added lots of production docs (RUNBOOK, SECURITY, ACCEPTED_RISKS, THIRD_PARTY_NOTICES, CHANGELOG).
- **Resolution:** Keep branch's HF-scrubbed versions. Re-add main's production doc content (RUNBOOK procedures, SECURITY policy, ACCEPTED_RISKS table, THIRD_PARTY_NOTICES packages, CHANGELOG entries) but remove any HF_TOKEN/ZeroGPU/YourMT3/deploy_space references from them.

**`.github/workflows/ci.yml`**
- Main added CI. Branch may have CI tweaks. **Keep main's CI** (production readiness). Ensure it doesn't reference HF/deploy_space.

**`.pip-audit-ignore`**
- Main added 111 lines. **Keep main's version** (prod vuln acceptances). Ensure no HF-specific entries.

**`tests/`** (`test_app.py`, `test_app_integration.py`, `test_app_wiring.py`, `test_hardening.py`):
- Branch: updated for removed APIs (no `prefer_yourmt3`, no `_convert_to_noteseq`, `hf_ratio`→`hi_freq_ratio`, 3-value return from `_process_audio_impl`).
- Main: added `test_hardening.py` (363 lines), updated wiring/hardening tests.
- **Resolution:** Keep branch's API-updated tests. Add main's `test_hardening.py` and any new test coverage, but adapt to branch's API (3-value returns, no YourMT3, device param only).

### 4. After resolving all conflicts

```bash
# Verify no conflict markers remain
grep -rnE "^(<<<<<<<|=======|>>>>>>>)" . --include="*.py" --include="*.lua" --include="*.md" --include="*.txt" --include="*.toml" --include="*.yml" --include="*.yaml" --include="*.sh" --include="*.json"
# must return nothing

# Python compiles
python -m py_compile agents.py app.py main.py device.py monitoring.py validate.py tools/check_reaper_settings.py

# Lua syntax
luac -p reaper/TabAgent.lua   # if luac available

# Lockfile static gate
python tools/check_lockfile.py --static

# REAPER settings check
PYTHONIOENCODING=utf-8 python tools/check_reaper_settings.py

# Test suite (expect ~265+ passing; only pre-existing Windows cp1252 read_text() failures)
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q -p no:cacheprovider

# No HF remnants in active source
grep -rnEi "huggingface|spaces\.GPU|SPACES_ZERO_GPU|HF_TOKEN|yourmt3|YourMT3|deploy_space" --include="*.py" --include="*.lua" --include="*.md" --include="*.txt" --include="*.toml" --include="*.yml" .
# Only historical docs (CHANGELOG.md, AUDIT.md) should remain
```

### 5. Commit & push

```bash
git add -A
git commit -m "merge: integrate origin/main production readiness into tab-agent-pro refactor"
git push origin arena/ebfd10d9-tab-agent-pro
```

## Windows-specific gotchas

- `python` → 3.11.16, `python3` → 3.14.8, `pip` → 3.11. Be explicit.
- MSYS path conversion is OFF: use `C:/Users/x`-style native paths for Windows programs.
- `print()` with `→`/emoji fails under cp1252; use `PYTHONIOENCODING=utf-8`.
- `read_text(app.py)` on cp1252 chokes on emoji — pre-existing test failures in `test_app_wiring.py`, `test_hardening.py`, `test_tools.py`. Do NOT "fix" by removing emoji; unrelated to merge.

## Hard rules

1. **Do NOT restore YourMT3+/HF Hub** in `agents.py` — Basic Pitch is the sole backend.
2. **Do NOT restore `tools/deploy_space.sh`** — HF-specific, intentionally removed.
3. **Do NOT restore `spaces`/`SPACES_ZERO_GPU`/`@spaces.GPU`/`HF_TOKEN`** — no cloud GPU.
4. **Do NOT revert `reaper/TabAgent.lua` to `os.execute`** — keep `io.popen` + `reaper.defer` streaming.
5. **Do NOT delete `device.py`** — new on branch, required for local GPU.
6. **Keep `setuptools<81`** — required by pretty_midi (load-bearing pin).
7. Remove ALL conflict markers before committing.

## The merge-base commits to reference

- Base: `4061696bc2438002dbb3b5f1689385d87665e8b1`
- Branch HEAD: `019c174` (prompt) / `29d8814` (refactor)
- Main HEAD: `82077bb` (production readiness 1.1.0 #9)

Use `git show <sha> -- <file>` to read either side's version of any file during resolution.

Good luck. This is a substantial, file-by-file merge. Work methodically.