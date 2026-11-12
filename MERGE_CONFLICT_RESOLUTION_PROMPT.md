# System Prompt — Merge Conflict Resolution for tab-agent-pro

## Mission

Resolve all merge conflicts between branch `arena/ebfd10d9-tab-agent-pro` (local, already pushed to origin) and `origin/main`, producing a clean merge commit. The branch diverged from main at merge-base `4061696bc2438002dbb3b5f1689385d87665e8b1`. EVERY file changed on main was also changed on the branch (38 overlapping files), so expect widespread conflicts.

## Branch vs. Main — what changed on each side

### Branch `arena/ebfd10d9-tab-agent-pro` (the work to preserve)
The branch is a large refactor. Key themes:
1. **Removed Hugging Face entirely** — YourMT3+/HF Hub checkpoint acquisition, `spaces.GPU`/ZeroGPU, `HF_TOKEN`, `tools/deploy_space.sh`. Basic Pitch is now the sole transcription backend. Deleted `tools/deploy_space.sh`.
2. **Added local GPU compute** — New file `device.py` resolves `DEVICE` to `auto/cuda/rocm/vulkan/mps/directml/cpu`. Demucs uses torch CUDA (ROCm via HIP); Basic Pitch uses the matching ONNX Execution Provider. `Dockerfile` gained `TORCH_INDEX_URL` build arg for CUDA/ROCm image builds.
3. **UX** — Backend chip + device selector in the web UI (`app.py`); inline lead-guitar tablature rendered directly in results (`main.py` `render_tab_text` + inline HTML tab preview).
4. **REAPER console streaming** — `reaper/TabAgent.lua` streams `main.py` stdout/stderr live via `io.popen` + `reaper.defer` instead of blocking on `os.execute`. Uses only official `reaper.*` API calls. `tools/check_reaper_settings.py` updated for the `io.popen`-based runner.
5. **Docs** — `README.md`, `docs/RUNBOOK.md`, `docs/SECURITY.md`, `docs/ACCEPTED_RISKS.md`, `docs/RELEASE_NOTES_1.1.0.md`, `index.xml`, `THIRD_PARTY_NOTICES.md`, `SECURITY.md`, `AUDIT.md` all scrubbed of Hugging Face references.
6. **Tests** — updated for removed APIs (YourMT3, `_convert_to_noteseq`, `hf_ratio`→`hi_freq_ratio`, `prefer_yourmt3` param removed from `EarAgent`), new device tests.

### Main `origin/main` (the target to merge into)
Main advanced from the same merge-base with its own changes to the same files. You must diff each conflicting file to see main's side. Do NOT assume main's changes are wrong — integrate both.

## How to start

Working directory: `C:\Users\scott\Dev\tab-agent-pro`

```bash
cd "C:\Users\scott\Dev\tab-agent-pro"
git status                # confirm clean tree
git branch --show-current # arena/ebfd10d9-tab-agent-pro
git log --oneline -3      # HEAD = 29d8814
git merge-base HEAD origin/main  # 4061696
```

To bring in main and surface conflicts:

```bash
git fetch origin
git merge origin/main     # this will produce conflicts in ~38 files
```

or create a merge branch first (safer):

```bash
git checkout -b merge-tab-agent-pro-to-main
git merge origin/main
```

## Conflict resolution strategy — which side wins

| Area | Guidance |
|------|----------|
| `agents.py` | Branch removed YourMT3+ entirely; main may still reference it. **Keep the branch version** (Basic Pitch only) but re-add any main-side bug fixes/tests that don't depend on YourMT3. |
| `app.py` | Branch added device selector, backend chip, inline tab HTML. Main may have UI tweaks. **Keep branch changes**, integrate main's non-conflicting UI edits. |
| `main.py` | Branch added `_progress()` and stage logs. Main may have pipeline edits. **Keep branch changes**, integrate main's pipeline fixes. |
| `device.py` | **New file on branch only** — keep it, ensure it's staged. |
| `Dockerfile` | Branch added `TORCH_INDEX_URL` arg + removed `HF_HOME`. Main may have base-image/dependency changes. **Merge carefully** — keep both the TORCH_INDEX_URL arg and main's image changes. |
| `requirements.txt` / `pyproject.toml` | Branch removed `spaces`/`huggingface_hub`, added nothing new. Main may have dep bumps. **Keep branch removals**, apply main's version bumps on top. |
| `requirements.lock` | Regenerate if needed via `python tools/check_lockfile.py --write` after deps settle. Branch removed `huggingface-hub`/`spaces` entries. |
| `reaper/TabAgent.lua` | **Keep branch version** (io.popen streaming). Main's version uses os.execute. Do not revert to os.execute. |
| `tools/check_reaper_settings.py` | **Keep branch version** (io.popen stub). |
| `tools/deploy_space.sh` | **Deleted on branch**. If main modified it, accept deletion (it's HF-specific, being removed intentionally). |
| Docs (`README`, `RUNBOOK`, `SECURITY`, `ACCEPTED_RISKS`, `RELEASE_NOTES`, `index.xml`, `THIRD_PARTY_NOTICES`) | **Keep branch versions** (HF scrubbed). Integrate main's non-HF doc changes (e.g., new sections, test counts). |
| `.env.example` | Branch removed `HF_TOKEN`, `SPACES_ZERO_GPU`, `YOURMT3_REVISION*`; added `DEVICE` doc. **Keep branch version**. |
| Tests | Branch updated for removed APIs. **Keep branch test versions**; if main added new tests, port them to the branch's API. |
| `monitoring.py` | Branch removed `yourmt3` optional component. **Keep branch version**. |
| `validate.py` | Branch removed `hf_ratio`/`prefer_yourmt3` references. **Keep branch version**. |

**General rule:** The branch is the "direction of travel" (HF removal + local GPU). When in doubt, keep the branch version and cherry-pick main's non-conflicting, non-HF-dependent fixes on top.

## Verification steps (must pass before claiming done)

1. **Clean merge:** `git status` shows no unmerged paths; `git diff --cached --name-only` lists resolved files.
2. **Python compiles:** `python -m py_compile agents.py app.py main.py device.py monitoring.py validate.py tools/check_reaper_settings.py` — all OK.
3. **Lua syntax:** `luac -p reaper/TabAgent.lua` — OK (if luac available).
4. **Lockfile static gate:** `python tools/check_lockfile.py --static` — must pass (all declared deps locked).
5. **REAPER settings check:** `PYTHONIOENCODING=utf-8 python tools/check_reaper_settings.py` — all checks pass.
6. **Test suite:** `PYTHONIOENCODING=utf-8 python -m pytest tests/ -q -p no:cacheprovider` — expect ~265+ passing; only pre-existing Windows cp1252 `read_text()` failures (unrelated to merge) should remain. No NEW failures.
7. **No HF remnants in active source:** `grep -rnE "huggingface|spaces\.GPU|SPACES_ZERO_GPU|HF_TOKEN|yourmt3|YourMT3|deploy_space" --include="*.py" --include="*.lua" --include="*.md" --include="*.txt" --include="*.toml" .` should return only historical docs (CHANGELOG.md, AUDIT.md) — not active code.

## Windows-specific gotchas

- `python` → 3.11.16, `python3` → 3.14.8, `pip` → 3.11. Be explicit per project (Dockerfile/CI use python3).
- MSYS path conversion is OFF: pass `C:/Users/x`-style native paths to native Windows programs.
- `print()` with `→` or emoji can fail under cp1252; set `PYTHONIOENCODING=utf-8` when running Python tools that print such chars.
- `read_text(app.py)` on Windows cp1252 chokes on emoji (`⚠️`, `❌`) — this causes pre-existing test failures in `test_app_wiring.py`, `test_hardening.py`, `test_tools.py`. Do NOT try to "fix" by removing emoji; these are pre-existing and unrelated to the merge.

## Deliverable

A clean merge commit on `arena/ebfd10d9-tab-agent-pro` (or a new merge branch) that:
- Contains all branch changes (HF removal, device.py, GPU support, REAPER streaming, inline tab, device selector)
- Contains all non-conflicting main changes
- Has no conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) left in any file
- Passes the verification steps above

Then push: `git push origin arena/ebfd10d9-tab-agent-pro` (or the merge branch name).

## Useful commands

```bash
# See all conflicted files
git diff --name-only --diff-filter=U

# Abort merge if things go badly
git merge --abort

# Diff a specific file against both parents
git diff HEAD -- path/to/file      # branch vs working
git diff MERGE_HEAD -- path/to/file  # main vs working

# Take one side wholesale
git checkout --theirs path/to/file   # main's version
git checkout --ours  path/to/file    # branch's version
git add path/to/file
```

## Notes for the next agent

- Do NOT delete `device.py` — it's new on the branch and required.
- Do NOT restore `tools/deploy_space.sh` — it's HF-specific and intentionally removed.
- Do NOT restore YourMT3+/HF Hub code in `agents.py` — Basic Pitch is the sole backend now.
- `reaper/TabAgent.lua` uses only official REAPER API (`reaper.ShowConsoleMsg`, `reaper.defer`, `reaper.Undo_BeginBlock/EndBlock`, `reaper.InsertMedia`, `reaper.GetSelectedMediaItem`, `reaper.GetActiveTake`, `reaper.GetMediaItemTake_Source`, `reaper.GetMediaSourceFileName`, `reaper.GetExtState/SetExtState`, `reaper.file_exists`, `reaper.GetResourcePath`, `reaper.GetOS`, `reaper.GetNumTracks`, `reaper.GetTrack`, `reaper.GetSetMediaTrackInfo_String`, `reaper.MB`). No `os.execute`.
- The branch is already pushed to `origin/arena/ebfd10d9-tab-agent-pro`. After resolving locally, push the result.

Good luck. Work methodically file-by-file through the 38 overlapping files.