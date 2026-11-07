# Changelog

## 1.1.0 (2026-11-07)

Production-readiness pass: the seven-phase audit chain (repo audit → claim
verification → build validation → dynamic testing → hardening → docs → sign-off).
Artifacts live in `artifacts/`. No API changes; no breaking changes to the CLI,
the web UI or the REAPER scripts.

### Added — observability and honest health
- **`/ready` readiness endpoint.** Returns `503` with the blocking component
  until a required backend is loadable; `/health` stays `200` as a liveness
  probe. The container `HEALTHCHECK` now probes `/ready`, so a broken image never
  receives traffic. Previously `/health` reported `up` with every backend dead.
- **Component health derived, not hardcoded.** `HealthTracker` computes status
  from recorded operations and load results, so `down`/`degraded` are facts
  rather than literals.
- **Bounded metrics.** `PipelineMetrics` keeps a fixed window per stage; the old
  unbounded list grew without limit on a long-lived server.
- **Error references.** A failed web request returns a short `error_ref`; the
  full exception stays in the logs, matched by that reference.

### Fixed — sign-off finding: readiness failed a serving instance
During the staging-equivalent live run, `/ready` returned `503` **while** the
instance completed a full transcription: Demucs has no weights in that
environment, so its passthrough fallback ran, and demucs was treated as
readiness-critical. Readiness now gates on transcription only; a quality
fallback shows up as `status: degraded` plus a new `degraded_components` field
in `/health`, and the instance stays in rotation. Regression tests assert both
directions (no transcription backend → `503`, no Demucs → `200` + degraded).

### Fixed — silent failure modes
Ten sites that swallowed errors now log, record health, and either recover or
escalate. Worst offenders: `.env` parsing, saved preferences (corrupt JSON),
YourMT3+ clone timeout/failure, malformed token parsing, and "Cannot read audio"
with an empty reason. A silent `except Exception: return None` around the
checkpoint clone had been hiding every download failure.

### Fixed — concurrency and resource leaks
- **Per-request session directories** (`mkdtemp`) — concurrent requests could
  previously collide in one shared output directory and return each other's
  files.
- **Bounded queue** (`MAX_CONCURRENT_REQUESTS`, `QUEUE_MAX_SIZE`) — the task
  queue is no longer unbounded.
- **Pruning** — stale sessions (1 h) and stale uploads (`UPLOAD_TTL_SEC`, 900 s)
  are removed; upload temp files previously accumulated forever.
- Verified no memory leak: 0.1 MB/run steady state, 0.03 MB retained by
  `tracemalloc` over 20 runs (an initial +4.7 MB/run reading was cold-start
  allocator growth, disproved by warm-up-controlled re-measurement).

### Added — supply chain and CI
- **`requirements.lock`** — 160 exact pins for what actually ships, with the
  two-stage Basic Pitch install documented in the header.
  `tools/check_lockfile.py` verifies the installed environment against it
  (`--write` regenerates, `--static` checks declared deps without installing).
- **Vulnerability gate** — `tools/check_vulns.py` audits the *installed* set
  (pip-audit cannot resolve `requirements.txt`/lock here without pulling the
  unused TensorFlow tree) and fails on any advisory that is not accepted in
  `.pip-audit-ignore` with a reason **and an unexpired date**. Negative-tested:
  an expired entry and a removed acceptance both fail the gate.
- **YourMT3+ checkpoint pinning** — `YOURMT3_REVISION` selects a commit sha;
  `YOURMT3_REQUIRE_PINNED_REVISION=1` makes an unpinned download a hard error
  (fail closed). Unpinned runs log a warning.
- **CI rewritten** as a gate pipeline: ruff/black → repo guards → gitleaks +
  bandit → CPU-torch install → lockfile verify → CVE gate → mypy → pytest with
  coverage → failure simulations → backend smoke → `validate.py`.

### Changed — container
- `Dockerfile` rewritten: non-root `tabagent` user, application code read-only
  at runtime with only the cache and output directories writable, CPU-only torch
  index, exec-form `CMD`, `/ready` healthcheck, and the lockfile check enforced
  during the build.

### Changed — docs
- **README corrections.** The example tablature is now the real output of
  `python main.py examples/guitar_solo.wav` (the previous sample was
  hand-written and did not match any code path). The 50 MB / 5 min limits are
  scoped to the web UI — the CLI has none. The GPU claim is conditional:
  acceleration is advertised only on a ZeroGPU Space that reports
  `SPACES_ZERO_GPU=1`, or with `DEVICE=cuda`; the default image is CPU-only.
  Test counts, project structure, env vars and the deploy file list are current.
- **`docs/RUNBOOK.md`** — deploy, health semantics, alert thresholds, log
  queries, rollback (with a rehearsal procedure), dependency refresh, capacity
  notes, known failure modes, release checklist.
- **`docs/ACCEPTED_RISKS.md`** — every accepted advisory with exposure analysis
  and expiry, plus the non-CVE design risks (no auth, no rate limiting, LGPL
  `soxr`).
- **`SECURITY.md`** — threat model, scope, disclosure process, built-in controls.
- **`THIRD_PARTY_NOTICES.md`** — license table verified from installed metadata;
  the two reciprocal licenses (`soxr` LGPL-2.1-or-later, `tqdm` MPL-2.0) with
  the reasoning that keeps MIT distribution valid. No GPL/AGPL anywhere.
- **`.env.example`** now documents `YOURMT3_REVISION`,
  `YOURMT3_REQUIRE_PINNED_REVISION` and `SPACES_ZERO_GPU` (11 variables, all
  verified as read by the code).

### Verified
- 290 tests pass; 91% statement coverage (up from 259 tests).
- `ruff`, `black --check`, `mypy` clean; `bandit` 0 issues across the tree.
- 15/15 failure-mode simulations behave as documented, including a live
  network-timeout fallback from YourMT3+ to Basic Pitch.
- Performance baseline recorded: 604 MB RSS after imports, tablature DP 5–7 µs
  per note (5 000 notes ≈ 31 ms), Basic Pitch 0.82 s for a 2 s clip.

### Not verified (honestly)
- No Docker daemon in the audit environment: the image build, the staging
  deploy and the rollback drill are validated by static guards and by running
  every command outside the container, not by a real build. See
  `artifacts/phase7_signoff.json` → `unverifiable`.
- The YourMT3+ happy path cannot run here (gated multi-GB checkpoint, no Hub
  egress); it is instrumented and fails closed instead.

## 1.0.1 (2026-10-04)

Fixes from the documentation/reality audit (`AUDIT.md`). No API changes.

### Fixed — entry points
- **Gradio UI could not transcribe anything.** The button was wired to
  `_process_audio_impl` with 5 inputs while the function required a 6th
  (`progress`); every click failed with `'NoneType' object is not callable`.
  Both entry points now default `progress=gr.Progress()`.
- **Zero GPU claim was inaccurate.** `spaces.GPU` decorated an unused function
  and applied unconditionally, including on CPU Spaces where it is a
  pass-through. GPU acceleration is now reported only when the runtime really
  provides a Zero GPU slice (`SPACES_ZERO_GPU`).
- **Docker/Spaces could not reach the app.** The server defaulted to
  `127.0.0.1:7860` and ignored `PORT`; it now binds `HOST` (default `0.0.0.0`)
  and reads `PORT` (default `7860`), with `ENV HOST/PORT` set in the image.
- **`--profile` and `--instrument` were parsed but ignored.** Profiles are now
  applied (and persisted to user memory), `--instrument` is accepted
  case-insensitively (`Guitar`, `Bass`, `All`) and filters the CLI pipeline, and
  threshold precedence is CLI flag → profile → Suno-adjusted/built-in default.
- **REAPER settings were inert.** `Settings.lua` persisted profile, instrument,
  thresholds and export toggles, but `TabAgent.lua` read only `install_path`.
  The values are now passed to `main.py` as CLI flags (verified by
  `tools/check_reaper_settings.py` in CI), and disabling MIDI export
  (`--no-midi`) no longer triggers a false `"No Output"` error dialog.
- **Suno artifact detection & preprocessing fixes.** `SunoAudioPreprocessor._reduce_hf_artifacts`
  no longer becomes a no-op when Nyquist is below 16 kHz, `SunoArtifactDetector`
  no longer false-positives on clean harmonic guitar audio (`examples/guitar_solo.wav`),
  and `SunoNotePostprocessor._smooth_timing` preserves positive note duration after quantization.
- **Web UI status & signal handling.** `app.py` no longer double-counts `.tab`/`.json`
  files in `Files Generated`, formats the enabled export list cleanly when `Tab` is
  unchecked, loads saved user preferences from `user_preferences.json`, and registers
  shutdown signal handlers inside `serve()` instead of at module import time.
- **Space deployment script (`tools/deploy_space.sh`).** Fixed `hf upload` invocation
  to pass `--revision "$REVISION"` and per-file `--include` arguments without
  excluding `README.md` (which holds the Space YAML frontmatter).

### Fixed — transcription and export
- **One unplayable note discarded the whole track.** `TabAgent.generate_tab`
  now skips notes outside the instrument's range, reports them in
  `last_skipped_notes`, and keeps the remaining notes; the UI/CLI say which
  tracks had no playable positions instead of silently producing nothing.
- **Tablature rendering.** Adjacent multi-character cells no longer run
  together (`1215` → `12 15`), notes sharing a string in one 50 ms column are
  no longer silently overwritten (rendered as `15/8p`), the string grid keeps
  the instrument's full width via `num_strings`, and `num_strings` is recorded
  in the JSON export.
- **Docker install line.** `pip install --no-deps basic-pitch>=0.4.0` was
  parsed by the shell as a redirection and never installed the package. The pin
  is quoted, and the real runtime imports of `basic_pitch.note_creation`
  (`mir_eval`, `resampy`) are installed explicitly so the documented fallback
  actually exists inside the image. `tools/check_docker_install.py` guards this.
- **YourMT3+ loading.** `resume_download` was removed (deprecated in
  huggingface_hub), downloads are filtered to the checkpoint that is actually
  loaded instead of the full multi-GB repo, `HF_TOKEN` is honoured, and the
  hardcoded experiment id is now a single constant.

### Fixed — tooling and docs
- **ReaPack `index.xml` was malformed** for the format: wrong root `version`,
  no `commit`, no `name`, `<description>` inside `<category>`, `type` on
  `<version>`, and inverted `<source>` elements pointing at repo paths. It is
  now generated-format compatible and validated by `tools/check_index.py`.
- **`validate.py` was unusable** (exit 1, silent output, bogus path checks).
  It now prints a PASS/FAIL/SKIP summary, exits non-zero only on real failures,
  and checks the fixed behaviours (unplayable notes, tab separators, profile
  thresholds, transcription backend availability).
- **Packaging:** the wheel contained no modules (`packages.find` matched
  nothing). `py-modules` is declared and `pip install .` now installs the app.
- **Dependencies:** `setuptools<81` (required by pretty-midi) and the
  `mir_eval`/`resampy` pair are declared in `pyproject.toml` as well as
  `requirements.txt`; `demucs` and `spaces` are explicitly declared; the unused
  `transformers` dependency is documented as intentionally absent.
- **`.env.example`** documented `HOST`, `PORT`, `CACHE_DIR`, `DEVICE` and
  `HF_TOKEN` that nothing read. `app.py` loads `.env` (no new dependency),
  `CACHE_DIR`/`DEVICE`/`HF_TOKEN` are honoured, and
  `tools/check_env_example.py` fails CI if a documented variable is unused.
- **README/CHANGELOG** now match the code: 259 tests, `0.0.0.0`/`PORT`
  behaviour, the `other` Demucs stem used for guitar, the Basic-Pitch-first
  transcription path, the export limits, and ReaPack setup that includes the
  glue-item step.

## 1.0.0 (2026-07-01)

Initial stable release of Tab Agent Pro — AI-powered guitar/bass tablature transcription.

### Features
- **Basic Pitch transcription** with ONNX backend (no TensorFlow required)
- **Demucs stem separation**: Isolates guitar/bass from full mixes using htdemucs
- **Spatial processing**: Mid-side technique separates lead and rhythm guitars
- **Suno/Udio artifact detection**: Automatic quality analysis and cleanup for AI-generated audio
- **Dynamic programming tablature**: Viterbi-style optimal fingering with technique detection (slides, hammer-ons, pull-offs)
- **Multi-format export**: MIDI, column-aligned ASCII tablature, JSON
- **ReaPack integration**: REAPER DAW scripts for in-editor transcription
- **Gradio web UI**: Hugging Face Spaces-ready (CPU by default, Zero GPU when deployed on a ZeroGPU Space)
- **CLI pipeline**: Batch processing via `python main.py`
- **8 preset profiles**: Standard, drop D, classical, bass, Suno-optimized, live band

### Infrastructure
- GitHub Actions CI: lint, format, type-check, test, security scan, Docker-config guards
- Pre-commit hooks: ruff lint + format, black, mypy, trailing whitespace, end-of-file
- Dockerfile with multi-stage build for Space deployment
- Python 3.10+ required
