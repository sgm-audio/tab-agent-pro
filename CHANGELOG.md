# Changelog

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
  applied (and persisted to user memory), and threshold precedence is
  CLI flag → profile → built-in default.
- **REAPER settings were inert.** `Settings.lua` persisted profile, instrument,
  thresholds and export toggles, but `TabAgent.lua` read only `install_path`.
  The values are now passed to `main.py` as CLI flags (verified by
  `tools/check_reaper_settings.py` in CI).

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
- **README/CHANGELOG** now match the code: 245 tests, `0.0.0.0`/`PORT`
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
