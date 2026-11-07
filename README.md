---
title: Tab Agent — AI Tablature Transcription
emoji: 🎸
colorFrom: blue
colorTo: purple
sdk: docker
app_file: Dockerfile
license: mit
---

[![CI](https://github.com/sgm-audio/tab-agent-pro/actions/workflows/ci.yml/badge.svg)](https://github.com/sgm-audio/tab-agent-pro/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

# 🎸 Tab Agent

**Upload guitar or bass audio. Get tablature, MIDI, and JSON back.**

Tab Agent transcribes guitar and bass recordings into playable tablature using
a multi-engine AI pipeline. Works on anything from a clean DI track to a full
mix with drums and vocals.

---

## Quick Start

### Web UI (recommended)

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

> **Linux note (which is most installs, and the image):** Basic Pitch's default
> Linux dependency set pulls TensorFlow `<2.15.1`, which has no Python 3.12+
> wheels and is unused by the ONNX backend. Install Basic Pitch without
> dependencies, then the rest. This is what `Dockerfile` and CI do, and it is
> the supported path — **Python 3.11+ is required**, and the image runs 3.12:
> ```bash
> pip install --no-deps "basic-pitch>=0.4.0,<0.5"
> grep -v '^basic-pitch' requirements.txt > /tmp/req-nobp.txt && pip install -r /tmp/req-nobp.txt
> pip install --no-deps -r requirements.lock   # exact, tested versions
> ```

**Reproducible install:** `requirements.lock` pins the 118 distributions that
actually ship — the runtime closure of the declared dependencies, including
transitive ones, and excluding the TensorFlow tree and NVIDIA CUDA wheels that
this project never installs. CI and the Docker build install from it and then
fail if the environment drifts from it:

```bash
pip install -r requirements.txt
python tools/check_lockfile.py          # verify
python tools/check_lockfile.py --write  # regenerate after a dependency change
```

The server binds to `HOST` (default `0.0.0.0` so containers and Hugging Face
Spaces can reach it) and listens on `PORT` (default `7860`). Both are read from
the environment or from an optional `.env` file — see `.env.example`.

The **web UI** limits uploads to 50 MB and 5 minutes and accepts
`.wav .mp3 .flac .ogg .m4a .aiff`; results are returned as a ZIP archive. The CLI
has no such limits — it processes whatever file you point it at, however long.

### CLI

```bash
python main.py input/your_song.wav
# Output: output/your_song_lead_guitar.tab
```

Useful flags:

```bash
python main.py song.wav --instrument Bass          # default: Guitar
python main.py song.wav --profile rock_drop_d      # preset profile (see init_memory.py --list)
python main.py song.wav --onset 0.4 --frame 0.3    # override profile thresholds
python main.py song.wav --no-midi --no-tab         # export JSON only
python main.py song.wav --output-dir /tmp/out
```

### Docker

```bash
docker build -t tab-agent .
docker run -p 7860:7860 -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output tab-agent
# or run the CLI inside the image:
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output tab-agent python main.py input/your_song.wav
```

The image binds `0.0.0.0:7860` (required by Hugging Face Spaces) and ships
Basic Pitch with its ONNX backend plus the stem-separation stack.

### Validation

```bash
python -m pytest tests/            # 291 tests, 91% coverage
python validate.py                 # end-to-end checklist (PASS/FAIL summary)

python tools/check_index.py           # ReaPack index format
python tools/check_env_example.py     # every documented env var is read
python tools/check_docker_install.py  # basic-pitch pin + required deps
python tools/check_reaper_settings.py # REAPER settings reach main.py
python tools/check_lockfile.py        # installed env == requirements.lock
python tools/check_vulns.py           # advisories: none unaccepted/expired
python tools/simulate_failures.py     # failure modes still behave (15 checks)
```

---

## Pipeline

```mermaid
graph LR
    A[Audio] --> B[AI Detection]
    B --> C[Stem Separation]
    C --> D[Transcription]
    D --> E[Tablature]
    E --> F[MIDI / Tab / JSON]
```

| Stage | Tool | What it does |
|-------|------|-------------|
| **Quality Analysis** | SunoDetector | Detects AI-generated audio artifacts, adjusts thresholds |
| **Stem Separation** | Demucs (htdemucs) | Splits the mix into vocals / drums / bass / other; `bass` is transcribed as bass and `other` as guitar |
| **Spatial Processing** | Mid-Side | Splits lead (centre) from rhythm (sides) by subtraction |
| **Transcription** | YourMT3+ → Basic Pitch | Converts audio → MIDI notes (Basic Pitch is the always-available fallback) |
| **Tablature** | DP Viterbi | Assigns notes to strings/frets optimally; unplayable notes are skipped |
| **Technique Detection** | Heuristic | Slides, hammer-ons, pull-offs |
| **Export** | — | MIDI, ASCII tab, JSON |

---

## Features

- **Basic Pitch transcription** — Spotify's lightweight ONNX model, no GPU required
- **Full-mix processing** — Demucs separates the mix; the residual `other` stem (guitars and other instruments) is transcribed as guitar
- **Multi-track output** — Lead guitar, rhythm L/R, bass in separate files
- **Column-aligned tablature** — Chords grouped vertically; a single unplayable note no longer discards the whole track
- **AI audio support** — Automatic detection and cleanup for Suno/Udio AI-generated audio
- **Technique detection** — Slides (`s`), hammer-ons (`h`), pull-offs (`p`) annotated in output
- **Plug-and-play presets** — 8 profiles (standard, drop D, classical, 4/5-string bass, Suno cleanup, live band) applied with `--profile`
- **REAPER integration** — Run transcription from the DAW via ReaPack, with the settings-script values passed through as CLI flags
- **CPU by default, GPU optional** — the transcription stack runs on CPU out of the box. Acceleration applies only in two narrow cases and is never claimed otherwise: on a Hugging Face **ZeroGPU** Space the web handler is wrapped in `spaces.GPU` *only* when the runtime reports `SPACES_ZERO_GPU=1`, and `DEVICE=cuda` moves torch work to a local GPU. Everything else is CPU, and the default Docker image installs CPU-only torch.

> **YourMT3+ status:** `agents.py` can load the YourMT3+ checkpoint from the
> Hugging Face Hub and falls back to Basic Pitch whenever it is unavailable. The
> Docker image ships Basic Pitch only; loading YourMT3+ additionally needs a
> multi-GB checkpoint download on first run (`HF_TOKEN` is honoured for gated
> repos, and `CACHE_DIR` controls where it is stored).

---

## Configuration

Environment variables (all optional; `.env` is read by `app.py`, `main.py`, and `run.sh`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `HOST` | `0.0.0.0` | Bind address for the web UI |
| `PORT` | `7860` | Port for the web UI (Spaces expects 7860) |
| `DEVICE` | `auto` | Torch device for transcription: `auto`, `cpu`, `cuda`, `mps` |
| `CACHE_DIR` | `~/.cache/tab_agent` | Model cache (YourMT3+ checkpoint and codebase) |
| `HF_TOKEN` | — | Hugging Face token for gated/private model repos |
| `MAX_CONCURRENT_REQUESTS` | `1` | Transcriptions allowed in flight at once |
| `QUEUE_MAX_SIZE` | `32` | Bounded backlog; requests beyond this are rejected |
| `UPLOAD_TTL_SEC` | `900` | Upload temp files older than this are pruned |
| `YOURMT3_REVISION` | — | Checkpoint commit sha. Empty = `main` **with a warning** |
| `YOURMT3_REQUIRE_PINNED_REVISION` | — | `1` makes an unpinned checkpoint a hard error (fail closed) |
| `SPACES_ZERO_GPU` | — | Set by Hugging Face on ZeroGPU Spaces; never set by hand |

Preset profiles configure tuning, thresholds and processing behaviour. Applying a
profile writes it to user memory so later runs (and the REAPER scripts) reuse it:

```bash
python init_memory.py --list                 # list the 8 profiles
python init_memory.py --profile bass_5_string
python main.py song.wav --profile suno_aggressive
```

Threshold precedence: explicit CLI flag → active profile → built-in default.

---

## Output Formats

| Format | Content | Use case |
|--------|---------|----------|
| **MIDI** (`.mid`) | Standard MIDI file with note events | Import into any DAW |
| **ASCII Tab** (`.tab`) | Human-readable fretboard notation | Printing, sharing |
| **JSON** (`.json`) | Structured note/fret data | Custom applications |

### Example tablature output

Produced by `python main.py examples/guitar_solo.wav` (verbatim, three of the
nine generated files omitted):

```
=== Lead Guitar Tablature ===

E|-  -  -      -  -  -      -  22 -
B|-  -  15/19h 22 24 -      20 -  -
G|-  -  -      -  -  -      -  -  -
D|-  -  -      -  21 24/17p 21 -  -
A|-  -  -      -  -  -      -  -  19
E|12 15 19     22 24 -      -  -  -

Legend: s=slide, h=hammer-on, p=pull-off
```

Each column is one onset group (notes within 50 ms); a note is never merged
into its neighbour, and a string that is unused in a column shows `-`.

---

## REAPER Integration

1. Extensions → ReaPack → Import repositories
2. Add: `https://raw.githubusercontent.com/sgm-audio/tab-agent-pro/main/index.xml`
3. Install **Tab Agent** and **Tab Agent Settings**
4. Glue the item you want analysed, select it, then run **Tab Agent**

`Tab Agent Settings` stores the install path, profile, instrument, onset/frame
thresholds, output directory and export toggles in REAPER's ExtState.
`Tab Agent` passes those values to `main.py` as CLI flags, so what you configure
is what runs.

---

## Project Structure

```
├── app.py                 # Gradio web UI + FastAPI health endpoints
├── agents.py              # Splitter, Ear, Tab agents
├── main.py                # CLI pipeline + tab/JSON export helpers
├── suno_postprocessor.py  # AI audio artifact detection
├── init_memory.py         # Preset profiles
├── monitoring.py          # Structured logging + health
├── validate.py            # End-to-end checklist
├── reaper/                # REAPER Lua scripts
├── tools/                 # Config/deployment guard scripts
├── tests/                 # 291 tests
├── examples/              # Demo audio
├── docs/RUNBOOK.md        # Deploy / monitor / rollback / dependency refresh
├── docs/ACCEPTED_RISKS.md # Every known risk, with exposure and expiry
├── SECURITY.md            # Threat model + disclosure process
├── requirements.lock      # 118 exact pins (runtime closure, CPU-only)
├── .pip-audit-ignore      # Accepted advisories; CI fails when one expires
├── pyproject.toml         # Project config
├── Dockerfile             # HF Spaces deployment (non-root, read-only code)
└── CHANGELOG.md           # Release history
```

---

## Security

- **Reporting:** see [`SECURITY.md`](SECURITY.md) — private advisories, not public issues.
- **Threat model:** single-tenant, unauthenticated by design; the hosting platform is the trust boundary.
- **Known risks:** [`docs/ACCEPTED_RISKS.md`](docs/ACCEPTED_RISKS.md) lists every accepted
  advisory with exposure analysis and an expiry date that turns CI red.
- **Gates:** `bandit` (SAST), `gitleaks` (secrets), `pip-audit` against the installed
  set (`tools/check_vulns.py`), and a lockfile check — all blocking in CI.
- **Dependencies:** `setuptools<81` is pinned deliberately (81+ removes
  `pkg_resources`, which `pretty_midi` needs) — do not "fix" it.
- **Model provenance:** set `YOURMT3_REVISION` to a reviewed sha and
  `YOURMT3_REQUIRE_PINNED_REVISION=1` in production.

---

## Deploying

### Hugging Face Spaces

The Space must be kept in sync with this repo. With a write token available:

```bash
export HF_TOKEN=hf_...
./tools/deploy_space.sh scottymills/tab-agent-pro
```

This uploads the exact files the Space needs (`Dockerfile`, `requirements.txt`,
`requirements.lock`, `agents.py`, `app.py`, `main.py`, `monitoring.py`,
`init_memory.py`, `suno_postprocessor.py`, `README.md`, `examples/`) — the file
list the Docker Space runtime expects. Health checks: `/health` (liveness +
component detail, including `degraded_components`), `/ready` (503 until a
transcription backend is loadable — used by the container healthcheck), and
`/health/metrics` (per-stage timings). Demucs loss degrades output quality but
does not fail readiness, because separation has a passthrough fallback.

---

## License

MIT — use it, modify it, share it.

---

*Built with [Basic Pitch](https://github.com/spotify/basic-pitch) by Spotify Research,
[Demucs](https://github.com/facebookresearch/demucs) by Meta,
and [Gradio](https://gradio.app).*
