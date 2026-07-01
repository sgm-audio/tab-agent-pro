---
title: Tab Agent - AI Tablature Transcription
emoji: 🎸
colorFrom: blue
colorTo: purple
sdk: docker
app_file: Dockerfile
license: mit
---

[![CI](https://github.com/scottmills306/tab-agent-pro/actions/workflows/ci.yml/badge.svg)](https://github.com/scottmills306/tab-agent-pro/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# 🎸 Tab Agent

Upload audio (guitar/bass). Get tablature, MIDI, and JSON back.

## Quick Start

```bash
git clone https://github.com/scottmills306/tab-agent-pro.git
cd tab-agent-pro
pip install -r requirements.txt
python main.py input/your_song.wav
```

Or with Docker:

```bash
docker build -t tab-agent .
docker run -v $(pwd)/input:/app/input tab-agent python main.py input/your_song.wav
```

Or use the web UI on [HuggingFace Spaces](https://scottymills-tab-agent-pro.hf.space).

## What It Does

1. **Quality Analysis** — Detects AI audio artifacts (Suno/Udio), adjusts thresholds
2. **Stem Separation** — Demucs isolates guitar/bass from the mix
3. **Spatial Processing** — Mid-side splits lead and rhythm guitars
4. **Transcription** — YourMT3+ (if available) or Basic Pitch → MIDI notes
5. **Tablature** — Dynamic programming assigns notes to strings/frets
6. **Technique Detection** — Slides, hammer-ons, pull-offs
7. **Export** — MIDI, ASCII tab, JSON

## Features

- Two transcription engines: YourMT3+ (primary) → Basic Pitch (fallback)
- Demucs stem separation
- Suno/Udio artifact detection and cleanup
- Multi-track: lead guitar, rhythm L/R, bass
- Technique detection: slides, hammer-ons, pull-offs
- MIDI / ASCII tab / JSON export
- ReaPack scripts for Reaper DAW integration
- HF Spaces Zero GPU support (if enabled on your Space)

## Reaper Integration

1. Extensions → ReaPack → Import repositories
2. Add: `https://raw.githubusercontent.com/scottmills306/tab-agent-pro/main/index.xml`
3. Install "Tab Agent" and "Tab Agent Settings"
4. Select audio → Run script

## Project Structure

```
├── agents.py              # Splitter, Ear, Tab agents
├── app.py                 # Gradio web UI
├── main.py                # CLI pipeline
├── suno_postprocessor.py  # AI audio artifact detection
├── init_memory.py         # Preset profiles (tunings, thresholds)
├── monitoring.py          # Structured logging + health checks
├── index.xml              # ReaPack package index
├── reaper/                # ReaPack Lua scripts
├── tests/                 # Unit tests (60+ passing)
├── examples/              # Demo audio files
├── input/                 # Place audio files here
└── output/                # Generated tablature lands here
```

## License

MIT
