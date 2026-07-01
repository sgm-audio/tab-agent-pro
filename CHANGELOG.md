# Changelog

## 1.0.0 (2026-07-01)

Initial stable release of Tab Agent Pro — AI-powered guitar/bass tablature transcription.

### Features
- **Multi-engine transcription**: YourMT3+ (primary) → Basic Pitch with ONNX backend (fallback)
- **Demucs stem separation**: Isolates guitar/bass from full mixes using htdemucs
- **Spatial processing**: Mid-side technique separates lead and rhythm guitars
- **Suno/Udio artifact detection**: Automatic quality analysis and cleanup for AI-generated audio
- **Dynamic programming tablature**: Viterbi-style optimal fingering with technique detection (slides, hammer-ons, pull-offs)
- **Multi-format export**: MIDI, column-aligned ASCII tablature, JSON
- **ReaPack integration**: REAPER DAW scripts for in-editor transcription
- **Gradio web UI**: HuggingFace Spaces-ready with Zero GPU acceleration
- **CLI pipeline**: Batch processing via `python main.py`

### Infrastructure
- GitHub Actions CI: ruff → black → mypy → pytest → bandit
- Pre-commit hooks: ruff lint + format, mypy, trailing whitespace, end-of-file
- PyTorch CPU + ONNX runtime for lightweight deployment
- Dockerfile with Python 3.10-slim for Space deployment

### Quality
- 194 passing tests, 84% branch coverage
- mypy strict: 0 type errors
- bandit: 0 HIGH/CRITICAL security issues
- ruff: 0 blocking lint errors
- Black-formatted codebase

### Notes
- Python 3.10+ required (3.12 recommended for fastest install)
- YourMT3+ model requires ~2.7GB download on first run (auto-detected)
- Basic Pitch uses ONNX backend by default (no TensorFlow required)
