# Changelog

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
- **Gradio web UI**: HuggingFace Spaces-ready with Zero GPU acceleration
- **CLI pipeline**: Batch processing via `python main.py`
- **8 preset profiles**: Standard, drop D, classical, bass, Suno-optimized, live band

### Infrastructure
- GitHub Actions CI: lint, format, type-check, test, security scan
- Pre-commit hooks: ruff lint + format, mypy, trailing whitespace, end-of-file
- Dockerfile with multi-stage build for Space deployment
- Python 3.10+ required
