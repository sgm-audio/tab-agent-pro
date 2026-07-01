# Tab Agent Pro - Complete Restoration Plan

## Project Status
- **Python**: 3.10 (officially supported by all packages)
- **Primary Model**: YourMT3+ (`mimbres/YourMT3-cpu` - CPU-optimized)
- **Fallback Model**: Basic Pitch (`spotify/basic-pitch`)
- **Zero GPU**: Real via `spaces` library + `@spaces.GPU` decorator
- **Stem Separation**: Demucs (htdemucs)
- **Audio Processing**: librosa, soundfile, scipy
- **Target Deployment**: HF Spaces with Zero GPU, CPU-only local execution

---

## Analysis Findings (Original Codebase Issues)

### Stubs & Placeholders
1. **agents.py:157** - `# TODO: Replace with scipy butterworth filters for production` (crude STFT bin zeroing)
2. **agents.py:444-463** - `_convert_to_noteseq()` returns empty `[]` (TODOs, not implemented)
3. **agents.py:306-309** - Silent mock fallback returns 3 hardcoded notes instead of erroring

### Bugs
4. **suno_postprocessor.py:281-284** - `_smooth_timing()` mutates Note objects in-place
5. **agents.py** - Model ID `mimbres/yourmt3` uses wrong case; correct ID is `mimbres/YourMT3` or `mimbres/YourMT3-cpu`

### Missing Components
6. **YourMT3+** - `_transcribe_with_yourmt3()` calls non-existent model; `_convert_to_noteseq()` is empty
7. **Zero GPU** - `@spaces.GPU` decorator present but `spaces` package missing from requirements
8. **Technique Detection** - Only slides detected; hammer-ons and pull-offs never generated (constants defined in main.py:71-74 but unused)
9. **ReaPack Integration** - `index.xml`, `reaper/TabAgent.lua`, `reaper/Settings.lua` don't exist
10. **Example Files** - `examples/guitar_solo.wav`, `examples/bass_groove.wav` referenced in UI but don't exist
11. **init_memory.py** - Referenced in main.py:48 but doesn't exist
12. **input/**, **output/** directories - Referenced but may not exist

### Documentation Issues
13. README contains `YOUR_USERNAME` placeholders
14. Processing time claims unverified
15. `git clone` URL is placeholder

### Dependencies
16. `spaces` package missing from requirements.txt
17. `transformers>=4.48.0` missing from requirements.txt
18. Dockerfile targets python:3.11-slim (needs 3.10)

---

## Execution Plan

### Phase 1: Core Bug Fixes
| # | Task | File:Line | Action |
|---|------|-----------|--------|
| 1.1 | Fix in-place mutation | `suno_postprocessor.py:281-284` | Clone NoteSequence.Note objects before modifying in `_smooth_timing()` |
| 1.2 | Remove silent mock fallback | `agents.py:306-309` | Raise `RuntimeError` if neither YourMT3+ nor Basic Pitch loads |
| 1.3 | Butterworth bass filters | `agents.py:157` | Replace STFT bin zeroing with `scipy.signal.butter` (remove TODO) |
| 1.4 | Implement YourMT3 output parsing | `agents.py:444-463` | Decode MIDI token sequences to NoteSequence (remove TODO) |

### Phase 2: YourMT3+ Primary Implementation
| # | Task | Description |
|---|------|-------------|
| 2.1 | Fix model ID | Change `mimbres/yourmt3` → `mimbres/YourMT3-cpu` |
| 2.2 | Implement `_transcribe_with_yourmt3()` | Load via Transformers, 16kHz resample, proper input format |
| 2.3 | Wire fallback chain | YourMT3+ → Basic Pitch → **Error** (no mock) |
| 2.4 | Fix `YOURMT3_AVAILABLE` flag logic | Actually check if `import yourmt3` succeeds AND model loads |
| 2.5 | Add `transformers>=4.48.0` to requirements.txt | |

### Phase 3: Dependencies (Python 3.10)
| # | Task | File | Action |
|---|------|------|--------|
| 3.1 | Update Dockerfile | `Dockerfile` | `FROM python:3.10-slim` |
| 3.2 | Add `spaces` package | `requirements.txt` | For Zero GPU |
| 3.3 | Add `transformers>=4.48.0` | `requirements.txt` | For YourMT3+ |
| 3.4 | Pin compatible versions | `requirements.txt` | numpy<2.0, verify tensorflow 3.10 compat |

### Phase 4: Zero GPU Implementation
| # | Task | File | Description |
|---|------|------|-------------|
| 4.1 | Verify `spaces` import | `app.py:18-24` | Graceful fallback if spaces not installed |
| 4.2 | `@spaces.GPU` decorator | `app.py:42` | Applied correctly |
| 4.3 | Processing time estimation | `app.py:104-189` | Add ETA display to match README claims |
| 4.4 | `gr.Progress()` integration | app.py | Ensure progress callbacks work through pipeline |

### Phase 5: Technique Detection Enhancement
| # | Task | File | Description |
|---|------|------|-------------|
| 5.1 | Hammer-on detection | `agents.py:725-742` | Same string, ascending fret, time_delta < 0.2 |
| 5.2 | Pull-off detection | `agents.py` | Same string, descending fret, time_delta < 0.2 |
| 5.3 | Wire technique constants | `main.py:71-74` | Ensure hammer/pull constants are actually used in export |

### Phase 6: ReaPack Integration
| # | Task | File | Description |
|---|------|------|-------------|
| 6.1 | ReaPack index | `index.xml` | Repository metadata, package listing |
| 6.2 | Main Reaper script | `reaper/TabAgent.lua` | Import selected audio, call Python pipeline, export tabs to track |
| 6.3 | Settings UI | `reaper/Settings.lua` | Tuning presets, instrument selection, export format options |
| 6.4 | Update README ReaPack URL | `README.md` | Replace placeholder URLs |

### Phase 7: Missing Files & init_memory.py
| # | Task | File | Description |
|---|------|------|-------------|
| 7.1 | Example audio placeholder | `examples/.gitkeep` | User will provide real recordings |
| 7.2 | Working directories | `input/.gitkeep`, `output/.gitkeep` | For local execution |
| 7.3 | init_memory.py | `init_memory.py` | Multiple preset profiles (see below) |

### Phase 7a: init_memory.py - Preset Profiles
```python
# Each profile includes:
# - tuning, num_strings, num_frets
# - onset_threshold, frame_threshold
# - prefer_low_strings (bass)
# - suno_aggressive_mode flag
# - technique_detection_sensitivity (0-1)

PROFILES = {
    "rock_standard": {
        "name": "Rock Guitar (Standard Tuning)",
        "description": "Standard EADGBE tuning for rock/metal, moderate processing",
        "tuning": [40, 45, 50, 55, 59, 64],   # E2-A2-D3-G3-B3-E4
        "num_strings": 6,
        "num_frets": 24,
        "onset_threshold": 0.5,
        "frame_threshold": 0.3,
        "suno_aggressive_mode": True,
        "technique_sensitivity": 0.7,
        "prefer_low_strings": False,
    },
    "rock_drop_d": {
        "name": "Rock Guitar (Drop D)",
        "description": "Drop D tuning (DADGBE) for heavy rock/metal",
        "tuning": [38, 45, 50, 55, 59, 64],   # D2-A2-D3-G3-B3-E4
        "num_strings": 6,
        "num_frets": 24,
        "onset_threshold": 0.5,
        "frame_threshold": 0.3,
        "suno_aggressive_mode": True,
        "technique_sensitivity": 0.7,
        "prefer_low_strings": False,
    },
    "classical": {
        "name": "Classical / Clean Guitar",
        "description": "Light processing for nylon-string/clean recordings, preserve dynamics",
        "tuning": [40, 45, 50, 55, 59, 64],
        "num_strings": 6,
        "num_frets": 19,
        "onset_threshold": 0.4,
        "frame_threshold": 0.25,
        "suno_aggressive_mode": False,
        "technique_sensitivity": 0.5,
        "prefer_low_strings": False,
    },
    "bass_5_string": {
        "name": "5-String Bass (B-E-A-D-G)",
        "description": "Standard 5-string bass tuning, low-string preference",
        "tuning": [23, 28, 33, 38, 43],        # B0-E1-A1-D2-G2
        "num_strings": 5,
        "num_frets": 24,
        "onset_threshold": 0.55,
        "frame_threshold": 0.35,
        "suno_aggressive_mode": True,
        "technique_sensitivity": 0.6,
        "prefer_low_strings": True,
    },
    "bass_4_string": {
        "name": "4-String Bass (E-A-D-G)",
        "description": "Standard 4-string bass tuning, low-string preference",
        "tuning": [28, 33, 38, 43],            # E1-A1-D2-G2
        "num_strings": 4,
        "num_frets": 24,
        "onset_threshold": 0.55,
        "frame_threshold": 0.35,
        "suno_aggressive_mode": True,
        "technique_sensitivity": 0.6,
        "prefer_low_strings": True,
    },
    "suno_aggressive": {
        "name": "Suno AI Audio (Aggressive Cleanup)",
        "description": "Maximum artifact removal for AI-generated audio",
        "tuning": [40, 45, 50, 55, 59, 64],
        "num_strings": 6,
        "num_frets": 24,
        "onset_threshold": 0.6,
        "frame_threshold": 0.4,
        "suno_aggressive_mode": True,
        "technique_sensitivity": 0.8,
        "prefer_low_strings": False,
    },
    "suno_conservative": {
        "name": "Suno AI Audio (Light Cleanup)",
        "description": "Light artifact removal, preserve more original content",
        "tuning": [40, 45, 50, 55, 59, 64],
        "num_strings": 6,
        "num_frets": 24,
        "onset_threshold": 0.55,
        "frame_threshold": 0.35,
        "suno_aggressive_mode": False,
        "technique_sensitivity": 0.7,
        "prefer_low_strings": False,
    },
    "live_band": {
        "name": "Live Band / Natural Recording",
        "description": "Minimal processing, preserve dynamics, natural feel",
        "tuning": [40, 45, 50, 55, 59, 64],
        "num_strings": 6,
        "num_frets": 24,
        "onset_threshold": 0.45,
        "frame_threshold": 0.25,
        "suno_aggressive_mode": False,
        "technique_sensitivity": 0.9,
        "prefer_low_strings": False,
    },
}
```

### Phase 8: README Alignment
| # | Task | Description |
|---|------|-------------|
| 8.1 | Replace `YOUR_USERNAME` placeholders | Point to actual GitHub repo URL |
| 8.2 | Verify processing time claims | Benchmark with 30s/60s/3min clips or adjust README |
| 8.3 | Fix `git clone` URL | Replace placeholder with real repo |
| 8.4 | Update feature list to match actual capabilities | YourMT3+/Basic Pitch dual-model |

### Phase 9: Testing
| # | Task | File | Description |
|---|------|------|-------------|
| 9.1 | Pipeline test | `test_pipeline.py` | End-to-end test with known audio |
| 9.2 | SplitterAgent tests | `tests/test_splitter.py` | Unit tests for stem separation |
| 9.3 | EarAgent tests | `tests/test_ear.py` | Unit tests for transcription |
| 9.4 | TabAgent tests | `tests/test_tab.py` | Unit tests for tab generation |
| 9.5 | Suno postprocessor tests | `tests/test_suno.py` | Unit tests for artifact detection |
| 9.6 | Docker build verification | | `docker build -t tab-agent .` |
| 9.7 | HF Spaces deployment test | | Verify Zero GPU integration works |

---

## Execution Order

```
Phase 3 (Dependencies - Dockerfile, requirements.txt)
    ↓
Phase 1 (Bug fixes - all 4 tasks in parallel)
Phase 7 (Missing files - init_memory.py, .gitkeep directories)
    ↓
Phase 2 (YourMT3+ primary implementation)
Phase 4 (Zero GPU - can run parallel)
    ↓
Phase 5 (Technique detection enhancement)
    ↓
Phase 6 (ReaPack integration - index.xml, Lua scripts)
Phase 8 (README alignment - can run parallel)
    ↓
Phase 9 (Testing - all tests before deployment)
```

## Files to Modify

| File | Action | Phases |
|------|--------|--------|
| `agents.py` | Major edits | 1, 2, 5 |
| `app.py` | Moderate edits | 4 |
| `main.py` | Minor edit | 5 |
| `suno_postprocessor.py` | Minor edit | 1 |
| `requirements.txt` | Edit | 3 |
| `Dockerfile` | Edit | 3 |
| `README.md` | Edit | 8 |

## Files to Create

| File | Phase |
|------|-------|
| `init_memory.py` | 7 |
| `input/.gitkeep` | 7 |
| `output/.gitkeep` | 7 |
| `examples/.gitkeep` | 7 |
| `index.xml` (ReaPack) | 6 |
| `reaper/TabAgent.lua` | 6 |
| `reaper/Settings.lua` | 6 |
| `test_pipeline.py` | 9 |
| `tests/` (directory) | 9 |

---

## Original Analysis: README Claims vs Reality

| README Claim | Original Status | Target Status |
|--------------|----------------|---------------|
| Basic Pitch AI Model | ✅ Implemented | ✅ Keep as fallback |
| YourMT3+ (modernized) | ❌ Stubbed | ✅ Primary model |
| Zero GPU Acceleration | ❌ Not implemented | ✅ Real implementation |
| Demucs stem separation | ✅ Implemented | ✅ Keep |
| Spatial audio processing | ✅ Implemented | ✅ Keep |
| Viterbi-style DP fingering | ⚠️ Basic DP | ✅ Keep |
| Slides/hammer-ons/pull-offs | ⚠️ Slides only | ✅ All three |
| Multi-track support | ✅ Implemented | ✅ Keep |
| MIDI/ASCII/JSON export | ✅ Implemented | ✅ Keep |
| Suno-aware processing | ⚠️ Bug (mutation) | ✅ Fixed |
| Example audio files | ❌ Missing | ⚠️ User provides |
| Reaper/ReaPack integration | ❌ Not implemented | ✅ Implement |
| init_memory.py | ❌ Missing | ✅ Implement with 8 presets |
| Python version | ⚠️ 3.11 Docker / 3.10+ docs | ✅ 3.10 |
