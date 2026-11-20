#!/usr/bin/env python3
# ruff: noqa
"""
Pipeline-level failure reproductions (real agents.py / main.py, stubbed backends).

  A. Demucs "succeeds" (health = loaded) while writing no stem files.
  B. A run that transcribes zero notes exits 0 and prints "Done" with no artifacts.
  C. A stale MIDI file from an earlier run survives a zero-note run (REAPER imports it).
  D. Memory arithmetic for a 60-minute input (no length limit on the CLI).

Run: PYTHONPATH=audit_poc/stubs:. /tmp/pocvenv/bin/python audit_poc/pipeline_probe.py
"""

import os
import shutil
import tempfile
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

import agents
import main
from monitoring import health

TMP = Path(tempfile.mkdtemp(prefix="tabagent-probe-"))


def silent_wav(path, seconds=2, sr=44100):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\0\0" * sr * seconds)
    return path


# ---------------------------------------------------------------------------
print("=== A. Demucs returns success but no stems ===")


class FakeSeparator:
    samplerate = 44100

    def __init__(self, **kw):
        pass

    def separate_audio_file(self, path):
        # A model whose stem keys do not match the hard-coded map, or a silent
        # write failure: no exception, nothing on disk.
        return None, {"drums": np.zeros((2, 10), dtype=np.float32)}


src = silent_wav(TMP / "song.wav")
splitter = agents.SplitterAgent(output_dir=str(TMP / "stems"), device="cpu")
with (
    patch.object(agents, "DEMUCS_API_AVAILABLE", True),
    patch.object(
        agents.demucs_api if hasattr(agents, "demucs_api") else agents,
        "Separator",
        FakeSeparator,
        create=True,
    ),
):
    agents.demucs_api = type("m", (), {"Separator": FakeSeparator})
    stems = splitter.separate_stems(str(src))

print("returned stems:", stems)
exists = all(Path(p).exists() for p in stems.values())
print("all returned paths exist:", exists)
print("health says demucs is:", health.as_dict()["components"]["demucs"])
try:
    splitter.process_guitars(stems["guitar"])
except Exception as e:
    print("next stage raises:", type(e).__name__, str(e)[:80])
# A working fallback returns *existing* files (copies of the mix) and marks demucs
# unavailable; the old code returned constructed paths that did not exist.
fallback_used = exists and health.as_dict()["components"]["demucs"] == "unavailable"
print("=> documented raw-audio fallback used:", fallback_used)

# ---------------------------------------------------------------------------
print("\n=== B. zero notes => exit 0, 'Done', no artifacts ===")
out = TMP / "out"
out.mkdir()
silent = silent_wav(TMP / "silent.wav")


class FakeSplitter:
    # main.py reads this to label the export (SplitterAgent sets "demucs" or
    # "raw_fallback"); the probe's fake separation is a passthrough.
    separation_mode = "raw_fallback"

    def __init__(self, output_dir=None, device=None):
        self.output_dir = output_dir or str(TMP / "stems")
        os.makedirs(self.output_dir, exist_ok=True)

    def separate_stems(self, path):
        return {"guitar": path, "bass": path}

    def process_guitars(self, path):
        return {"lead": path, "left": path, "right": path}

    def process_bass(self, path):
        return path


class FakeEar:
    def __init__(self, device=None):
        pass

    def transcribe_stem(self, *a, **k):
        return []  # Basic Pitch "succeeded" with nothing

    def humanize_and_clean(self, notes, is_bass=False):
        return notes

    def export_midi(self, notes, path):
        return agents.EarAgent.export_midi(self, notes, path)


class NoopSuno:
    def process(self, notes, is_suno, metrics):
        return notes


with (
    patch.object(main, "process_suno_audio", lambda *a, **k: (str(silent), False, {})),
    patch.object(main, "SplitterAgent", FakeSplitter),
    patch.object(main, "EarAgent", FakeEar),
    patch.object(main, "SunoNotePostprocessor", NoopSuno),
):
    try:
        main.main([str(silent), "-o", str(out)])
        code = 0
    except SystemExit as e:
        code = int(e.code or 0)
print("exit code:", code)
print("files produced:", sorted(p.name for p in out.rglob("*") if p.is_file()))

# ---------------------------------------------------------------------------
print("\n=== C. stale MIDI survives the zero-note run ===")
stale = out / "silent_lead_guitar.mid"
stale.write_bytes(b"MThd" + b"\0" * 20)
print("stale file present before re-run:", stale.exists())
with (
    patch.object(main, "process_suno_audio", lambda *a, **k: (str(silent), False, {})),
    patch.object(main, "SplitterAgent", FakeSplitter),
    patch.object(main, "EarAgent", FakeEar),
    patch.object(main, "SunoNotePostprocessor", NoopSuno),
):
    main.main([str(silent), "-o", str(out)])
print("stale file still present, untouched:", stale.exists())
print("REAPER import_midi() would import this file and report success")

# ---------------------------------------------------------------------------
print("\n=== D. 60-minute input, memory arithmetic (44.1 kHz) ===")
sec, sr, ch = 3600, 44100, 2
f32 = 4
print(
    f"  Demucs mix load      : {sec*sr*ch*f32/1e9:.2f} GB (+4 stems ~{4*sec*sr*ch*f32/1e9:.2f} GB)"
)
print(
    f"  process_guitars L/R  : {sec*sr*ch*f32/1e9:.2f} GB in, ~{3*sec*sr*f32/1e9:.2f} GB of mid/L/R copies"
)
n_fft, hop, bins = 2048, 512, 1025
frames = int(sec * 22050 / hop)
print(
    f"  Suno 22.05k STFT     : {frames*bins*16/1e9:.2f} GB complex per array "
    f"(mag+phase+recon ~3x, twice: {2*3*frames*bins*16/1e9:.1f} GB transient)"
)

shutil.rmtree(TMP, ignore_errors=True)
