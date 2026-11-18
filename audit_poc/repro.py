# ruff: noqa
#!/usr/bin/env python3
"""
Reproductions for the reliability review. Run from the repo root:

    PYTHONPATH=audit_poc/stubs:. python audit_poc/repro.py

Uses tiny stubs for torch/librosa/soundfile/basic_pitch (the real ones are
multi-GB) but imports the *real* agents.py, main.py and monitoring.py.
"""

import json
import os
import tempfile
import time

import note_seq

import agents
import main
from agents import EarAgent, TabAgent
from monitoring import health

STANDARD = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4


def note(pitch, start, end=None, vel=80):
    return note_seq.NoteSequence.Note(
        pitch=pitch, start_time=start, end_time=end if end is not None else start + 0.2, velocity=vel
    )


def show(title):
    print(f"\n=== {title} ===")


def fmt(tab):
    return [(p["string"], p["fret"], p.get("technique"), round(p["start_time"], 3)) for p in tab]


# ---------------------------------------------------------------------------
show("1. A simultaneous two-note chord is assigned to ONE string")
# E4 (open high E) and G4 (3rd fret high E) at the same instant.
# On a guitar that is impossible: a string can only sound one pitch at a time.
agent = TabAgent(tuning=STANDARD)
tab = agent.generate_tab([note(64, 0.0), note(67, 0.0)], technique_sensitivity=0.7)
print("input :", [("pitch", 64, 0.0), ("pitch", 67, 0.0)])
print("output:", fmt(tab))
print("same string for both notes:", len(tab) == 2 and tab[0]["string"] == tab[1]["string"])
print("annotated technique       :", tab[1]["technique"])
print("rendered tab:")
print(main.render_tab_text(tab, num_strings=6))

# ---------------------------------------------------------------------------
show("2. Picked 16th notes are annotated as a slide (no pitch-bend data exists)")
# A2 (open A, fret 0) -> B2 (fret 2) on the same string 125 ms later: a normal
# picked run at 120 BPM 16ths. Basic Pitch is called with
# multiple_pitch_bends=False, so no bend evidence exists anywhere in the data.
tab = agent.generate_tab([note(45, 0.0), note(47, 0.125)], technique_sensitivity=0.7)
print("output:", fmt(tab))
print("annotated technique:", tab[1]["technique"])

# ---------------------------------------------------------------------------
show("3. Per-note drops are invisible in the exported JSON")
tab = agent.generate_tab([note(64, 0.0), note(30, 0.1)], technique_sensitivity=0.7)
print("output notes:", fmt(tab))
print("last_skipped_notes:", agent.last_skipped_notes)
with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as fh:
    path = fh.name
main.export_tab_to_json(tab, path, instrument="Guitar", num_strings=6)
payload = json.load(open(path))
print("JSON keys:", sorted(payload))
print("JSON mentions the dropped pitch 30?", "30" in json.dumps(payload["tablature"]))
print("JSON has a skipped/ warning field?", any(k for k in payload if "skip" in k or "warn" in k))
os.unlink(path)

# ---------------------------------------------------------------------------
show("4. Readiness flips to 503 after one failed transcription")
print("ready before:", health.is_ready(), "| component:", health.component_health())
ear = EarAgent(device="cpu")  # same construction the web UI does per request
try:
    ear.transcribe_stem("does-not-exist.wav", target="Guitar")
except Exception as exc:
    print("transcribe_stem raised:", type(exc).__name__, str(exc)[:70])
after = health.component_health()
print("component after:", after)
print("ready after  :", health.is_ready(), "   <- /ready now returns 503")
print("health status:", health.effective_status())

# ---------------------------------------------------------------------------
show("5. SunoNotePostprocessor._remove_octave_errors is O(n^2)")
from suno_postprocessor import SunoNotePostprocessor  # noqa: E402

proc = SunoNotePostprocessor()
for n in (1000, 2000, 4000, 8000):
    notes = [note(40 + (i % 24), i * 0.01) for i in range(n)]
    t0 = time.time()
    proc._remove_octave_errors(notes)
    dt = time.time() - t0
    print(f"  {n:5d} notes -> {dt:7.3f}s (extrapolated 60 min @ ~1200 notes/min: {dt * 15:8.1f}s)")
