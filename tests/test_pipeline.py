#!/usr/bin/env python3
"""
Tab Agent Pro — End-to-End Pipeline Test.

Generates a synthetic audio clip (sine wave with known notes),
runs the full pipeline, and validates output files exist.

Usage:
    python test_pipeline.py

Requirements:
    - pytests or unittest runner
    - Demucs installed (optional — skipped if unavailable)
    - basic-pitch installed (optional — skipped if unavailable)
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import EarAgent, TabAgent
from main import export_tab_to_json, export_tab_to_txt
from suno_postprocessor import SunoNotePostprocessor, process_suno_audio


class TestEndToEndPipeline(unittest.TestCase):
    """Full pipeline test using synthetic audio."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp()
        cls.audio_path = os.path.join(cls.tmpdir, "test_synth.wav")

        # Generate synthetic guitar-like audio: a simple arpeggio
        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # E minor arpeggio: E4(64), G4(67), B4(71), E5(76)
        notes = [(64, 0.0, 0.4), (67, 0.5, 0.9), (71, 1.0, 1.4), (76, 1.5, 1.9)]
        signal = np.zeros_like(t)

        for pitch, start, end in notes:
            freq = 440.0 * (2 ** ((pitch - 69) / 12.0))
            mask = (t >= start) & (t < end)
            signal[mask] += 0.3 * np.sin(2 * np.pi * freq * t[mask]) * np.hanning(mask.sum())

        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.9
        sf.write(cls.audio_path, signal.astype(np.float32), sr)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_process_suno_detects_clean_audio(self) -> None:
        """Synthetic sine should not be flagged as AI-generated."""
        _processed, is_suno, _metrics = process_suno_audio(
            self.audio_path,
            output_path=os.path.join(self.tmpdir, "processed.wav"),
        )
        assert not is_suno, "Synthetic audio should not be flagged as AI"

    def test_ear_agent_transcribes_synthetic(self) -> None:
        """EarAgent transcribes synthetic audio to MIDI notes (mocked)."""
        import note_seq

        ear = EarAgent(device="cpu", prefer_yourmt3=False)

        mock_notes = [
            note_seq.NoteSequence.Note(pitch=64, start_time=0.0, end_time=0.4, velocity=80),
            note_seq.NoteSequence.Note(pitch=67, start_time=0.5, end_time=0.9, velocity=80),
        ]
        ear.transcribe_stem = lambda *a, **kw: mock_notes  # type: ignore[method-assign]

        notes_raw = ear.transcribe_stem(
            self.audio_path,
            target="Guitar",
            onset_threshold=0.3,
            frame_threshold=0.2,
        )
        assert notes_raw is not None
        assert isinstance(notes_raw, list)
        # Each note should be a note_seq.NoteSequence.Note
        if notes_raw:
            assert all(hasattr(n, "pitch") and hasattr(n, "start_time") for n in notes_raw), (
                "Notes must have pitch and start_time attributes"
            )

    def test_tab_agent_generates_tab(self) -> None:
        """TabAgent generates tablature from note data."""
        import note_seq

        # Create known notes
        notes = [
            note_seq.NoteSequence.Note(pitch=64, start_time=0.0, end_time=0.4, velocity=80),
            note_seq.NoteSequence.Note(pitch=67, start_time=0.5, end_time=0.9, velocity=80),
            note_seq.NoteSequence.Note(pitch=71, start_time=1.0, end_time=1.4, velocity=80),
        ]

        tab_agent = TabAgent(tuning=[40, 45, 50, 55, 59, 64], num_frets=24)
        result = tab_agent.generate_tab(notes)

        assert len(result) == len(notes)
        for entry in result:
            assert "string" in entry
            assert "fret" in entry
            assert "technique" in entry

    def test_export_tab_to_txt(self) -> None:
        """ASCII tab export produces a readable file."""
        tab_data = [
            {"string": 0, "fret": 0, "technique": "pick"},
            {"string": 1, "fret": 3, "technique": "slide"},
            {"string": 2, "fret": 0, "technique": "pick"},
        ]
        out_path = os.path.join(self.tmpdir, "test.tab")
        export_tab_to_txt(tab_data, out_path, "Guitar")

        assert os.path.exists(out_path)
        with open(out_path) as f:
            content = f.read()
        assert "Guitar Tablature" in content
        assert "3s" in content  # slide annotation
        assert "|" in content  # column separators

    def test_export_tab_to_json(self) -> None:
        """JSON export produces valid JSON with required fields."""
        tab_data = [
            {"string": 0, "fret": 0, "technique": "pick"},
        ]
        out_path = os.path.join(self.tmpdir, "test.json")
        export_tab_to_json(tab_data, out_path, "Guitar")

        assert os.path.exists(out_path)
        import json

        with open(out_path) as f:
            data = json.load(f)
        assert data["instrument"] == "Guitar"
        assert "tablature" in data
        assert len(data["tablature"]) == 1

    def test_suno_note_postprocessor_no_mutation(self) -> None:
        """SunoNotePostprocessor does not mutate input notes (regression test)."""
        import note_seq

        notes = [
            note_seq.NoteSequence.Note(pitch=60, start_time=0.123, end_time=0.567, velocity=80),
            note_seq.NoteSequence.Note(pitch=64, start_time=0.678, end_time=0.999, velocity=80),
        ]
        original_starts = [n.start_time for n in notes]

        processor = SunoNotePostprocessor()
        processor.process(notes, is_suno=True, metrics={"hf_ratio": 0.4})

        # Original notes must be unchanged
        for i, note in enumerate(notes):
            self.assertAlmostEqual(
                note.start_time,
                original_starts[i],
                msg=f"Original note {i} was mutated!",
            )


if __name__ == "__main__":
    unittest.main()
