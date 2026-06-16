"""
Tab Agent Pro — Unit Tests: Suno Postprocessor
Tests for Suno artifact detection, audio preprocessing, and note cleaning.
"""

import unittest
import os
import sys
import tempfile
import numpy as np
import soundfile as sf
import note_seq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suno_postprocessor import (
    SunoArtifactDetector,
    SunoAudioPreprocessor,
    SunoNotePostprocessor,
    process_suno_audio,
)


def make_note(pitch, start, end=0.5, velocity=80):
    return note_seq.NoteSequence.Note(
        pitch=pitch, start_time=start, end_time=end, velocity=velocity
    )


class TestSunoArtifactDetector(unittest.TestCase):
    """Tests for SunoArtifactDetector heuristic analysis."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.detector = SunoArtifactDetector()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_wav(self, filename, signal, sr=22050):
        path = os.path.join(self.tmpdir, filename)
        sf.write(path, signal, sr)
        return path

    def test_analyze_returns_bool_and_dict(self):
        """analyze returns (is_suno: bool, metrics: dict)."""
        # Simple sine wave (should NOT be detected as AI)
        sr = 22050
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        clean = 0.5 * np.sin(2 * np.pi * 440 * t)
        path = self._write_wav("clean.wav", clean)
        is_suno, metrics = self.detector.analyze(path)
        self.assertIsInstance(is_suno, (bool, np.bool_))
        self.assertIsInstance(metrics, dict)
        self.assertIn("hf_ratio", metrics)
        self.assertIn("spectral_flatness", metrics)

    def test_clean_sine_not_ai(self):
        """A pure sine wave should not be flagged as AI-generated."""
        sr = 22050
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        clean = 0.5 * np.sin(2 * np.pi * 440 * t)
        path = self._write_wav("clean.wav", clean)
        is_suno, _ = self.detector.analyze(path)
        self.assertFalse(is_suno)


class TestSunoAudioPreprocessor(unittest.TestCase):
    """Tests for audio preprocessing pipeline."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.preprocessor = SunoAudioPreprocessor()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_wav(self, filename, signal, sr=22050):
        path = os.path.join(self.tmpdir, filename)
        sf.write(path, signal, sr)
        return path

    def test_process_returns_path(self):
        """process returns the output file path."""
        sr = 22050
        signal = np.random.randn(sr).astype(np.float32)
        in_path = self._write_wav("input.wav", signal)
        out_path = os.path.join(self.tmpdir, "output.wav")
        result = self.preprocessor.process(in_path, out_path)
        self.assertEqual(result, out_path)
        self.assertTrue(os.path.exists(out_path))

    def test_highpass_removes_low_frequencies(self):
        """High-pass filter attenuates sub-40Hz content."""
        sr = 22050
        t = np.linspace(0, 1, sr, endpoint=False)
        # 20 Hz tone (below 40 Hz cutoff)
        signal = np.sin(2 * np.pi * 20 * t).astype(np.float32)
        filtered = self.preprocessor._highpass_filter(signal, sr, cutoff=40)
        # Filtered signal should have lower amplitude (attenuated)
        self.assertLess(np.max(np.abs(filtered)), np.max(np.abs(signal)))


class TestSunoNotePostprocessor(unittest.TestCase):
    """Tests for note post-processing (octave errors, spurious notes, timing)."""

    def setUp(self):
        self.postprocessor = SunoNotePostprocessor()

    def test_no_processing_for_clean_audio(self):
        """If is_suno=False, notes pass through unchanged."""
        notes = [make_note(60, 0.0), make_note(64, 0.5)]
        result = self.postprocessor.process(notes, is_suno=False, metrics={})
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].pitch, 60)

    def test_remove_octave_errors(self):
        """Simultaneous octave notes — keep the lower one."""
        notes = [
            make_note(60, 0.0),  # lower
            make_note(72, 0.01),  # octave above, same time
        ]
        result = self.postprocessor._remove_octave_errors(notes)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].pitch, 60)  # lower kept

    def test_remove_spurious_high_notes(self):
        """Ultra-high notes (>30% ratio) are removed."""
        notes = [
            make_note(60, 0.0),
            make_note(90, 0.1),  # ultra high
            make_note(91, 0.2),
            make_note(92, 0.3),
            make_note(64, 0.4),
        ]
        result = self.postprocessor._remove_spurious_high_notes(
            notes, threshold_pitch=84
        )
        self.assertEqual(len(result), 2)  # only the two normal notes remain

    def test_smooth_timing_does_not_mutate_input(self):
        """_smooth_timing returns new Note objects, doesn't mutate originals."""
        notes = [
            make_note(60, 0.123, 0.567),
            make_note(64, 0.678, 0.999),
        ]
        original_pitches = [n.pitch for n in notes]
        original_starts = [n.start_time for n in notes]

        result = self.postprocessor._smooth_timing(notes, quantize_ms=50)

        # Input notes must be unchanged
        for i, note in enumerate(notes):
            self.assertEqual(note.pitch, original_pitches[i])
            self.assertAlmostEqual(note.start_time, original_starts[i],
                                   msg=f"Input note {i} was mutated!")

        # Output notes have quantized timing
        for note in result:
            self.assertAlmostEqual(
                note.start_time % 0.05, 0.0, places=5,
                msg=f"start_time {note.start_time} not quantized to 50ms"
            )


if __name__ == "__main__":
    unittest.main()
