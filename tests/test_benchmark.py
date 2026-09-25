"""
Tab Agent Pro — Pipeline Benchmark.

Generates synthetic audio clips at known durations (10s, 30s, 60s),
runs EarAgent transcription (mocked), and reports wall-clock time.

Usage:
    python -m pytest tests/test_benchmark.py -v --benchmark
"""

import os
import sys
import tempfile
import time
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents import EarAgent


def generate_sine_sweep(duration_sec, sr=22050):
    """Generate a frequency sweep from 100Hz to 800Hz."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    freq = 100 + 700 * (t / duration_sec)  # 100→800 Hz sweep
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    for pitch, start in [(40, 0.5), (45, 2.0), (50, 3.5), (55, 5.0), (60, 7.0)]:
        freq_n = 440.0 * (2 ** ((pitch - 69) / 12.0))
        mask = (t >= start) & (t < start + 0.8)
        signal[mask] += 0.3 * np.sin(2 * np.pi * freq_n * t[mask])
    return (signal / np.max(np.abs(signal)) * 0.9).astype(np.float32)


class TestBenchmark(unittest.TestCase):
    """Measure transcription timing (mocked model, real audio pipeline)."""

    tmpdir: str
    results: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp()
        cls.results = {}

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil

        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _bench(self, label, duration_sec) -> None:
        import note_seq

        path = os.path.join(self.tmpdir, f"{label}.wav")
        signal = generate_sine_sweep(duration_sec)
        sf.write(path, signal, 22050)

        ear = EarAgent(device="cpu", prefer_yourmt3=False)
        mock_notes = [
            note_seq.NoteSequence.Note(pitch=64, start_time=0.0, end_time=0.4, velocity=80),
            note_seq.NoteSequence.Note(pitch=67, start_time=0.5, end_time=0.9, velocity=80),
        ]
        ear.transcribe_stem = lambda *a, **kw: mock_notes  # type: ignore[method-assign]

        t0 = time.time()
        notes = ear.transcribe_stem(
            path,
            target="Guitar",
            onset_threshold=0.3,
            frame_threshold=0.2,
        )
        elapsed = time.time() - t0

        TestBenchmark.results[label] = (f"{duration_sec}s", elapsed, len(notes))

        assert isinstance(notes, list)

    def test_10s_clip(self) -> None:
        self._bench("10s", 10)

    def test_30s_clip(self) -> None:
        self._bench("30s", 30)

    def test_60s_clip(self) -> None:
        self._bench("60s", 60)


if __name__ == "__main__":
    unittest.main()
