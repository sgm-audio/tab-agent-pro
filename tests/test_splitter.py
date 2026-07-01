"""
Tab Agent Pro — Unit Tests: SplitterAgent
Tests for Demucs stem separation and spatial audio processing.
"""

import os

# Add parent directory to path for imports
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import SplitterAgent


class TestSplitterAgent(unittest.TestCase):
    """Unit tests for SplitterAgent spatial processing logic."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.splitter = SplitterAgent(output_dir=self.tmpdir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_output_dir_created(self):
        """Output directory is created on init."""
        self.assertTrue(os.path.isdir(self.splitter.output_dir))

    def test_center_kill_factor_reduces_center(self):
        """Mid-side processing isolates centre from sides."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # Create stereo: same signal in both channels (hard centre)
        mono_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        stereo = np.vstack((mono_signal, mono_signal))

        # Write temp stereo file
        stereo_path = os.path.join(self.tmpdir, "stereo_test.wav")
        sf.write(stereo_path, stereo.T, sr)

        # Process
        result = self.splitter.process_guitars(stereo_path)

        # Read back processed files
        lead, lead_sr = sf.read(result["lead"])
        rhythm_l, _ = sf.read(result["left"])
        rhythm_r, _ = sf.read(result["right"])

        # Lead (mid) should be stronger than rhythm sides (after centre kill)
        self.assertGreater(np.max(np.abs(lead)), np.max(np.abs(rhythm_l)))
        self.assertGreater(np.max(np.abs(lead)), np.max(np.abs(rhythm_r)))

    def test_process_guitars_returns_correct_keys(self):
        """process_guitars returns dict with lead, left, right keys."""
        sr = 44100
        stereo = np.random.randn(2, sr).astype(np.float32)
        stereo_path = os.path.join(self.tmpdir, "stereo_test.wav")
        sf.write(stereo_path, stereo.T, sr)

        result = self.splitter.process_guitars(stereo_path)
        self.assertIn("lead", result)
        self.assertIn("left", result)
        self.assertIn("right", result)
        for path in result.values():
            self.assertTrue(os.path.exists(path))

    def test_process_bass_returns_path(self):
        """process_bass returns a valid file path."""
        sr = 44100
        mono = np.random.randn(sr).astype(np.float32)
        bass_path = os.path.join(self.tmpdir, "bass_test.wav")
        sf.write(bass_path, mono, sr)

        result = self.splitter.process_bass(bass_path)
        self.assertTrue(os.path.exists(result))


if __name__ == "__main__":
    unittest.main()
