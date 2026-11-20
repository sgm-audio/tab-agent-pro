"""
Tab Agent Pro — Unit Tests: SplitterAgent
Tests for Demucs stem separation and spatial audio processing.
"""

import os

# Add parent directory to path for imports
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import SplitterAgent


class TestSplitterAgent(unittest.TestCase):
    """Unit tests for SplitterAgent spatial processing logic."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.splitter = SplitterAgent(output_dir=self.tmpdir)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_output_dir_created(self) -> None:
        """Output directory is created on init."""
        assert os.path.isdir(self.splitter.output_dir)

    def test_centred_signal_is_not_split_into_false_rhythm_parts(self) -> None:
        """A hard-centred guitar has no left or right part to recover.

        The old mid-side "centre kill" produced two rhythm stems by subtracting
        0.8*mid from each channel, which is the centre content at -14 dB with
        opposite polarity on each side — i.e. the lead, inverted, twice. A
        centred source now yields a lead stem and no rhythm stems at all.
        """
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False)
        mono_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        stereo = np.column_stack((mono_signal, mono_signal))
        stereo_path = os.path.join(self.tmpdir, "centred.wav")
        sf.write(stereo_path, stereo, sr)

        result = self.splitter.process_guitars(stereo_path)

        assert result["stereo"] is False
        assert result["left"] is None and result["right"] is None
        lead, _ = sf.read(result["lead"])
        assert np.max(np.abs(lead)) == pytest.approx(0.5, abs=1e-3)

    def test_panned_parts_are_separated_not_inverted(self) -> None:
        """Left/right rhythm parts are recovered from their own channels.

        Three distinct tones — one centred, one left-panned, one right-panned —
        must end up in three different stems. This is the property the old
        algebra could not provide: `L - mid` and `R - mid` are the same signal
        with opposite sign, so the left and right stems were identical.
        """
        sr = 22050
        t = np.arange(sr) / sr
        centre = 0.20 * np.sin(2 * np.pi * 440 * t)
        left_part = 0.15 * np.sin(2 * np.pi * 330 * t)
        right_part = 0.15 * np.sin(2 * np.pi * 550 * t)
        stereo = np.column_stack((centre + 0.9 * left_part, centre + 0.9 * right_part))
        path = os.path.join(self.tmpdir, "panned.wav")
        sf.write(path, stereo, sr, subtype="FLOAT")

        result = self.splitter.process_guitars(path)
        assert result["stereo"] is True

        def magnitude(signal, freq):
            spectrum = np.abs(np.fft.rfft(signal))
            freqs = np.fft.rfftfreq(len(signal), 1 / sr)
            return float(spectrum[np.argmin(np.abs(freqs - freq))])

        lead, _ = sf.read(result["lead"])
        rhythm_l, _ = sf.read(result["left"])
        rhythm_r, _ = sf.read(result["right"])

        # Each stem answers to its own pitch far more strongly than to the
        # other two (measured ~47 dB of separation on this mix).
        assert magnitude(lead, 440) > 100 * max(magnitude(lead, 330), magnitude(lead, 550))
        assert magnitude(rhythm_l, 330) > 100 * max(
            magnitude(rhythm_l, 440), magnitude(rhythm_l, 550)
        )
        assert magnitude(rhythm_r, 550) > 100 * max(
            magnitude(rhythm_r, 440), magnitude(rhythm_r, 330)
        )

    def test_mono_source_is_reported_not_faked(self) -> None:
        """A mono file must not produce rhythm stems from nothing."""
        sr = 22050
        mono = (0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
        path = os.path.join(self.tmpdir, "mono.wav")
        sf.write(path, mono, sr)

        result = self.splitter.process_guitars(path)
        assert result["stereo"] is False
        assert result["left"] is None and result["right"] is None
        assert os.path.exists(result["lead"])

    def test_process_guitars_returns_correct_keys(self) -> None:
        """process_guitars returns dict with lead, left, right keys."""
        sr = 44100
        stereo = np.random.randn(2, sr).astype(np.float32)
        stereo_path = os.path.join(self.tmpdir, "stereo_test.wav")
        sf.write(stereo_path, stereo.T, sr)

        result = self.splitter.process_guitars(stereo_path)
        assert set(result) == {"lead", "left", "right", "stereo"}
        assert result["stereo"] is True
        for key in ("lead", "left", "right"):
            assert os.path.exists(result[key])

    def test_process_bass_returns_path(self) -> None:
        """process_bass returns a valid file path."""
        sr = 44100
        mono = np.random.randn(sr).astype(np.float32)
        bass_path = os.path.join(self.tmpdir, "bass_test.wav")
        sf.write(bass_path, mono, sr)

        result = self.splitter.process_bass(bass_path)
        assert os.path.exists(result)

    def test_separate_with_subprocess_resolves_extensionless_demucs_dir(self) -> None:
        """_separate_with_subprocess finds Demucs stems under htdemucs/<stem_without_ext>/."""
        from unittest.mock import MagicMock, patch

        def complete_demucs(command, **kwargs):
            assert kwargs["check"]
            assert command[command.index("-d") + 1] == "cpu"
            demucs_dir = Path(command[command.index("-o") + 1]) / "htdemucs" / "song"
            demucs_dir.mkdir(parents=True)
            for stem in ("other", "bass"):
                sf.write(demucs_dir / f"{stem}.wav", np.zeros(128), 44100)
            return MagicMock(returncode=0)

        splitter = SplitterAgent(output_dir=self.tmpdir, device="cpu")
        with patch("agents.subprocess.run", side_effect=complete_demucs):
            stems = splitter._separate_with_subprocess("/tmp/song.wav", "song.wav")
        assert Path(stems["guitar"]).name == "other.wav"
        assert Path(stems["guitar"]).parent.name == "song"
        assert set(stems) == {"guitar", "bass"}
        assert all(sf.info(p).frames == 128 for p in stems.values())

    def test_separate_stems_failure_records_failed_metric(self) -> None:
        """When Demucs separation fails, default_metrics.stage_errors records the failure."""
        from unittest.mock import patch

        from monitoring import default_metrics

        before_errors = default_metrics.stage_errors.get("demucs_cli", 0)
        with (
            patch.object(
                self.splitter,
                "_separate_with_api",
                side_effect=RuntimeError("demucs failed"),
            ),
            patch.object(
                self.splitter,
                "_separate_with_subprocess",
                side_effect=RuntimeError("demucs cli failed"),
            ),
            patch.object(
                self.splitter,
                "_raw_audio_fallback",
                return_value={"guitar": "g.wav", "bass": "b.wav"},
            ),
        ):
            self.splitter.separate_stems("bad.wav")
        after_errors = default_metrics.stage_errors.get("demucs_cli", 0)
        assert after_errors == before_errors + 1

    def test_separate_with_api_honors_explicit_device(self) -> None:
        """SplitterAgent(device='cpu') passes device='cpu' to demucs_api.Separator."""
        from unittest.mock import MagicMock, patch

        import torch

        import agents as _a

        sp = SplitterAgent(output_dir=self.tmpdir, device="cpu")
        mock_api = MagicMock()
        mock_sep = MagicMock()
        mock_sep.samplerate = 44100
        mock_sep.separate_audio_file.return_value = (
            None,
            {"other": torch.zeros(128), "bass": torch.zeros(128)},
        )
        mock_api.Separator.return_value = mock_sep

        with (
            patch.object(_a, "demucs_api", mock_api, create=True),
            patch("torch.cuda.is_available", return_value=True),
        ):
            sp._separate_with_api("/tmp/song.wav", "song")

        assert mock_api.Separator.call_args.kwargs["device"] == "cpu"


if __name__ == "__main__":
    unittest.main()
