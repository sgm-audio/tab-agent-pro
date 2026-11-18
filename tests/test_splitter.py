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

    def test_center_kill_factor_reduces_center(self) -> None:
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
        lead, _lead_sr = sf.read(result["lead"])
        rhythm_l, _ = sf.read(result["left"])
        rhythm_r, _ = sf.read(result["right"])

        # Lead (mid) should be stronger than rhythm sides (after centre kill)
        assert np.max(np.abs(lead)) > np.max(np.abs(rhythm_l))
        assert np.max(np.abs(lead)) > np.max(np.abs(rhythm_r))

    def test_process_guitars_returns_correct_keys(self) -> None:
        """process_guitars returns dict with lead, left, right keys."""
        sr = 44100
        stereo = np.random.randn(2, sr).astype(np.float32)
        stereo_path = os.path.join(self.tmpdir, "stereo_test.wav")
        sf.write(stereo_path, stereo.T, sr)

        result = self.splitter.process_guitars(stereo_path)
        assert "lead" in result
        assert "left" in result
        assert "right" in result
        for path in result.values():
            assert os.path.exists(path)

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


@pytest.mark.parametrize("mode", ["missing", "empty", "corrupt", "inconsistent", "nan"])
def test_cli_rejects_invalid_or_stale_stems(tmp_path, monkeypatch, mode):
    """A successful exit is insufficient; output must be fresh readable audio."""
    import agents

    old = tmp_path / "htdemucs" / "song"
    old.mkdir(parents=True)
    for name in ("bass", "other"):
        sf.write(old / f"{name}.wav", np.zeros(100), 8000)

    def run(cmd, **kwargs):
        dest = os.path.join(cmd[cmd.index("-o") + 1], "htdemucs", "song")
        os.makedirs(dest)
        sf.write(os.path.join(dest, "other.wav"), np.zeros(100), 8000)
        path = os.path.join(dest, "bass.wav")
        if mode == "empty":
            sf.write(path, np.zeros(0), 8000)
        elif mode == "corrupt":
            with open(path, "wb") as f:
                f.write(b"not audio")
        elif mode == "inconsistent":
            sf.write(path, np.zeros(101), 8000)
        elif mode == "nan":
            sf.write(path, np.full(100, np.nan), 8000, subtype="FLOAT")

    monkeypatch.setattr(agents.subprocess, "run", run)
    sp = SplitterAgent(str(tmp_path), device="cpu")
    with pytest.raises(ValueError):
        sp._separate_with_subprocess("/tmp/song.wav", "song.wav")
    assert not list(tmp_path.glob("demucs-cli-*"))


def test_api_invalid_audio_falls_back_to_valid_cli(tmp_path, monkeypatch):
    import torch

    import agents

    source = tmp_path / "song.wav"
    sf.write(source, np.zeros(100), 8000)
    from unittest.mock import MagicMock

    api = MagicMock()
    api.Separator.return_value.samplerate = 8000
    api.Separator.return_value.separate_audio_file.return_value = (
        None,
        {"other": torch.full((1, 100), float("nan")), "bass": torch.zeros((1, 100))},
    )
    monkeypatch.setattr(agents, "demucs_api", api, raising=False)
    monkeypatch.setattr(agents, "DEMUCS_API_AVAILABLE", True)

    def run(cmd, **kwargs):
        dest = os.path.join(cmd[cmd.index("-o") + 1], "htdemucs", "song")
        os.makedirs(dest)
        for stem in ("other", "bass"):
            sf.write(os.path.join(dest, f"{stem}.wav"), np.zeros(100), 8000)

    monkeypatch.setattr(agents.subprocess, "run", run)
    sp = SplitterAgent(str(tmp_path / "out"), device="cpu")
    result = sp.separate_stems(str(source))
    assert sp.separation_mode == "demucs"
    assert all(os.path.isfile(path) for path in result.values())
    assert not list((tmp_path / "out").glob("demucs-api-*"))
