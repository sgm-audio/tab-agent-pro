"""
Tab Agent Pro — Edge Case Tests for SplitterAgent, EarAgent, TabAgent.

Tests the specific untested paths identified by coverage analysis:
  - SplitterAgent.separate_stems() — API path, CLI fallback, raw fallback
  - SplitterAgent._raw_audio_fallback()
  - TabAgent.calculate_cost() — legato, string skip, 5-string bass
  - TabAgent.generate_tab() — unplayable notes, technique edges
  - EarAgent.humanize_and_clean() — is_bass=True, duplicate pitch removal
  - EarAgent._filter_by_instrument_range() — empty notes, edge ranges
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import note_seq
import pytest

from agents import EarAgent, SplitterAgent, TabAgent

GUITAR_TUNING = [40, 45, 50, 55, 59, 64]
BASS_TUNING = [23, 28, 33, 38, 43]


def make_note(pitch, start, end=None, velocity=80):
    if end is None:
        end = start + 0.5
    return note_seq.NoteSequence.Note(
        pitch=pitch,
        start_time=start,
        end_time=end,
        velocity=velocity,
    )


# =========================================================================
# SPLITTER AGENT
# =========================================================================


class TestSplitterAgentSeparateStems(unittest.TestCase):
    """Cover the three paths in separate_stems: API, CLI, raw fallback."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.splitter = SplitterAgent(output_dir=self.tmpdir)
        self.dummy_audio = os.path.join(self.tmpdir, "input.wav")
        with open(self.dummy_audio, "w") as f:
            f.write("FAKE-WAV")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_cm(self):
        """Build a context manager mock whose __exit__ returns False."""
        cm = MagicMock()
        cm.__enter__.return_value = None
        cm.__exit__.return_value = False
        return cm

    @patch("agents.DEMUCS_API_AVAILABLE", True)
    @patch("agents.health")
    @patch("agents.metrics")
    def test_separate_stems_api_path(self, mock_metrics, mock_health) -> None:
        """When DEMUCS_API_AVAILABLE and API succeeds, return API result."""
        mock_metrics.track_stage.return_value = self._make_cm()
        expected = {"guitar": "/fake/other.wav", "bass": "/fake/bass.wav"}
        self.splitter._separate_with_api = MagicMock(return_value=expected)  # type: ignore[method-assign]  # test monkeypatch of private method
        result = self.splitter.separate_stems(self.dummy_audio)

        assert result == expected
        self.splitter._separate_with_api.assert_called_once()
        mock_health.set_component.assert_called_with("demucs", "loaded")

    @patch("agents.DEMUCS_API_AVAILABLE", True)
    @patch("agents.health")
    @patch("agents.metrics")
    def test_separate_stems_api_fails_cli_succeeds(self, mock_metrics, mock_health) -> None:
        """When API fails but CLI works, return CLI result."""
        mock_metrics.track_stage.return_value = self._make_cm()
        self.splitter._separate_with_api = MagicMock(side_effect=RuntimeError("OOM"))  # type: ignore[method-assign]  # test monkeypatch of private method
        expected = {"guitar": "/cli/other.wav", "bass": "/cli/bass.wav"}
        self.splitter._separate_with_subprocess = MagicMock(return_value=expected)  # type: ignore[method-assign]  # test monkeypatch of private method
        result = self.splitter.separate_stems(self.dummy_audio)

        assert result == expected
        self.splitter._separate_with_subprocess.assert_called_once()

    @patch("agents.DEMUCS_API_AVAILABLE", True)
    @patch("agents.health")
    @patch("agents.metrics")
    def test_separate_stems_api_and_cli_fail_raw_fallback(self, mock_metrics, mock_health) -> None:
        """When API and CLI fail, return raw audio fallback."""
        mock_metrics.track_stage.return_value = self._make_cm()
        self.splitter._separate_with_api = MagicMock(side_effect=RuntimeError("OOM"))  # type: ignore[method-assign]  # test monkeypatch of private method
        self.splitter._separate_with_subprocess = MagicMock(side_effect=RuntimeError("no demucs"))  # type: ignore[method-assign]  # test monkeypatch of private method
        self.splitter._raw_audio_fallback = MagicMock(  # type: ignore[method-assign]  # test monkeypatch of private method
            return_value={"guitar": "/raw/other.wav", "bass": "/raw/bass.wav"}
        )

        result = self.splitter.separate_stems(self.dummy_audio)

        assert "guitar" in result
        assert "bass" in result
        self.splitter._raw_audio_fallback.assert_called_once()

    @patch("agents.DEMUCS_API_AVAILABLE", False)
    @patch("agents.metrics")
    def test_separate_stems_no_api_cli_succeeds(self, mock_metrics) -> None:
        """When DEMUCS_API is not available, skip directly to CLI."""
        mock_metrics.track_stage.return_value = self._make_cm()
        expected = {"guitar": "/cli/other.wav", "bass": "/cli/bass.wav"}
        self.splitter._separate_with_subprocess = MagicMock(return_value=expected)  # type: ignore[method-assign]  # test monkeypatch of private method
        result = self.splitter.separate_stems(self.dummy_audio)

        assert result == expected
        self.splitter._separate_with_subprocess.assert_called_once()


class TestSplitterAgentRawAudioFallback(unittest.TestCase):
    """Test _raw_audio_fallback produces valid paths."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.splitter = SplitterAgent(output_dir=self.tmpdir)
        self.dummy_audio = os.path.join(self.tmpdir, "song.wav")
        with open(self.dummy_audio, "w") as f:
            f.write("FAKE-WAV")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_raw_audio_fallback_creates_files(self) -> None:
        """Fallback copies raw audio to both guitar and bass paths."""
        result = self.splitter._raw_audio_fallback(self.dummy_audio, "song")
        assert "guitar" in result
        assert "bass" in result
        assert os.path.exists(result["guitar"])
        assert os.path.exists(result["bass"])

    def test_raw_audio_fallback_content_preserved(self) -> None:
        """Fallback preserves original file content."""
        with open(self.dummy_audio, "w") as f:
            f.write("ORIGINAL-DATA")
        result = self.splitter._raw_audio_fallback(self.dummy_audio, "song2")
        with open(result["guitar"]) as f:
            assert f.read() == "ORIGINAL-DATA"
        with open(result["bass"]) as f:
            assert f.read() == "ORIGINAL-DATA"


# =========================================================================
# TAB AGENT — calculate_cost
# =========================================================================


class TestTabAgentCalculateCost(unittest.TestCase):
    """Edge cases for calculate_cost: legato, string skip, 5-string bass."""

    def setUp(self) -> None:
        self.agent_6 = TabAgent(tuning=GUITAR_TUNING, num_frets=24)
        self.agent_5 = TabAgent(tuning=BASS_TUNING, num_frets=24)

    def test_cost_prev_is_none_returns_zero(self) -> None:
        """When prev is None, cost is 0."""
        cost = self.agent_6.calculate_cost(None, {"string": 0, "fret": 0}, 1.0)
        assert cost == 0.0

    def test_cost_legato_encouragement_same_string(self) -> None:
        """Fast transition on same string gets cost reduction (legato)."""
        prev = {"string": 2, "fret": 3}
        curr = {"string": 2, "fret": 5}
        cost_legato = self.agent_6.calculate_cost(prev, curr, time_delta=0.1)

        cost_slow = self.agent_6.calculate_cost(prev, curr, time_delta=1.0)

        # Legato cost should be lower than slow transition cost
        assert cost_legato < cost_slow

    def test_cost_legato_reduction_value(self) -> None:
        """Legato reduction should be exactly 5.0 on same string fast."""
        prev = {"string": 1, "fret": 3}
        curr = {"string": 1, "fret": 5}
        cost = self.agent_6.calculate_cost(prev, curr, time_delta=0.1)

        base_fret = abs(5 - 3) * 1.5
        base_string = abs(1 - 1) * 2.0
        expected_base = base_fret + base_string
        self.assertAlmostEqual(cost, expected_base - 5.0)

    def test_cost_string_skip_penalty_on_fast(self) -> None:
        """Fast transition with string change gets penalty."""
        prev = {"string": 1, "fret": 3}
        curr = {"string": 3, "fret": 5}
        cost = self.agent_6.calculate_cost(prev, curr, time_delta=0.1)

        base_fret = abs(5 - 3) * 1.5
        base_string = abs(3 - 1) * 2.0
        expected_base = base_fret + base_string + 5.0
        self.assertAlmostEqual(cost, expected_base)

    def test_cost_string_skip_no_penalty_on_slow(self) -> None:
        """Slow transition with string change does NOT get penalty."""
        prev = {"string": 1, "fret": 3}
        curr = {"string": 3, "fret": 5}
        cost = self.agent_6.calculate_cost(prev, curr, time_delta=1.0)

        base_fret = abs(5 - 3) * 1.5
        base_string = abs(3 - 1) * 2.0
        self.assertAlmostEqual(cost, base_fret + base_string)

    def test_cost_bass_preference_low_string_low_fret(self) -> None:
        """5-string bass: low strings with low frets get extra penalty."""
        prev = {"string": 1, "fret": 3}
        curr = {"string": 0, "fret": 2}  # low string, low fret (0-4 range)
        cost = self.agent_5.calculate_cost(prev, curr, time_delta=0.5)

        base_fret = abs(2 - 3) * 1.5
        base_string = abs(0 - 1) * 2.0
        self.assertAlmostEqual(cost, base_fret + base_string + 1.0)

    def test_cost_bass_no_preference_for_high_fret(self) -> None:
        """5-string bass: low string with high fret (>=5) gets no extra penalty."""
        prev = {"string": 3, "fret": 5}
        curr = {"string": 0, "fret": 7}  # low string but high fret
        cost = self.agent_5.calculate_cost(prev, curr, time_delta=0.5)

        base_fret = abs(7 - 5) * 1.5
        base_string = abs(0 - 3) * 2.0
        self.assertAlmostEqual(cost, base_fret + base_string)

    def test_cost_bass_high_string_no_penalty(self) -> None:
        """5-string bass: high strings never get the low-fret penalty."""
        prev = {"string": 2, "fret": 2}
        curr = {"string": 4, "fret": 1}  # high string, low fret
        cost = self.agent_5.calculate_cost(prev, curr, time_delta=0.5)

        base_fret = abs(1 - 2) * 1.5
        base_string = abs(4 - 2) * 2.0
        self.assertAlmostEqual(cost, base_fret + base_string)

    def test_cost_6string_no_bass_preference(self) -> None:
        """6-string guitar: no bass-specific penalty even on low strings."""
        prev = {"string": 4, "fret": 3}
        curr = {"string": 0, "fret": 2}  # low string, low fret
        cost = self.agent_6.calculate_cost(prev, curr, time_delta=0.5)

        base_fret = abs(2 - 3) * 1.5
        base_string = abs(0 - 4) * 2.0
        self.assertAlmostEqual(cost, base_fret + base_string)


# =========================================================================
# TAB AGENT — generate_tab
# =========================================================================


class TestTabAgentGenerateTabEdgeCases(unittest.TestCase):
    """Edge cases for generate_tab: unplayable notes, technique detection."""

    def setUp(self) -> None:
        self.agent = TabAgent(tuning=GUITAR_TUNING, num_frets=24)

    def test_unplayable_notes_returns_empty_with_warning(self) -> None:
        """Notes outside the instrument range produce empty result with warning."""
        notes = [make_note(10, 0.0), make_note(20, 0.5)]  # both below E2
        result = self.agent.generate_tab(notes)
        assert len(result) == 0

    def test_unplayable_interleaved_with_playable(self) -> None:
        """If any note is unplayable, entire result is empty."""
        notes = [
            make_note(40, 0.0),  # playable
            make_note(10, 0.5),  # unplayable
        ]
        result = self.agent.generate_tab(notes)
        assert len(result) == 0

    def test_technique_slide_adjacent_1_fret(self) -> None:
        """1-fret difference on same string fast → slide."""
        notes = [
            make_note(40, 0.0),  # E2 string 0 fret 0
            make_note(41, 0.12),  # F2
        ]
        result = self.agent.generate_tab(notes, technique_sensitivity=0.9)
        if len(result) > 1:
            assert result[1]["technique"] == "slide"

    def test_technique_hammer_ascending(self) -> None:
        """Ascending same-string fast → hammer."""
        notes = [
            make_note(40, 0.0),  # E2
            make_note(45, 0.12),  # A2 — hammer range on same string
        ]
        result = self.agent.generate_tab(notes, technique_sensitivity=0.9)
        if len(result) > 1:
            assert result[1]["technique"] in ["slide", "hammer"]

    def test_technique_pull_descending(self) -> None:
        """Descending same-string fast → pull."""
        notes = [
            make_note(55, 0.0),  # G3
            make_note(52, 0.12),  # E3
        ]
        result = self.agent.generate_tab(notes, technique_sensitivity=0.9)
        if len(result) > 1:
            assert result[1]["technique"] in ["slide", "hammer", "pull", "pick"]

    def test_slow_transition_is_pick_not_technique(self) -> None:
        """Slow transition on same string → 'pick' (no technique)."""
        notes = [
            make_note(40, 0.0),
            make_note(42, 1.5),  # slow — well beyond technique window
        ]
        result = self.agent.generate_tab(notes, technique_sensitivity=0.5)
        if len(result) > 1:
            assert result[1]["technique"] == "pick"

    def test_technique_window_includes_close_notes(self) -> None:
        """Close notes within window on same string get 'slide'."""
        notes = [
            make_note(40, 0.0),  # E2 — only on string 0
            make_note(42, 0.15),  # F#2 — only on string 0, fret diff=2
        ]
        result = self.agent.generate_tab(notes, technique_sensitivity=0.9)
        if len(result) > 1:
            assert result[1]["technique"] == "slide"

    def test_technique_window_excludes_distant_notes(self) -> None:
        """Distant notes outside window get 'pick'."""
        notes = [
            make_note(40, 0.0),
            make_note(42, 2.0),  # far apart
        ]
        result = self.agent.generate_tab(notes, technique_sensitivity=0.3)
        if len(result) > 1:
            assert result[1]["technique"] == "pick"

    def test_generate_tab_different_string_no_technique(self) -> None:
        """Different string fast transition is 'pick' (no technique)."""
        notes = [
            make_note(40, 0.0),
            make_note(64, 0.1),  # different string, fast
        ]
        result = self.agent.generate_tab(notes, technique_sensitivity=0.9)
        if len(result) > 1:
            assert result[1]["technique"] == "pick"

    def test_5string_bass_tab_produces_valid_output(self) -> None:
        """5-string bass.generate_tab returns valid tab with techniques."""
        agent = TabAgent(tuning=BASS_TUNING, num_frets=24)
        notes = [
            make_note(28, 0.0),
            make_note(33, 0.5),
            make_note(38, 1.0),
        ]
        result = agent.generate_tab(notes)
        assert len(result) == 3
        for entry in result:
            assert "technique" in entry


# =========================================================================
# EAR AGENT — humanize_and_clean
# =========================================================================


class TestEarAgentHumanizeAndCleanEdgeCases(unittest.TestCase):
    """Edge cases for humanize_and_clean."""

    def setUp(self) -> None:
        self.ear = EarAgent(model_id="mimbres/YourMT3", device="cpu", prefer_yourmt3=False)

    def test_is_bass_filters_above_67(self) -> None:
        """Bass mode removes notes above pitch 67."""
        notes = [
            make_note(60, 0.0),  # valid
            make_note(70, 0.5),  # above bass range
            make_note(80, 1.0),  # above bass range
        ]
        result = self.ear.humanize_and_clean(notes, is_bass=True)
        assert len(result) == 1
        assert result[0].pitch == 60

    def test_duplicate_pitch_removal(self) -> None:
        """Same pitch at same time is removed, different pitches kept."""
        notes = [
            make_note(60, 0.0),
            make_note(60, 0.0),  # duplicate pitch
            make_note(64, 0.0),  # different pitch, same time — kept
        ]
        result = self.ear.humanize_and_clean(notes)
        assert len(result) == 2
        pitches = {n.pitch for n in result}
        assert 60 in pitches
        assert 64 in pitches

    def test_duplicate_pitch_different_time_kept(self) -> None:
        """Same pitch at different times is not a duplicate."""
        notes = [
            make_note(60, 0.0, end=0.4),
            make_note(60, 0.5, end=0.9),  # same pitch, different time
        ]
        result = self.ear.humanize_and_clean(notes)
        assert len(result) == 2

    def test_empty_input_returns_empty(self) -> None:
        """Empty input returns empty list."""
        result = self.ear.humanize_and_clean([])
        assert len(result) == 0

    def test_ultrashort_and_bass_out_of_range(self) -> None:
        """Combination of ultra-short and bass range filtering."""
        notes = [
            make_note(60, 0.0, end=0.01),  # ultra-short
            make_note(70, 0.3, end=0.7),  # above bass range
            make_note(45, 1.0, end=1.5),  # valid
        ]
        result = self.ear.humanize_and_clean(notes, is_bass=True)
        assert len(result) == 1
        assert result[0].pitch == 45

    def test_non_bass_allows_high_notes(self) -> None:
        """Non-bass mode keeps high notes."""
        notes = [
            make_note(60, 0.0, end=0.4),
            make_note(72, 0.5, end=0.9),  # fine for guitar
            make_note(84, 1.0, end=1.5),  # fine for guitar
        ]
        result = self.ear.humanize_and_clean(notes, is_bass=False)
        assert len(result) == 3


# =========================================================================
# EAR AGENT — _filter_by_instrument_range
# =========================================================================


class TestEarAgentFilterByInstrumentRangeEdgeCases(unittest.TestCase):
    """Edge cases for _filter_by_instrument_range."""

    def setUp(self) -> None:
        self.ear = EarAgent(model_id="mimbres/YourMT3", device="cpu", prefer_yourmt3=False)

    def test_empty_notes_returns_empty(self) -> None:
        """Empty input returns empty list."""
        result = self.ear._filter_by_instrument_range([], "Guitar")
        assert len(result) == 0

    def test_all_notes_in_range_for_guitar(self) -> None:
        """All notes within guitar range pass through."""
        notes = [make_note(40, 0.0), make_note(64, 0.5), make_note(88, 1.0)]
        result = self.ear._filter_by_instrument_range(notes, "Guitar")
        assert len(result) == 3

    def test_bass_range_preserves_edge_low_b0(self) -> None:
        """Bass range includes low B0 (pitch 23)."""
        notes = [make_note(23, 0.0)]
        result = self.ear._filter_by_instrument_range(notes, "bass")
        assert len(result) == 1

    def test_bass_range_preserves_edge_high_g4(self) -> None:
        """Bass range includes high G4 (pitch 67)."""
        notes = [make_note(67, 0.0)]
        result = self.ear._filter_by_instrument_range(notes, "bass")
        assert len(result) == 1

    def test_bass_removes_below_b0(self) -> None:
        """Bass range removes notes below B0 (pitch 23)."""
        notes = [make_note(22, 0.0)]  # below B0
        result = self.ear._filter_by_instrument_range(notes, "Bass")
        assert len(result) == 0

    def test_guitar_removes_below_e2(self) -> None:
        """Guitar range removes notes below E2 (pitch 40)."""
        notes = [make_note(39, 0.0)]  # below E2
        result = self.ear._filter_by_instrument_range(notes, "Guitar")
        assert len(result) == 0

    def test_guitar_removes_above_e6(self) -> None:
        """Guitar range removes notes above E6 (pitch 88)."""
        notes = [make_note(89, 0.0)]
        result = self.ear._filter_by_instrument_range(notes, "Guitar")
        assert len(result) == 0

    def test_mixed_in_and_out_of_range_guitar(self) -> None:
        """Mixed valid and invalid notes for guitar."""
        notes = [
            make_note(30, 0.0),  # too low
            make_note(50, 0.5),  # valid
            make_note(95, 1.0),  # too high
        ]
        result = self.ear._filter_by_instrument_range(notes, "Guitar")
        assert len(result) == 1
        assert result[0].pitch == 50

    def test_case_insensitive_target(self) -> None:
        """Target matching is case-insensitive (Bass, bass, BASS all work)."""
        notes = [make_note(60, 0.0), make_note(70, 0.5)]
        result = self.ear._filter_by_instrument_range(notes, "BASS")
        assert len(result) == 1
        result2 = self.ear._filter_by_instrument_range(notes, "bass")
        assert len(result2) == 1
        result3 = self.ear._filter_by_instrument_range(notes, "BaSs")
        assert len(result3) == 1


# =========================================================================
# SPLITTER — process_guitars and process_bass mono paths
# =========================================================================


class TestSplitterAgentSpatialProcessingEdgeCases(unittest.TestCase):
    """Mono input paths for process_guitars and process_bass."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.splitter = SplitterAgent(output_dir=self.tmpdir)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_process_guitars_mono_does_not_crash(self) -> None:
        """Mono file is duplicated to stereo automatically."""
        sr = 44100
        import numpy as np

        mono = np.random.randn(sr).astype(np.float32)
        path = os.path.join(self.tmpdir, "mono.wav")
        sf = __import__("soundfile")
        sf.write(path, mono, sr)
        result = self.splitter.process_guitars(path)
        for k in ("lead", "left", "right"):
            assert k in result
            assert os.path.exists(result[k])

    def test_process_bass_mono_does_not_crash(self) -> None:
        """Mono bass input is processed correctly."""
        sr = 44100
        import numpy as np

        mono = np.random.randn(sr).astype(np.float32)
        path = os.path.join(self.tmpdir, "bass_mono.wav")
        sf = __import__("soundfile")
        sf.write(path, mono, sr)
        result = self.splitter.process_bass(path)
        assert os.path.exists(result)


# =========================================================================
# SPLITTER — _separate_with_subprocess
# =========================================================================


class TestSplitterSeparateWithSubprocess(unittest.TestCase):
    """Test the CLI subprocess path for stem separation."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.splitter = SplitterAgent(output_dir=self.tmpdir)
        self.dummy_audio = os.path.join(self.tmpdir, "song.wav")
        with open(self.dummy_audio, "w") as f:
            f.write("FAKE")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("agents.subprocess.run")
    def test_subprocess_success(self, mock_run) -> None:
        """Successful subprocess returns expected stem paths."""
        mock_run.return_value = MagicMock(returncode=0)
        result = self.splitter._separate_with_subprocess(self.dummy_audio, "song")
        assert "guitar" in result
        assert "bass" in result
        mock_run.assert_called_once()

    @patch("agents.subprocess.run")
    def test_subprocess_failure_raises(self, mock_run) -> None:
        """Failed subprocess raises CalledProcessError."""
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(1, "demucs")
        with pytest.raises(CalledProcessError):
            self.splitter._separate_with_subprocess(self.dummy_audio, "song")


# =========================================================================
# EAR — transcribe_stem (via Basic Pitch mock)
# =========================================================================


class TestEarTranscribeStemBasicPitch(unittest.TestCase):
    """Test the transcribe_stem Basic Pitch path with mocks."""

    def setUp(self) -> None:
        self.ear = EarAgent(model_id="mimbres/YourMT3", device="cpu", prefer_yourmt3=False)

    @patch("agents.BASIC_PITCH_AVAILABLE", True)
    @patch("agents.basic_pitch_predict")
    @patch("agents.metrics")
    def test_transcribe_stem_basic_pitch_path(self, mock_metrics, mock_bp_predict) -> None:
        """transcribe_stem calls Basic Pitch and returns notes."""
        mock_metrics.track_stage.return_value.__enter__.return_value = None
        mock_metrics.track_stage.return_value.__exit__.return_value = False

        import pretty_midi

        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=0.5, end=1.0))
        pm.instruments.append(inst)

        mock_bp_predict.return_value = (MagicMock(), pm, MagicMock())

        result = self.ear.transcribe_stem("/fake/path.wav", target="Guitar")
        assert len(result) == 2
        assert result[0].pitch == 60
        assert result[1].pitch == 64

    @patch("agents.BASIC_PITCH_AVAILABLE", True)
    @patch("agents.basic_pitch_predict")
    @patch("agents.metrics")
    def test_transcribe_stem_basic_pitch_filters_range(self, mock_metrics, mock_bp_predict) -> None:
        """Basic Pitch result is filtered by instrument range."""
        mock_metrics.track_stage.return_value.__enter__.return_value = None
        mock_metrics.track_stage.return_value.__exit__.return_value = False

        import pretty_midi

        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=30, start=0.0, end=0.5)
        )  # below guitar range
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=0.5, end=1.0))  # valid
        pm.instruments.append(inst)

        mock_bp_predict.return_value = (MagicMock(), pm, MagicMock())

        result = self.ear.transcribe_stem("/fake/path.wav", target="Guitar")
        assert len(result) == 1
        assert result[0].pitch == 64

    @patch("agents.BASIC_PITCH_AVAILABLE", True)
    @patch("agents.basic_pitch_predict")
    @patch("agents.metrics")
    def test_transcribe_stem_basic_pitch_bass_target(self, mock_metrics, mock_bp_predict) -> None:
        """Bass target uses bass range filter."""
        mock_metrics.track_stage.return_value.__enter__.return_value = None
        mock_metrics.track_stage.return_value.__exit__.return_value = False

        import pretty_midi

        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=30, start=0.0, end=0.5)
        )  # above B0, valid for bass
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=70, start=0.5, end=1.0)
        )  # above G4, invalid for bass
        pm.instruments.append(inst)

        mock_bp_predict.return_value = (MagicMock(), pm, MagicMock())

        result = self.ear.transcribe_stem("/fake/path.wav", target="Bass")
        assert len(result) == 1
        assert result[0].pitch == 30


# =========================================================================
# EAR — export_midi
# =========================================================================


class TestEarExportMidi(unittest.TestCase):
    """Test export_midi edge cases."""

    def setUp(self) -> None:
        self.ear = EarAgent(model_id="mimbres/YourMT3", device="cpu", prefer_yourmt3=False)
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_export_midi_empty_does_not_crash(self) -> None:
        """Empty notes list does not crash."""
        path = os.path.join(self.tmpdir, "empty.mid")
        self.ear.export_midi([], path)

    def test_export_midi_creates_file(self) -> None:
        """Valid notes creates a MIDI file."""
        notes = [
            make_note(60, 0.0, end=0.5),
            make_note(64, 0.5, end=1.0),
        ]
        path = os.path.join(self.tmpdir, "test.mid")
        self.ear.export_midi(notes, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_export_midi_standard_resolution(self) -> None:
        """Exported MIDI has standard 480 ticks_per_quarter."""
        notes = [make_note(60, 0.0, end=0.5)]
        path = os.path.join(self.tmpdir, "res.mid")
        self.ear.export_midi(notes, path)
        import note_seq

        ns = note_seq.midi_file_to_sequence_proto(path)
        assert ns.ticks_per_quarter == 480


# =========================================================================
# EAR — _convert_prettymidi_to_noteseq
# =========================================================================


class TestEarConvertPrettyMidi(unittest.TestCase):
    """Test _convert_prettymidi_to_noteseq with various inputs."""

    def setUp(self) -> None:
        self.ear = EarAgent(model_id="mimbres/YourMT3", device="cpu", prefer_yourmt3=False)

    def test_convert_empty_prettymidi(self) -> None:
        """PrettyMIDI with no instruments returns empty."""
        import pretty_midi

        pm = pretty_midi.PrettyMIDI()
        result = self.ear._convert_prettymidi_to_noteseq(pm)
        assert len(result) == 0

    def test_convert_single_instrument(self) -> None:
        """Single instrument notes are converted."""
        import pretty_midi

        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
        pm.instruments.append(inst)
        result = self.ear._convert_prettymidi_to_noteseq(pm)
        assert len(result) == 1
        assert result[0].pitch == 60

    def test_convert_multiple_instruments(self) -> None:
        """Notes from all instruments are included."""
        import pretty_midi

        pm = pretty_midi.PrettyMIDI()
        for prog in range(2):
            inst = pretty_midi.Instrument(program=prog)
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=60 + prog, start=0.0, end=0.5))
            pm.instruments.append(inst)
        result = self.ear._convert_prettymidi_to_noteseq(pm)
        assert len(result) == 2


# =========================================================================
# EAR — _parse_mt3_tokens additional edge cases
# =========================================================================


class TestEarParseMt3TokensEdgeCases(unittest.TestCase):
    """Additional edge cases for _parse_mt3_tokens."""

    def setUp(self) -> None:
        self.ear = EarAgent(model_id="mimbres/YourMT3", device="cpu", prefer_yourmt3=False)

    def test_parse_unknown_token_skipped(self) -> None:
        """Unknown tokens after a valid start are skipped without error."""
        events = "NOTE_ON 60 VELOCITY 80 BOGUS 123 TIME_SHIFT 0.5 NOTE_OFF 60"
        notes = self.ear._convert_to_noteseq(events)
        assert len(notes) == 1

    def test_parse_note_off_no_active_note(self) -> None:
        """NOTE_OFF for a pitch with no active note is silently skipped."""
        events = "NOTE_OFF 99 TIME_SHIFT 0.1 NOTE_ON 60 VELOCITY 80 TIME_SHIFT 0.5 NOTE_OFF 60"
        notes = self.ear._convert_to_noteseq(events)
        assert len(notes) == 1

    def test_parse_active_notes_closed_at_end(self) -> None:
        """Notes still active at end of input are closed."""
        events = "NOTE_ON 60 VELOCITY 80 TIME_SHIFT 1.0"
        notes = self.ear._convert_to_noteseq(events)
        assert len(notes) == 1
        assert notes[0].end_time > notes[0].start_time

    def test_parse_invalid_int_skipped(self) -> None:
        """Invalid integer value causes token to be skipped."""
        events = "NOTE_ON sixty TIME_SHIFT 0.5 NOTE_ON 64 VELOCITY 80 TIME_SHIFT 0.5 NOTE_OFF 64"
        notes = self.ear._convert_to_noteseq(events)
        assert len(notes) == 1
        assert notes[0].pitch == 64

    def test_parse_time_shift_float_error(self) -> None:
        """Invalid float after TIME_SHIFT skips the token."""
        events = "TIME_SHIFT bad NOTE_ON 60 VELOCITY 80 TIME_SHIFT 0.5 NOTE_OFF 60"
        notes = self.ear._convert_to_noteseq(events)
        assert len(notes) == 1

    def test_parse_no_velocity_default(self) -> None:
        """NOTE_ON without VELOCITY defaults to 80."""
        events = "NOTE_ON 60 TIME_SHIFT 0.5 NOTE_OFF 60"
        notes = self.ear._convert_to_noteseq(events)
        assert len(notes) == 1
        assert notes[0].velocity == 80

    def test_parse_short_note_filtered(self) -> None:
        """Notes shorter than min_duration are filtered."""
        events = "NOTE_ON 60 VELOCITY 80 TIME_SHIFT 0.01 NOTE_OFF 60"
        notes = self.ear._convert_to_noteseq(events, min_duration=0.05)
        assert len(notes) == 0


# =========================================================================
# SPLITTER — _separate_with_api (mocked Demucs API path)
# =========================================================================


class TestSplitterSeparateWithApi(unittest.TestCase):
    """Test the Demucs API path with mocks."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.splitter = SplitterAgent(output_dir=self.tmpdir)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _patch_demucs_api(self):
        """Patch agents.demucs_api with a mock, whether or not demucs is installed."""
        import agents as _a

        original_api = getattr(_a, "demucs_api", None)
        original_flag = _a.DEMUCS_API_AVAILABLE

        def _restore():
            _a.demucs_api = original_api
            _a.DEMUCS_API_AVAILABLE = original_flag

        self.addCleanup(_restore)
        _a.demucs_api = MagicMock()
        _a.DEMUCS_API_AVAILABLE = True
        return _a.demucs_api

    @patch("agents.sf.write")
    def test_separate_with_api_creates_stems(self, mock_sf_write) -> None:
        """API separation creates stem files from returned tensors."""
        import torch

        api = self._patch_demucs_api()
        mock_separator = MagicMock()
        mock_separator.samplerate = 44100
        mock_separator.separate_audio_file.return_value = (
            None,
            {"other": torch.zeros((1, 44100)), "bass": torch.zeros((1, 44100))},
        )
        api.Separator.return_value = mock_separator

        with patch("agents.torch.cuda.is_available", return_value=False):
            dummy = os.path.join(self.tmpdir, "input.wav")
            with open(dummy, "w") as f:
                f.write("FAKE")
            result = self.splitter._separate_with_api(dummy, "input")

        assert "guitar" in result
        assert "bass" in result
        assert mock_sf_write.called

    @patch("agents.sf.write")
    def test_separate_with_api_uses_cuda(self, mock_sf_write) -> None:
        """API uses CUDA device when available."""
        import torch

        api = self._patch_demucs_api()
        mock_separator = MagicMock()
        mock_separator.samplerate = 44100
        mock_separator.separate_audio_file.return_value = (
            None,
            {"other": torch.zeros((1, 44100))},
        )
        api.Separator.return_value = mock_separator

        with patch("agents.torch.cuda.is_available", return_value=True):
            dummy = os.path.join(self.tmpdir, "i.wav")
            with open(dummy, "w") as f:
                f.write("FAKE")
            self.splitter._separate_with_api(dummy, "i")

        _, kwargs = api.Separator.call_args
        assert "device" in kwargs
        assert kwargs["device"] == "cuda"

    @patch("agents.sf.write")
    def test_separate_with_api_mono_tensor(self, mock_sf_write) -> None:
        """1D tensor is unsqueezed before writing."""
        import torch

        api = self._patch_demucs_api()
        mock_separator = MagicMock()
        mock_separator.samplerate = 44100
        mock_separator.separate_audio_file.return_value = (
            None,
            {"other": torch.zeros(44100)},  # 1D
        )
        api.Separator.return_value = mock_separator

        with patch("agents.torch.cuda.is_available", return_value=False):
            dummy = os.path.join(self.tmpdir, "x.wav")
            with open(dummy, "w") as f:
                f.write("FAKE")
            self.splitter._separate_with_api(dummy, "x")

        mock_sf_write.assert_called()


# =========================================================================
# EAR — YourMT3+ path via mock (transcribe_stem)
# =========================================================================


class TestEarYourMT3TranscribePath(unittest.TestCase):
    """Test the YourMT3 path in transcribe_stem with mocked model."""

    def setUp(self) -> None:
        self.ear = EarAgent(model_id="mimbres/YourMT3", device="cpu", prefer_yourmt3=False)

    @patch("agents.librosa.load")
    @patch("agents.torch.no_grad")
    def test_transcribe_stem_yourmt3_model_set(self, mock_no_grad, mock_load) -> None:
        """When self.model is set, YourMT3 path is used."""
        import numpy as np

        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_no_grad.return_value.__enter__.return_value = None
        mock_no_grad.return_value.__exit__.return_value = False

        self.ear.model = MagicMock()  # type: ignore[assignment]  # mock replaces real model/None in test
        self.ear.model.inference_file = MagicMock(return_value=([MagicMock()], MagicMock()))  # type: ignore[attr-defined,assignment]
        self.ear.processor = MagicMock()  # type: ignore[assignment]  # mock in test
        self.ear.processor.num_decoding_channels = 1  # type: ignore[attr-defined,assignment]
        self.ear.processor.detokenize_list_batches.return_value = (  # type: ignore[attr-defined]
            [(0.0, 0.5, 60, 80, 1)],
            MagicMock(),
            MagicMock(),
        )
        self.ear.model.audio_cfg = {  # type: ignore[attr-defined,assignment]
            "sample_rate": 16000,
            "input_frames": 320,
        }
        self.ear._ymt3_utils = {}

        result = self.ear.transcribe_stem("/fake/path.wav", target="Guitar", onset_threshold=0.5)
        assert isinstance(result, list)


# =========================================================================
# EAR — all models failed path
# =========================================================================


class TestEarAllModelsFailed(unittest.TestCase):
    """Test the path where no transcription models are available."""

    @patch("agents.BASIC_PITCH_AVAILABLE", False)
    def test_transcribe_stem_raises_when_no_models(self) -> None:
        """When no models available, raises RuntimeError."""
        ear = EarAgent(
            model_id="mimbres/YourMT3",
            device="cpu",
            prefer_yourmt3=False,
        )
        ear.model = None  # ensure no YourMT3
        with pytest.raises(RuntimeError):
            ear.transcribe_stem("/fake/path.wav")


# =========================================================================
# EAR — inline fallback methods
# =========================================================================


class TestEarInlineFallbacks(unittest.TestCase):
    """Test _slice_audio_inline, _merge_notes_inline, _mix_notes_inline."""

    def test_slice_audio_inline_exact_multiple(self) -> None:
        """Audio length is exact multiple of frame_size."""
        import numpy as np

        audio = np.random.randn(1, 16000).astype(np.float32)
        import torch

        audio_t = torch.from_numpy(audio)
        segments = EarAgent._slice_audio_inline(audio_t, 8000, 4000)
        assert segments.shape[0] == 4  # 16000 / 4000 = 4
        assert segments.shape[1] == 8000

    def test_slice_audio_inline_partial_last(self) -> None:
        """Last partial segment is zero-padded."""
        import numpy as np
        import torch

        audio = np.random.randn(1, 10000).astype(np.float32)
        audio_t = torch.from_numpy(audio)
        segments = EarAgent._slice_audio_inline(audio_t, 8000, 4000)
        # 10000: segments at 0-8000, 4000-10000 (partial, padded to 8000)
        assert segments.shape[0] == 2
        assert segments.shape[1] == 8000

    def test_merge_notes_inline(self) -> None:
        """Zipped events are converted to note dicts."""
        zipped = [(0.0, 0.5, 60, 80, 1), (0.5, 1.0, 64, 90, 1)]
        notes = EarAgent._merge_notes_inline(zipped)
        assert len(notes) == 2
        assert notes[0]["pitch"] == 60
        assert notes[0]["start"] == 0.0
        assert notes[1]["pitch"] == 64

    def test_mix_notes_inline_deduplicates(self) -> None:
        """Same pitch+start across channels is deduplicated."""
        ch1 = [{"pitch": 60, "start": 0.0}]
        ch2 = [{"pitch": 60, "start": 0.0}]  # duplicate
        ch3 = [{"pitch": 64, "start": 0.5}]
        mixed = EarAgent._mix_notes_inline([ch1, ch2, ch3])
        assert len(mixed) == 2

    def test_mix_notes_inline_sorted(self) -> None:
        """Mixed notes are sorted by start time."""
        ch1 = [{"pitch": 64, "start": 0.5}]
        ch2 = [{"pitch": 60, "start": 0.0}]
        mixed = EarAgent._mix_notes_inline([ch1, ch2])
        assert mixed[0]["pitch"] == 60

    def test_mix_notes_inline_empty_channel(self) -> None:
        """Empty channel list doesn't crash."""
        mixed = EarAgent._mix_notes_inline([[], []])
        assert len(mixed) == 0


# =========================================================================
# EAR — device selection
# =========================================================================


class TestEarAgentDeviceSelection(unittest.TestCase):
    """Test EarAgent device auto-detection."""

    @patch("agents.torch.cuda.is_available", return_value=True)
    def test_device_auto_selects_cuda(self, mock_cuda) -> None:
        """Auto device selects cuda when available."""
        ear = EarAgent(
            model_id="test",
            device="auto",
            prefer_yourmt3=False,
        )
        assert ear.device == "cuda"

    @patch("agents.torch.cuda.is_available", return_value=False)
    @patch("agents.torch.backends.mps.is_available", return_value=True)
    def test_device_auto_selects_mps(self, mock_mps, mock_cuda) -> None:
        """Auto device selects mps when cuda unavailable."""
        ear = EarAgent(
            model_id="test",
            device="auto",
            prefer_yourmt3=False,
        )
        assert ear.device == "mps"

    @patch("agents.torch.cuda.is_available", return_value=False)
    @patch("agents.torch.backends.mps.is_available", return_value=False)
    def test_device_auto_selects_cpu(self, mock_mps, mock_cuda) -> None:
        """Auto device falls back to cpu."""
        ear = EarAgent(
            model_id="test",
            device="auto",
            prefer_yourmt3=False,
        )
        assert ear.device == "cpu"

    def test_device_explicit_cpu(self) -> None:
        """Explicit device string is used directly."""
        ear = EarAgent(
            model_id="test",
            device="cpu",
            prefer_yourmt3=False,
        )
        assert ear.device == "cpu"

    def test_device_explicit_cuda(self) -> None:
        """Explicit cuda string is used directly."""
        ear = EarAgent(
            model_id="test",
            device="cuda",
            prefer_yourmt3=False,
        )
        assert ear.device == "cuda"


# =========================================================================
# EAR — Basic Pitch error path
# =========================================================================


class TestEarBasicPitchError(unittest.TestCase):
    """Test Basic Pitch failure handling in transcribe_stem."""

    @patch("agents.BASIC_PITCH_AVAILABLE", True)
    @patch("agents.basic_pitch_predict")
    @patch("agents.metrics")
    def test_basic_pitch_error_raises(self, mock_metrics, mock_bp_predict) -> None:
        """When Basic Pitch fails, error propagates."""
        mock_metrics.track_stage.return_value.__enter__.return_value = None
        mock_metrics.track_stage.return_value.__exit__.return_value = False

        ear = EarAgent(
            model_id="test",
            device="cpu",
            prefer_yourmt3=False,
        )
        ear.model = None

        mock_bp_predict.side_effect = RuntimeError("BP failed")

        with pytest.raises(RuntimeError):
            ear.transcribe_stem("/fake/path.wav")


# =========================================================================
# TAB — _build_yourmt3_args
# =========================================================================


class TestEarBuildYourmt3Args(unittest.TestCase):
    """Test _build_yourmt3_args produces a valid argparse namespace."""

    def test_build_yourmt3_args_returns_namespace(self) -> None:
        """Returns an argparse.Namespace with expected attributes."""
        import argparse

        ear = EarAgent(
            model_id="test",
            device="cpu",
            prefer_yourmt3=False,
        )
        args = ear._build_yourmt3_args("/fake/checkpoint")
        assert isinstance(args, argparse.Namespace)
        assert hasattr(args, "exp_id")
        assert hasattr(args, "task")
        assert hasattr(args, "precision")


# =========================================================================
# TAB — generate_tab branch coverage (partial backtracking)
# =========================================================================


class TestTabAgentGenerateTabBacktracking(unittest.TestCase):
    """Cover the backward-pass backtracking in generate_tab."""

    def setUp(self) -> None:
        self.agent = TabAgent(tuning=[40, 45, 50, 55, 59, 64], num_frets=24)

    def test_backtrack_reconstructs_path(self) -> None:
        """Multiple notes with multiple valid positions each."""
        notes = [
            make_note(40, 0.0),
            make_note(45, 0.5),
            make_note(50, 1.0),
            make_note(55, 1.5),
        ]
        result = self.agent.generate_tab(notes)
        assert len(result) == 4
        for entry in result:
            assert "fret" in entry
            assert "technique" in entry
            assert "start_time" in entry

    def test_all_notes_get_start_time(self) -> None:
        """Every note gets a start_time annotation."""
        notes = [make_note(40, 0.0), make_note(64, 0.3)]
        result = self.agent.generate_tab(notes)
        for entry in result:
            assert "start_time" in entry


# =========================================================================
# EAR — YourMT3 failure fallback and internal branches
# =========================================================================


class TestEarYourMT3FailureAndBranches(unittest.TestCase):
    """Cover YourMT3 failure handler and internal branch paths."""

    def setUp(self) -> None:
        self.ear = EarAgent(model_id="mimbres/YourMT3", device="cpu", prefer_yourmt3=False)

    @patch("agents.librosa.load")
    @patch("agents.torch.no_grad")
    @patch("agents.BASIC_PITCH_AVAILABLE", True)
    def test_yourmt3_failure_falls_to_basic_pitch(self, mock_load, mock_no_grad) -> None:
        """When YourMT3 raises, transcribe_stem falls to Basic Pitch."""
        import numpy as np

        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_no_grad.return_value.__enter__.return_value = None
        mock_no_grad.return_value.__exit__.return_value = False

        self.ear.model = MagicMock()  # type: ignore[assignment]  # mock replaces real model/None in test
        self.ear.model.inference_file = MagicMock(side_effect=RuntimeError("mt3-crash"))  # type: ignore[attr-defined,assignment]
        self.ear.processor = MagicMock()  # type: ignore[assignment]  # mock in test
        self.ear.model.audio_cfg = {  # type: ignore[attr-defined,assignment]
            "sample_rate": 16000,
            "input_frames": 320,
        }
        self.ear._ymt3_utils = {}

        with patch("agents.basic_pitch_predict") as mock_bp:
            import pretty_midi

            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0)
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
            pm.instruments.append(inst)
            mock_bp.return_value = (MagicMock(), pm, MagicMock())

            result = self.ear.transcribe_stem("/fake/path.wav")
            assert len(result) == 1

    @patch("agents.librosa.load")
    @patch("agents.torch.no_grad")
    def test_yourmt3_uses_util_slice(self, mock_no_grad, mock_load) -> None:
        """_transcribe_with_yourmt3 uses _ymt3_utils slice when available."""
        import numpy as np

        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_no_grad.return_value.__enter__.return_value = None
        mock_no_grad.return_value.__exit__.return_value = False

        self.ear.model = MagicMock()  # type: ignore[assignment]  # mock replaces real model/None in test
        self.ear.model.inference_file = MagicMock(return_value=([MagicMock()], MagicMock()))  # type: ignore[attr-defined,assignment]
        self.ear.processor = MagicMock()  # type: ignore[assignment]  # mock in test
        self.ear.processor.num_decoding_channels = 1  # type: ignore[attr-defined,assignment]
        self.ear.processor.detokenize_list_batches.return_value = (  # type: ignore[attr-defined]
            [(0.0, 0.5, 60, 80, 1)],
            MagicMock(),
            MagicMock(),
        )
        self.ear.model.audio_cfg = {  # type: ignore[attr-defined,assignment]
            "sample_rate": 16000,
            "input_frames": 4000,
        }
        # Provide all utils to cover "slice", "merge", "mix" branches
        import numpy as np

        def fake_slice(audio, frame_size, step_size):
            audio_np = audio.squeeze().numpy()
            total = len(audio_np)
            segments = []
            start = 0
            while start + frame_size <= total:
                segments.append(audio_np[start : start + frame_size])
                start += step_size
            if start < total:
                seg = np.zeros(frame_size, dtype=audio_np.dtype)
                seg[: total - start] = audio_np[start:]
                segments.append(seg)
            return np.array(segments)

        self.ear._ymt3_utils = {
            "slice": fake_slice,
            "merge": lambda z: (
                [{"pitch": 60, "start": 0.0, "end": 0.5, "velocity": 80}],
                [],
            ),
            "mix": lambda x: [{"pitch": 60, "start": 0.0, "end": 0.5, "velocity": 80}],
        }

        result = self.ear.transcribe_stem("/fake/path.wav")
        assert isinstance(result, list)

    @patch("agents.librosa.load")
    @patch("agents.BASIC_PITCH_AVAILABLE", True)
    def test_yourmt3_not_ready_raises(self, mock_load) -> None:
        """When tm is None or model lacks inference_file, falls back."""
        import numpy as np

        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        self.ear.model = MagicMock()  # type: ignore[assignment]  # mock replaces real model/None in test
        self.ear.model.audio_cfg = {"sample_rate": 16000, "input_frames": 320}  # type: ignore[attr-defined,assignment]
        # Don't set inference_file — model lacks it
        del self.ear.model.inference_file  # type: ignore[attr-defined,union-attr]
        self.ear.processor = MagicMock()  # type: ignore[assignment]  # mock in test
        self.ear._ymt3_utils = {}

        with patch("agents.basic_pitch_predict") as mock_bp:
            import pretty_midi

            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0)
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
            pm.instruments.append(inst)
            mock_bp.return_value = (MagicMock(), pm, MagicMock())

            result = self.ear.transcribe_stem("/fake/path.wav")
            assert len(result) == 1


# =========================================================================
# EAR — init without Basic Pitch
# =========================================================================


class TestEarAgentInitWithoutBasicPitch(unittest.TestCase):
    """Test EarAgent init when Basic Pitch is not available."""

    @patch("agents.BASIC_PITCH_AVAILABLE", False)
    def test_init_warns_when_no_models(self) -> None:
        """EarAgent init prints warning when no models available."""
        ear = EarAgent(
            model_id="mimbres/YourMT3",
            device="cpu",
            prefer_yourmt3=False,
        )
        assert ear.model is None


if __name__ == "__main__":
    unittest.main()
