"""Safety regressions with independent oracles; no inference models are mocked as real.

Strict xfails document known product defects, not acceptable output.  An XPASS
fails CI so the marker must be removed when the underlying defect is fixed.
"""

import itertools
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

import agents
from agents import EarAgent, SplitterAgent, TabAgent

GUITAR = [40, 45, 50, 55, 59, 64]


def note(pitch, start, duration=0.4):
    return SimpleNamespace(pitch=pitch, start_time=start, end_time=start + duration, velocity=80)


@pytest.mark.parametrize(
    "tuning,frets,pitches,times",
    [
        (GUITAR, 24, [45, 50, 52, 55], [0.0, 0.12, 0.8, 1.1]),
        ([38, 45, 50, 55, 59, 64], 19, [38, 45, 50], [0.0, 0.3, 0.6]),
        ([23, 28, 33, 38, 43], 24, [28, 35, 40], [0.0, 0.1, 0.9]),
    ],
)
def test_viterbi_matches_exhaustive_monophonic_minimum(tuning, frets, pitches, times):
    agent = TabAgent(tuning, frets)
    # Keep events disjoint: this oracle scores the monophonic transition cost.
    result = agent.generate_tab([note(p, t, 0.08) for p, t in zip(pitches, times, strict=True)])
    assert len(result) == len(pitches)
    for p, t, pos in zip(pitches, times, result, strict=True):
        assert 0 <= pos["string"] < len(tuning)
        assert 0 <= pos["fret"] <= frets
        assert tuning[pos["string"]] + pos["fret"] == p
        assert pos["start_time"] == pytest.approx(t)

    def path_cost(path):
        return sum(
            agent.calculate_cost(path[i - 1], path[i], times[i] - times[i - 1])
            for i in range(1, len(path))
        )

    choices = [agent.get_valid_positions(p) for p in pitches]
    assert path_cost(result) == pytest.approx(min(map(path_cost, itertools.product(*choices))))


def test_skipped_notes_preserve_pitch_order_and_reset():
    agent = TabAgent(GUITAR, 24)
    notes = [note(20, 0), note(40, 0.5), note(100, 1), note(45, 1.5), note(20, 2)]
    tab = agent.generate_tab(notes)
    assert agent.last_skipped_notes == [20, 100, 20]
    assert [GUITAR[p["string"]] + p["fret"] for p in tab] == [40, 45]
    assert [p["start_time"] for p in tab] == [0.5, 1.5]
    assert agent.generate_tab([]) == []
    assert agent.last_skipped_notes == []


def test_demucs_double_failure_copies_actual_pcm_and_marks_unavailable(tmp_path, monkeypatch):
    sr = 8000
    t = np.arange(sr) / sr
    audio = np.column_stack((0.15 * np.sin(2 * np.pi * 110 * t), 0.1 * np.sin(2 * np.pi * 330 * t)))
    source = tmp_path / "mix.wav"
    sf.write(source, audio, sr, subtype="PCM_16")
    splitter = SplitterAgent(str(tmp_path / "stems"), device="cpu")
    monkeypatch.setattr(agents, "DEMUCS_API_AVAILABLE", True)
    api = Mock(side_effect=RuntimeError("missing weights"))
    cli = Mock(side_effect=FileNotFoundError("demucs"))
    health = Mock()
    monkeypatch.setattr(splitter, "_separate_with_api", api)
    monkeypatch.setattr(splitter, "_separate_with_subprocess", cli)
    monkeypatch.setattr(agents, "health", health)
    stems = splitter.separate_stems(str(source))
    api.assert_called_once()
    cli.assert_called_once()
    health.set_component.assert_called_with("demucs", "unavailable")
    assert set(stems) == {"guitar", "bass"}
    for path in stems.values():
        data, actual_sr = sf.read(path, always_2d=True)
        reference, _ = sf.read(source, always_2d=True)
        assert actual_sr == sr
        np.testing.assert_array_equal(data, reference)


def test_demucs_missing_api_stem_must_not_report_success(tmp_path, monkeypatch):
    import torch

    separator = Mock(samplerate=8000)
    separator.separate_audio_file.return_value = (None, {"other": torch.zeros(1, 8000)})
    monkeypatch.setattr(
        agents, "demucs_api", SimpleNamespace(Separator=Mock(return_value=separator)), raising=False
    )
    splitter = SplitterAgent(str(tmp_path), device="cpu")
    with pytest.raises(ValueError, match=r"did not return required stems: \['bass'\]"):
        splitter._separate_with_api("mix.wav", "mix")
    separator.separate_audio_file.assert_called_once_with("mix.wav")


def test_basic_pitch_failure_updates_health_and_does_not_return_empty_success(monkeypatch):
    ear = EarAgent(device="cpu")
    health = Mock()
    monkeypatch.setattr(agents, "health", health)
    monkeypatch.setattr(agents, "BASIC_PITCH_AVAILABLE", True)
    missing = ModuleNotFoundError("missing", name="mir_eval")
    monkeypatch.setattr(agents, "basic_pitch_predict", Mock(side_effect=missing))
    with pytest.raises(RuntimeError, match="Basic Pitch dependency missing: mir_eval") as error:
        ear.transcribe_stem("bad.wav")
    assert error.value.__cause__ is missing
    health.set_component.assert_called_with("basic_pitch", "error: missing mir_eval")
    health.record_request.assert_called_once_with(success=False, details="transcribe: bad.wav")


def test_cleaner_preserves_distinct_simultaneous_chord_pitches():
    ear = EarAgent(device="cpu")
    chord = [note(40, 0), note(47, 0), note(52, 0), note(52, 0)]
    assert [n.pitch for n in ear.humanize_and_clean(chord)] == [40, 47, 52]


class UpperOctaveDroppedError(AssertionError):
    """Suno cleanup removed only the upper member of a real octave pair."""


def test_suno_cleanup_preserves_legitimate_octave_chord():
    import note_seq

    from suno_postprocessor import SunoNotePostprocessor

    chord = [
        note_seq.NoteSequence.Note(pitch=p, start_time=0, end_time=0.5, velocity=80)
        for p in (40, 52, 59)
    ]
    result = SunoNotePostprocessor().process(chord, is_suno=True, metrics={})
    pitches = sorted(n.pitch for n in result)
    if pitches == [40, 59]:
        raise UpperOctaveDroppedError(pitches)
    assert pitches == [40, 52, 59]


def test_stereo_mid_side_routing_with_distinct_tones(tmp_path):
    sr = 8000
    t = np.arange(sr) / sr
    center = 0.08 * np.sin(2 * np.pi * 440 * t)
    left_part = 0.05 * np.sin(2 * np.pi * 330 * t)
    right_part = 0.04 * np.sin(2 * np.pi * 550 * t)
    source = tmp_path / "stereo.wav"
    sf.write(
        source, np.column_stack((center + left_part, center + right_part)), sr, subtype="FLOAT"
    )
    result = SplitterAgent(str(tmp_path / "out"), device="cpu").process_guitars(str(source))
    original, _ = sf.read(source, always_2d=True)
    mid = (original[:, 0] + original[:, 1]) / 2
    for key, expected in (
        ("lead", mid),
        ("left", original[:, 0] - 0.8 * mid),
        ("right", original[:, 1] - 0.8 * mid),
    ):
        data, rate = sf.read(result[key])
        assert rate == sr
        np.testing.assert_allclose(data, expected, atol=4e-5)


def test_suno_processing_preserves_high_notes_and_octaves():
    import note_seq

    from suno_postprocessor import SunoNotePostprocessor

    notes = [
        note_seq.NoteSequence.Note(pitch=p, start_time=0, end_time=0.5, velocity=80)
        for p in (76, 88, 90)
    ]
    processed = SunoNotePostprocessor().process(notes, is_suno=True, metrics={})
    assert [n.pitch for n in processed] == [76, 88, 90]
