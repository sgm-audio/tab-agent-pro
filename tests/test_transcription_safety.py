"""Safety regressions with independent oracles and synthetic fixtures."""

import itertools
import json
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


@pytest.mark.parametrize(
    "start,end",
    [(-0.1, 0.4), (0.2, 0.2), (0.2, 0.1), (0.2, None), (float("nan"), 0.5), (0.1, float("inf"))],
    ids=["negative", "zero-duration", "reversed", "missing-end", "nan-onset", "infinite-release"],
)
def test_invalid_note_timing_is_reported_without_discarding_valid_notes(start, end):
    agent = TabAgent(GUITAR, 24)
    invalid = SimpleNamespace(pitch=40, start_time=start, end_time=end)
    tab = agent.generate_tab([invalid, note(45, 0.8, 0.2)])
    (placed,) = tab
    assert GUITAR[placed["string"]] + placed["fret"] == 45
    (omitted,) = agent.last_skipped_events
    assert omitted["pitch"] == 40
    assert omitted["reason"] == "invalid_timing" and agent.last_skipped_notes == [40]
    # Non-finite times become null, not JSON NaN / Infinity.
    json.dumps(omitted, allow_nan=False)
    assert agent.generate_tab([note(45, 0.8, 0.2)]) == tab
    assert agent.last_skipped_events == [] and agent.last_skipped_notes == []


def test_cleaner_drops_invalid_times_before_midi_export():
    ear = EarAgent(device="cpu")
    valid = note(42, 0.2)
    invalid = [
        note(44, float("nan")),
        note(44, 0.2, float("inf")),
        note(44, -0.1),
        note(44, 0.3, -0.1),
    ]
    assert ear.humanize_and_clean([*invalid, valid]) == [valid]


@pytest.mark.parametrize(
    "pitches", [[40, 52, 59], [76, 88, 90]], ids=["octave-chord", "high-notes"]
)
def test_suno_cleanup_preserves_octaves_and_high_notes(pitches):
    import note_seq

    from suno_postprocessor import SunoNotePostprocessor

    chord = [
        note_seq.NoteSequence.Note(pitch=p, start_time=0, end_time=0.5, velocity=80)
        for p in pitches
    ]
    result = SunoNotePostprocessor().process(chord, is_suno=True, metrics={})
    assert sorted(n.pitch for n in result) == pitches


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

    # Assert routing by content, not by waveform. The previous version of this
    # test pinned `channel - 0.8*mid`, which is the *centre* signal at -14 dB
    # with opposite polarity in each stem: it was asserting the bug.
    def magnitude(signal, freq):
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1 / sr)
        return float(spectrum[np.argmin(np.abs(freqs - freq))])

    assert result["stereo"] is True
    lead, rate = sf.read(result["lead"])
    left, _ = sf.read(result["left"])
    right, _ = sf.read(result["right"])
    assert rate == sr
    for stem in (lead, left, right):
        assert len(stem) == len(original)

    # 440 Hz is centred, 330 Hz is left-only, 550 Hz is right-only.
    assert magnitude(lead, 440) > 100 * max(magnitude(lead, 330), magnitude(lead, 550))
    assert magnitude(left, 330) > 100 * max(magnitude(left, 440), magnitude(left, 550))
    assert magnitude(right, 550) > 100 * max(magnitude(right, 440), magnitude(right, 330))
    # And the two rhythm stems are genuinely different signals, not the same
    # one inverted (their correlation must not be -1).
    correlation = float(np.corrcoef(left, right)[0, 1])
    assert abs(correlation) < 0.9, f"rhythm L/R are the same signal: r={correlation:.3f}"


def _naive_octave_cleanup(notes):
    """The all-pairs reference the bucketed scan must match exactly."""
    cleaned = []
    skip_indices: set[int] = set()
    for i, note in enumerate(notes):
        if i in skip_indices:
            continue
        is_upper_octave = False
        for j, other in enumerate(notes):
            if i == j or j in skip_indices:
                continue
            if abs(other.start_time - note.start_time) < 0.05:
                if note.pitch - other.pitch == 12:
                    is_upper_octave = True
                    break
                if other.pitch - note.pitch == 12:
                    skip_indices.add(j)
        if not is_upper_octave:
            cleaned.append(note)
    return cleaned


def test_octave_cleanup_matches_the_all_pairs_reference() -> None:
    """Bucketing by onset must not change which notes are dropped."""
    import random

    import note_seq

    from suno_postprocessor import SunoNotePostprocessor

    rng = random.Random(20261118)
    processor = SunoNotePostprocessor()
    for _trial in range(40):
        notes = [
            note_seq.NoteSequence.Note(
                pitch=rng.randint(36, 84),
                start_time=round(rng.uniform(0.0, 1.0), 3),
                end_time=0.0,
            )
            for _ in range(rng.randint(1, 40))
        ]
        for note in notes:
            note.end_time = note.start_time + 0.2
        expected = [n.pitch for n in _naive_octave_cleanup(notes)]
        actual = [n.pitch for n in processor._remove_octave_errors(notes)]
        assert actual == expected


def test_octave_cleanup_is_not_quadratic() -> None:
    """The old all-pairs scan took ~6 s for 8000 notes and scaled 4x per doubling."""
    import time

    import note_seq

    from suno_postprocessor import SunoNotePostprocessor

    processor = SunoNotePostprocessor()
    notes = [
        note_seq.NoteSequence.Note(pitch=40 + (i % 24), start_time=i * 0.01, end_time=0.0)
        for i in range(8000)
    ]
    for note in notes:
        note.end_time = note.start_time + 0.05
    started = time.perf_counter()
    processor._remove_octave_errors(notes)
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0, f"octave cleanup took {elapsed:.2f}s for 8000 notes"
