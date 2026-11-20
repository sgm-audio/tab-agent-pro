"""Exercise real cleanup/tab/export with deterministic, stubbed inference boundaries."""

import json

import note_seq
import pretty_midi

import main
from agents import EarAgent


def test_all_instruments_export_correct_track_notes_and_tunings(tmp_path, monkeypatch):
    audio = tmp_path / "mix.wav"
    audio.touch()
    output = tmp_path / "out"
    pitches = {
        "Lead Guitar": [38, 42],
        "Rhythm Guitar L": [45, 47],
        "Rhythm Guitar R": [55, 57],
        "Bass": [23, 28],
    }
    seen = []

    class FakeSplitter:
        separation_mode = "demucs"

        def __init__(self, **kwargs):
            pass

        def separate_stems(self, path):
            return {"guitar": "guitar", "bass": "bass"}

        def process_guitars(self, path):
            # Says it carries real stereo information, so the L/R rhythm path
            # must be exercised. The mono path is covered separately.
            return {"lead": "lead", "left": "left", "right": "right", "stereo": True}

        def process_bass(self, path):
            return "clean_bass"

    class StubInferenceEar(EarAgent):
        def transcribe_stem(self, path, target, **kwargs):
            seen.append((path, target, kwargs))
            return [
                note_seq.NoteSequence.Note(
                    pitch=p, start_time=i * 0.6, end_time=i * 0.6 + 0.4, velocity=80
                )
                for i, p in enumerate(pitches[target])
            ]

    config = {
        "guitar_tuning": [38, 45, 50, 55, 59, 64],
        "bass_tuning": [23, 28, 33, 38, 43],
        "num_frets": 24,
    }
    monkeypatch.setattr(main, "load_user_memory", lambda: (str(tmp_path / "absent.json"), config))
    monkeypatch.setattr(main, "process_suno_audio", lambda path, **kw: (path, False, {}))
    monkeypatch.setattr(main, "SplitterAgent", FakeSplitter)
    monkeypatch.setattr(main, "EarAgent", StubInferenceEar)
    main.main(
        [
            str(audio),
            "--instrument",
            "All",
            "--output-dir",
            str(output),
            "--fresh-output-dir",
            "--onset",
            "0.62",
            "--frame",
            "0.31",
            "--device",
            "cpu",
        ]
    )

    assert [(p, t) for p, t, _ in seen] == [
        ("lead", "Lead Guitar"),
        ("left", "Rhythm Guitar L"),
        ("right", "Rhythm Guitar R"),
        ("clean_bass", "Bass"),
    ]
    for _, target, kw in seen:
        assert (kw["onset_threshold"], kw["frame_threshold"]) == (0.62, 0.31)
        assert kw["min_pitch"] == (23 if target == "Bass" else 38)
    for suffix, target in [
        ("lead_guitar", "Lead Guitar"),
        ("rhythm_L", "Rhythm Guitar L"),
        ("rhythm_R", "Rhythm Guitar R"),
        ("bass", "Bass"),
    ]:
        tuning = config["bass_tuning"] if target == "Bass" else config["guitar_tuning"]
        data = json.loads((output / f"mix_{suffix}.json").read_text())
        assert data["num_strings"] == len(tuning)
        assert [tuning[p["string"]] + p["fret"] for p in data["tablature"]] == pitches[target]
        assert (output / f"mix_{suffix}.tab").is_file()
        midi = pretty_midi.PrettyMIDI(str(output / f"mix_{suffix}.mid"))
        assert [n.pitch for inst in midi.instruments for n in inst.notes] == pitches[target]


def test_unplayable_chord_preserves_midi_and_reports_partial_tab(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "song.wav"
    audio.touch()
    output = tmp_path / "out"

    class FakeSplitter:
        separation_mode = "raw_fallback"

        def __init__(self, **kwargs):
            pass

        def separate_stems(self, path):
            return {"guitar": "guitar", "bass": "bass"}

        def process_bass(self, path):
            return "clean_bass"

    class StubInferenceEar(EarAgent):
        def transcribe_stem(self, path, target, **kwargs):
            return [
                note_seq.NoteSequence.Note(pitch=p, start_time=0.2, end_time=0.6, velocity=80)
                for p in (23, 24)
            ]

    monkeypatch.setattr(main, "load_user_memory", lambda: (str(tmp_path / "absent.json"), {}))
    monkeypatch.setattr(main, "process_suno_audio", lambda path, **kw: (path, False, {}))
    monkeypatch.setattr(main, "SplitterAgent", FakeSplitter)
    monkeypatch.setattr(main, "EarAgent", StubInferenceEar)
    main.main([str(audio), "--instrument", "Bass", "--output-dir", str(output), "--device", "cpu"])
    message = capsys.readouterr().err
    assert main.RAW_MIX_WARNING in message
    assert "1 detected note(s) omitted" in message and "fingering_conflict" in message
    assert "1 detected note(s) omitted" in (output / "song_bass.tab").read_text()
    data = json.loads((output / "song_bass.json").read_text())
    assert data["separation_mode"] == "raw_fallback"
    (placed,) = data["tablature"]
    (omitted,) = data["skipped_notes"]
    assert omitted["reason"] == "fingering_conflict"
    placed_pitch = data["tuning"][placed["string"]] + placed["fret"]
    assert sorted([placed_pitch, omitted["pitch"]]) == [23, 24]
    midi = pretty_midi.PrettyMIDI(str(output / "song_bass.mid"))
    assert [n.pitch for inst in midi.instruments for n in inst.notes] == [23, 24]


def test_mono_guitar_stem_emits_no_rhythm_tracks(tmp_path, monkeypatch, capsys):
    """A source with no stereo information must not become three tracks.

    End-to-end guard on the defect: the pipeline used to transpose whichever
    stem it was given under three names ("Lead", "Rhythm L", "Rhythm R"), so a
    mono recording produced three MIDI files and three tabs of the same notes.
    """
    audio = tmp_path / "mono.wav"
    audio.touch()
    output = tmp_path / "out"
    seen = []

    class MonoSplitter:
        separation_mode = "demucs"

        def __init__(self, **kwargs):
            pass

        def separate_stems(self, path):
            return {"guitar": "guitar", "bass": "bass"}

        def process_guitars(self, path):
            return {"lead": "lead", "left": None, "right": None, "stereo": False}

        def process_bass(self, path):
            return "clean_bass"

    class StubInferenceEar(EarAgent):
        def transcribe_stem(self, path, target, **kwargs):
            seen.append((path, target))
            return [
                note_seq.NoteSequence.Note(
                    pitch=p, start_time=i * 0.6, end_time=i * 0.6 + 0.4, velocity=80
                )
                for i, p in enumerate([40, 45])
            ]

    config = {
        "guitar_tuning": [40, 45, 50, 55, 59, 64],
        "bass_tuning": [23, 28, 33, 38, 43],
        "num_frets": 24,
    }
    monkeypatch.setattr(main, "load_user_memory", lambda: (str(tmp_path / "absent.json"), config))
    monkeypatch.setattr(main, "process_suno_audio", lambda path, **kw: (path, False, {}))
    monkeypatch.setattr(main, "SplitterAgent", MonoSplitter)
    monkeypatch.setattr(main, "EarAgent", StubInferenceEar)

    main.main(
        [str(audio), "--instrument", "Guitar", "--output-dir", str(output), "--device", "cpu"]
    )

    # Only the lead stem is transcribed; no rhythm transcription was attempted.
    assert seen == [("lead", "Lead Guitar")]
    assert "stereo information" in capsys.readouterr().err
    assert (output / "mono_lead_guitar.mid").is_file()
    assert not (output / "mono_rhythm_L.mid").exists()
    assert not (output / "mono_rhythm_R.mid").exists()
    assert not (output / "mono_rhythm_L.tab").exists()
    assert not (output / "mono_rhythm_R.tab").exists()
