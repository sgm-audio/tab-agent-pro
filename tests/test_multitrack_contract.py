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
            return {"lead": "lead", "left": "left", "right": "right"}

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
