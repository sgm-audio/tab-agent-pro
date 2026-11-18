"""Offline CLI integration. Only the Demucs and Basic Pitch boundaries are faked."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
import torch

import agents
import init_memory
import main
from monitoring import HealthTracker, PipelineMetrics
from tests.audio_fixtures import BASS_5, DROP_D, SAMPLE_RATE, plucks, tab_pitches


def pm_notes(events):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(0)
    instrument.notes = [pretty_midi.Note(80, pitch, start, end) for pitch, start, end in events]
    midi.instruments.append(instrument)
    return None, midi, []  # predict()'s return shape


@pytest.fixture
def stage_state(monkeypatch):
    monitor, stats = HealthTracker(), PipelineMetrics()
    monkeypatch.setattr(agents, "health", monitor)
    monkeypatch.setattr(agents, "metrics", stats)
    return monitor, stats


def test_cli_all_writes_distinct_lead_rhythm_bass_note_content(
    tmp_path, monkeypatch, stage_state, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_memory, "get_memory_dir", lambda: str(tmp_path / "memory"))
    # Mix with distinct notes, independent attacks, overlapping energy, and a
    # known wide dyad on lead, which has no option of sharing one string.
    lead = [(38, 0.20, 0.62), (69, 0.20, 0.62)]
    left = [(52, 0.45, 0.96)]
    right = [(65, 0.70, 1.10)]
    bass = [(23, 0.28, 0.78), (31, 1.30, 1.80)]
    center_pcm, left_pcm, right_pcm, bass_pcm = (
        plucks(track) for track in (lead, left, right, bass)
    )
    t = np.arange(len(bass_pcm)) / SAMPLE_RATE
    buzz = 0.06 * np.sin(2 * np.pi * 1600 * t)
    guitar_stem = np.column_stack((center_pcm + left_pcm, center_pcm + right_pcm))
    bass_stem = np.column_stack((bass_pcm + buzz, bass_pcm + buzz))
    mix = tmp_path / "spatial rehearsal.take.wav"
    sf.write(mix, guitar_stem + bass_stem, SAMPLE_RATE)

    def separate(path):
        assert Path(path) == mix
        return None, {
            "other": torch.from_numpy(guitar_stem.T),
            "bass": torch.from_numpy(bass_stem.T),
        }

    def separator(**kwargs):
        assert kwargs["model"] == "htdemucs" and kwargs["device"] == "cpu"
        return SimpleNamespace(samplerate=SAMPLE_RATE, separate_audio_file=separate)

    monkeypatch.setattr(agents, "DEMUCS_API_AVAILABLE", True)
    monkeypatch.setattr(agents, "demucs_api", SimpleNamespace(Separator=separator), raising=False)
    expected = {
        "processed_lead": (lead, center_pcm + (left_pcm + right_pcm) / 2),
        "processed_rhythm_L": (left, 0.2 * center_pcm + 0.6 * left_pcm - 0.4 * right_pcm),
        "processed_rhythm_R": (right, 0.2 * center_pcm - 0.4 * left_pcm + 0.6 * right_pcm),
        "processed_bass_clean": (bass, None),
    }
    calls = []

    def predict(path, **kwargs):
        stem = Path(path).stem
        calls.append(stem)
        assert kwargs["onset_threshold"] == 0.62
        assert kwargs["frame_threshold"] == 0.3  # profile, not an explicit CLI flag
        pcm, rate = sf.read(path)
        events, guitar_pcm = expected[stem]
        assert rate == SAMPLE_RATE and pcm.shape == center_pcm.shape
        if guitar_pcm is not None:
            np.testing.assert_allclose(pcm, guitar_pcm, atol=9e-5)
        else:
            # Bass fundamentals are kept; fret buzz above the cutoff is halved.
            assert abs(np.dot(pcm, buzz) / np.dot(buzz, buzz) - 0.5) < 0.03
            assert np.dot(pcm, bass_pcm) / np.dot(bass_pcm, bass_pcm) > 0.95
        return pm_notes(events)

    monkeypatch.setattr(agents, "BASIC_PITCH_AVAILABLE", True)
    monkeypatch.setattr(agents, "basic_pitch_predict", predict)
    output = tmp_path / "all tracks"
    main.main(
        [
            str(mix),
            "--instrument",
            "All",
            "--reaper-report-midi",
            "--device",
            "cpu",
            "--profile",
            "rock_drop_d",
            "--onset",
            "0.62",
            "--output-dir",
            str(output),
        ]
    )
    assert sorted(calls) == sorted(expected)
    assert stage_state[0].as_dict()["components"]["demucs"] == "loaded"
    tracks = {
        "lead_guitar": ("Lead Guitar", lead, DROP_D),
        "rhythm_L": ("Rhythm Guitar L", left, DROP_D),
        "rhythm_R": ("Rhythm Guitar R", right, DROP_D),
        "bass": ("5-String Bass", bass, BASS_5),
    }
    reported = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("[tab-agent-midi]")
    ]
    assert sorted(reported) == sorted(f"[tab-agent-midi] {suffix}" for suffix in tracks)
    assert {p.name for p in output.iterdir() if p.suffix in {".json", ".mid", ".tab"}} == {
        f"{mix.stem}_{track}.{kind}" for track in tracks for kind in ("json", "mid", "tab")
    }
    for suffix, (label, events, tuning) in tracks.items():
        prefix = f"{mix.stem}_{suffix}"
        midi = pretty_midi.PrettyMIDI(str(output / f"{prefix}.mid"))
        midi_notes = sorted((n.pitch, n.start, n.end) for i in midi.instruments for n in i.notes)
        for actual, expected_note in zip(midi_notes, sorted(events), strict=True):
            assert actual == pytest.approx(expected_note, abs=0.003)

        data = json.loads((output / f"{prefix}.json").read_text())
        tab = data["tablature"]
        assert data["instrument"] == label and data["num_strings"] == len(tuning)
        assert data["tuning"] == tuning
        assert data["separation_mode"] == "demucs" and data["skipped_notes"] == []
        pitches = tab_pitches(tab, tuning, 24)
        assert sorted(
            (pitch, p["start_time"], p["end_time"]) for pitch, p in zip(pitches, tab, strict=True)
        ) == sorted(events)

        tab_lines = (output / f"{prefix}.tab").read_text().splitlines()
        onset_line = next(line for line in tab_lines if line.startswith("t(s)|"))
        assert onset_line.split("|", 1)[1].split() == sorted(
            {f"{start:.3f}" for _, start, _ in events}
        )
        rows = [
            line.strip().split("|", 1) for line in tab_lines if "|" in line and line != onset_line
        ]
        assert [name for name, _ in rows] == (
            ["G", "D", "A", "E", "B"] if suffix == "bass" else ["E", "B", "G", "D", "A", "D"]
        )
        columns = list(zip(*(row.split() for _, row in rows), strict=True))
        rendered_pitches = [
            sorted(
                tuning[-1 - i] + int(cell.rstrip("shp"))
                for i, cell in enumerate(col)
                if cell != "-"
            )
            for col in columns
        ]
        assert rendered_pitches == [
            sorted(pitch for pitch, start, _ in events if start == onset)
            for onset in sorted({start for _, start, _ in events})
        ]


@pytest.mark.parametrize("cli_artifact", ["absent", "corrupt", "empty"])
def test_incomplete_demucs_outputs_fall_back_to_readable_mix(
    tmp_path, monkeypatch, stage_state, capsys, cli_artifact
):
    source = tmp_path / "offline session.wav"
    left = plucks([(43, 0.2, 0.6)])
    right = plucks([(64, 0.5, 1.1)])
    sf.write(source, np.column_stack((left, right)), SAMPLE_RATE)
    work = tmp_path / "stems"
    # A previous run has a complete pair. Neither today's missing API bass
    # nor today's zero-exit CLI should be allowed to use it.
    old = work / "htdemucs" / source.name
    old.mkdir(parents=True)
    for name in ("bass", "other"):
        sf.write(old / f"{name}.wav", np.zeros(128), SAMPLE_RATE)

    api = SimpleNamespace(
        samplerate=SAMPLE_RATE,
        separate_audio_file=lambda _: (None, {"other": torch.zeros((2, 128))}),
    )
    monkeypatch.setattr(agents, "DEMUCS_API_AVAILABLE", True)
    monkeypatch.setattr(
        agents, "demucs_api", SimpleNamespace(Separator=lambda **_: api), raising=False
    )

    def cli(command, **_):
        workdir = Path(command[command.index("-o") + 1])
        folder = workdir / "htdemucs" / source.stem
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / "other.wav", np.zeros(128), SAMPLE_RATE)
        if cli_artifact == "corrupt":
            (folder / "bass.wav").write_bytes(b"not audio")
        elif cli_artifact == "empty":
            sf.write(folder / "bass.wav", np.zeros(0), SAMPLE_RATE)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(agents.subprocess, "run", cli)
    splitter = agents.SplitterAgent(str(work), device="cpu")
    stems = splitter.separate_stems(str(source))

    assert splitter.separation_mode == "raw_fallback"
    assert set(stems) == {"guitar", "bass"}
    assert len(set(stems.values())) == 2
    assert all(Path(path).read_bytes() == source.read_bytes() for path in stems.values())
    processed = [
        *splitter.process_guitars(stems["guitar"]).values(),
        splitter.process_bass(stems["bass"]),
    ]
    assert all(sf.info(path).frames == len(left) for path in processed)
    monitor, stats = stage_state
    assert monitor.as_dict()["components"]["demucs"] == "unavailable"
    assert stats.stage_errors["demucs_separation"] == stats.stage_errors["demucs_cli"] == 1
    assert "demucs_raw_fallback" in capsys.readouterr().out
