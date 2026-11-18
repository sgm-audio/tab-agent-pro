"""Real inference: the Basic Pitch CPU tests are required, not import smokes."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

import agents
import device
import init_memory
import main
from agents import EarAgent, SplitterAgent
from monitoring import HealthTracker, PipelineMetrics
from tests.audio_fixtures import BASS_5, SAMPLE_RATE, plucks, tab_pitches


@pytest.fixture
def cpu_model(monkeypatch):
    import onnxruntime as ort

    assert agents.BASIC_PITCH_AVAILABLE, "Install the project with Basic Pitch and ONNX Runtime"
    model = agents.BASIC_PITCH_MODEL
    assert model and model.suffix == ".onnx" and model.is_file(), "Bundled ONNX weights are missing"
    assert "CPUExecutionProvider" in ort.get_available_providers()
    real_session = ort.InferenceSession
    sessions = []

    def observe(path, *, providers):
        # Limit worker creation on small CI runners, but construct and run the real model.
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = real_session(path, sess_options=options, providers=providers)
        sessions.append((Path(path), session.get_providers()))
        return session

    monkeypatch.setattr(ort, "InferenceSession", observe)
    monkeypatch.setattr(agents, "health", HealthTracker())
    monkeypatch.setattr(agents, "metrics", PipelineMetrics())
    return sessions, model


@pytest.mark.basic_pitch_model
def test_basic_pitch_cpu_recovers_plucked_chord_and_later_note(tmp_path, cpu_model):
    truth = [(57, 0.22, 0.78), (61, 0.22, 0.78), (64, 0.22, 0.78), (69, 1.18, 1.72)]
    audio = tmp_path / "plucked chord.wav"
    sf.write(audio, plucks(truth), SAMPLE_RATE)

    ear = EarAgent(device="cpu")
    notes = ear.transcribe_stem(str(audio))
    sessions, model = cpu_model
    assert sessions == [(model, ["CPUExecutionProvider"])]
    assert agents.health.as_dict()["components"]["basic_pitch"] == "loaded"
    assert all(0 <= n.start_time < n.end_time <= 2.5 for n in notes)

    # Pitch- and onset-matched, one-to-one; permit a small harmonic false-positive budget.
    unmatched = list(notes)
    for pitch, onset, _ in truth:
        matches = [n for n in unmatched if n.pitch == pitch and abs(n.start_time - onset) <= 0.09]
        assert matches, f"missing pitch {pitch} near {onset}s from {notes!r}"
        unmatched.remove(min(matches, key=lambda n: abs(n.start_time - onset)))
    assert len(unmatched) <= 3, unmatched


@pytest.mark.basic_pitch_model
def test_real_basic_pitch_cpu_transcribes_known_tone(tmp_path, cpu_model):
    sr = SAMPLE_RATE
    t = np.arange(sr * 2) / sr
    # Sustained harmonic A4 with attack/release, unlike the plucked fixture above.
    envelope = np.minimum(
        1, np.minimum(np.maximum(0, (t - 0.2) / 0.05), np.maximum(0, (1.7 - t) / 0.1))
    )
    signal = envelope * (0.35 * np.sin(2 * np.pi * 440 * t) + 0.12 * np.sin(2 * np.pi * 880 * t))
    path = tmp_path / "a4.wav"
    sf.write(path, signal, sr)

    notes = EarAgent(device="cpu").transcribe_stem(str(path))
    sessions, model = cpu_model
    assert sessions == [(model, ["CPUExecutionProvider"])]
    assert any(n.pitch == 69 and 0 <= n.start_time < n.end_time <= 2 for n in notes), [
        (n.pitch, n.start_time, n.end_time) for n in notes
    ]


@pytest.mark.basic_pitch_model
@pytest.mark.parametrize("audio_case", ["silence", "percussion"])
def test_basic_pitch_cpu_unpitched_audio_is_empty_not_backend_failure(
    tmp_path, cpu_model, audio_case
):
    sr = SAMPLE_RATE
    audio = tmp_path / f"{audio_case}.wav"
    signal = np.zeros(2 * sr)
    if audio_case == "percussion":
        rng = np.random.default_rng(1729)
        length = round(0.055 * sr)
        for onset in (0.25, 0.52, 0.76, 1.10, 1.34, 1.58):
            i = round(onset * sr)
            burst = np.clip(rng.normal(0, 0.30, length), -0.25, 0.25)
            signal[i : i + length] = burst * np.exp(-np.arange(length) / (sr * 0.012))
    sf.write(audio, signal, sr)
    ear = EarAgent(device="cpu")
    assert ear.transcribe_stem(str(audio)) == []
    sessions, model = cpu_model
    assert sessions == [(model, ["CPUExecutionProvider"])]
    assert agents.metrics.stage_errors.get("basic_pitch_transcription", 0) == 0


@pytest.mark.basic_pitch_model
@pytest.mark.parametrize("weights", ["missing", "corrupt"])
def test_basic_pitch_bad_weights_fail_as_runtime_error(tmp_path, cpu_model, monkeypatch, weights):
    audio = tmp_path / "valid.wav"
    sf.write(audio, plucks([(64, 0.22, 0.78)]), SAMPLE_RATE)
    broken = tmp_path / "broken.onnx"
    if weights == "corrupt":
        broken.write_bytes(b"truncated graph")
    monkeypatch.setattr(agents, "BASIC_PITCH_MODEL", broken)

    ear = EarAgent(device="cpu")
    with pytest.raises(RuntimeError, match="Check the audio, ONNX model and runtime") as failure:
        ear.transcribe_stem(str(audio))
    assert isinstance(failure.value.__cause__, ValueError)  # actual Basic Pitch model loader
    assert agents.health.as_dict()["components"]["basic_pitch"] == "error: ValueError"
    assert agents.metrics.stage_errors["basic_pitch_transcription"] == 1


@pytest.mark.basic_pitch_model
def test_cli_with_missing_gpu_and_demucs_runs_real_cpu_inference(tmp_path, cpu_model, monkeypatch):
    # Hardware and the failed separator are the only injected dependencies.
    # Run the real EarAgent and check the bytes downstream of ONNX all the way to disk.
    monkeypatch.setattr(device, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(agents, "DEMUCS_API_AVAILABLE", False)
    monkeypatch.setattr(init_memory, "get_memory_dir", lambda: str(tmp_path / "memory"))

    def no_cli(*args, **kwargs):
        raise FileNotFoundError("Demucs executable unavailable in this scenario")

    monkeypatch.setattr(agents.SplitterAgent, "_separate_with_subprocess", no_cli)
    truth = [(28, 0.22, 0.78), (35, 1.18, 1.72)]
    audio = tmp_path / "bass.wav"
    sf.write(audio, plucks(truth), SAMPLE_RATE)
    out = tmp_path / "out"
    main.main([str(audio), "--device", "cuda", "--instrument", "Bass", "--output-dir", str(out)])

    sessions, model = cpu_model
    assert sessions == [(model, ["CPUExecutionProvider"])]
    assert agents.health.as_dict()["components"]["demucs"] == "unavailable"
    midi = pretty_midi.PrettyMIDI(str(out / "bass_bass.mid"))
    midi_notes = [(n.pitch, n.start) for track in midi.instruments for n in track.notes]
    tab = json.loads((out / "bass_bass.json").read_text())["tablature"]
    pitches = tab_pitches(tab, BASS_5, 24)
    tab_notes = list(zip(pitches, (p["start_time"] for p in tab), strict=True))
    for expected_pitch, expected_start, _ in truth:
        assert any(p == expected_pitch and abs(t - expected_start) < 0.12 for p, t in midi_notes)
        assert any(p == expected_pitch and abs(t - expected_start) < 0.12 for p, t in tab_notes)
    for tab_note, midi_note in zip(sorted(tab_notes), sorted(midi_notes), strict=True):
        assert tab_note == pytest.approx(midi_note, abs=0.003)


@pytest.mark.demucs_model
def test_demucs_cached_cpu_separates_reference_mix(tmp_path, monkeypatch):
    if os.getenv("TAB_AGENT_TEST_DEMUCS") != "1":
        pytest.skip("opt in with TAB_AGENT_TEST_DEMUCS=1 and a pre-provisioned htdemucs checkpoint")
    import torch

    assert agents.DEMUCS_API_AVAILABLE, "the requested Demucs API must import"
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    checkpoint = cache / "955717e8-8726e21a.th"
    assert checkpoint.is_file(), f"htdemucs checkpoint missing in {cache}"
    with checkpoint.open("rb") as weights:
        assert hashlib.file_digest(weights, "sha256").hexdigest().startswith("8726e21a")

    def no_fallback_or_download(*args, **kwargs):
        pytest.fail("the cached real-inference lane must not download or use a fallback")

    monkeypatch.setattr(torch.hub, "download_url_to_file", no_fallback_or_download)
    monkeypatch.setattr(SplitterAgent, "_separate_with_subprocess", no_fallback_or_download)
    monkeypatch.setattr(agents, "health", HealthTracker())
    audio = tmp_path / "mix.wav"
    sr = 44100
    center = plucks([(28, 0.2, 1.9), (64, 0.8, 1.4)], sr=sr)
    side = plucks([(57, 0.4, 1.4), (61, 0.4, 1.4)], sr=sr)
    mix = np.column_stack((center + side, center + 0.25 * side))
    sf.write(audio, mix, sr)

    splitter = SplitterAgent(output_dir=str(tmp_path / "stems"), device="cpu")
    paths = splitter.separate_stems(str(audio))
    assert agents.health.as_dict()["components"]["demucs"] == "loaded"
    arrays = []
    for key in ("guitar", "bass"):
        pcm, rate = sf.read(paths[key], always_2d=True)
        assert rate == sr and pcm.shape == mix.shape
        assert np.isfinite(pcm).all() and np.max(np.abs(pcm)) > 0.001
        assert not np.allclose(pcm, mix, atol=1e-3)
        arrays.append(pcm)
    assert not np.allclose(arrays[0], arrays[1], atol=1e-3)
