"""Opt-in CPU inference smoke; enabling it makes missing models a failure.

TAB_AGENT_REAL_INFERENCE=1 pytest -q -o addopts='' tests/test_real_inference.py
This is a smoke test, not a substitute for the labelled evaluation corpus.
"""

import os

import numpy as np
import pytest
import soundfile as sf


@pytest.mark.skipif(
    os.environ.get("TAB_AGENT_REAL_INFERENCE") != "1",
    reason="real inference not requested; set TAB_AGENT_REAL_INFERENCE=1",
)
def test_real_basic_pitch_cpu_transcribes_known_tone(tmp_path):
    import agents

    assert (
        agents.BASIC_PITCH_AVAILABLE
    ), "Real inference requested but Basic Pitch/ONNX is unavailable"
    sr = 22050
    t = np.arange(sr * 2) / sr
    # A stable harmonic tone with attack/release, bracketed by silence.
    envelope = np.minimum(
        1, np.minimum(np.maximum(0, (t - 0.2) / 0.05), np.maximum(0, (1.7 - t) / 0.1))
    )
    signal = envelope * (0.35 * np.sin(2 * np.pi * 440 * t) + 0.12 * np.sin(2 * np.pi * 880 * t))
    path = tmp_path / "a4.wav"
    sf.write(path, signal, sr)
    notes = agents.EarAgent(device="cpu").transcribe_stem(str(path), target="Guitar")
    assert any(abs(n.pitch - 69) <= 1 and n.end_time > n.start_time for n in notes), [
        (n.pitch, n.start_time, n.end_time) for n in notes
    ]


@pytest.mark.skipif(
    os.environ.get("TAB_AGENT_REAL_INFERENCE") != "1",
    reason="real inference not requested; set TAB_AGENT_REAL_INFERENCE=1",
)
def test_real_basic_pitch_cpu_silence_does_not_invent_notes(tmp_path):
    import agents

    assert (
        agents.BASIC_PITCH_AVAILABLE
    ), "Real inference requested but Basic Pitch/ONNX is unavailable"
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(22050 * 2), 22050)
    assert agents.EarAgent(device="cpu").transcribe_stem(str(path), target="Guitar") == []
