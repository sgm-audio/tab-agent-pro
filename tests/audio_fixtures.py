"""Small, deterministic audio and note fixtures with independent ground truth."""

import note_seq
import numpy as np

SAMPLE_RATE = 32000  # neither Basic Pitch's input rate nor Demucs' model rate
STANDARD = [40, 45, 50, 55, 59, 64]
DROP_D = [38, 45, 50, 55, 59, 64]
BASS_5 = [23, 28, 33, 38, 43]
BASS_4 = [28, 33, 38, 43]


def make_note(pitch: int, start: float, end: float) -> note_seq.NoteSequence.Note:
    return note_seq.NoteSequence.Note(pitch=pitch, start_time=start, end_time=end, velocity=80)


def plucks(events, *, sr=SAMPLE_RATE, duration=2.4):
    """Render attacks, decays and harmonics, without sampled instruments or downloads."""
    signal = np.zeros(round(duration * sr))
    for pitch, start, end in events:
        t = np.arange(round((end - start) * sr)) / sr
        freq = 440 * 2 ** ((pitch - 69) / 12)
        envelope = np.minimum(t / 0.007, 1) * np.exp(-t / 0.43)
        envelope *= np.minimum((t[-1] - t) / 0.025, 1)
        wave = sum(
            amp * np.sin(2 * np.pi * harmonic * freq * t + (harmonic - 1) * 0.3)
            for harmonic, amp in enumerate((0.18, 0.055, 0.023, 0.011), 1)
        )
        first = round(start * sr)
        signal[first : first + len(t)] += wave * envelope
    return signal


def tab_pitches(tab, tuning, fret_limit):
    """Decode tab independently, checking indices before indexing the tuning."""
    for pos in tab:
        assert 0 <= pos["string"] < len(tuning), pos
        assert 0 <= pos["fret"] <= fret_limit, pos
    return [tuning[pos["string"]] + pos["fret"] for pos in tab]
