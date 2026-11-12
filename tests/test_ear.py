"""
Tab Agent Pro — Unit Tests: EarAgent
Tests for transcription logic, note conversion, and instrument filtering.
"""

import os
import sys
import unittest

import note_seq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import EarAgent


class TestEarAgentInstrumentFilter(unittest.TestCase):
    """Tests for instrument range filtering."""

    def setUp(self) -> None:
        self.ear = EarAgent(device="cpu")

    def _make_note(self, pitch, start=0.0, end=0.5, velocity=80):
        return note_seq.NoteSequence.Note(
            pitch=pitch,
            start_time=start,
            end_time=end,
            velocity=velocity,
        )

    def test_guitar_range_filters_outrageous(self) -> None:
        """Notes outside standard guitar range are removed."""
        notes = [
            self._make_note(30),  # too low (below E2=40)
            self._make_note(45),  # valid
            self._make_note(90),  # too high (above E6=88)
            self._make_note(64),  # valid
        ]
        result = self.ear._filter_by_instrument_range(notes, "Guitar")
        assert len(result) == 2
        assert 45 in [n.pitch for n in result]
        assert 64 in [n.pitch for n in result]

    def test_bass_range_allows_low_notes(self) -> None:
        """5-string bass range includes B0 (23) through G4 (67)."""
        notes = [
            self._make_note(23),  # B0 — valid
            self._make_note(40),  # valid
            self._make_note(67),  # G4 — valid (upper limit)
            self._make_note(70),  # too high
        ]
        result = self.ear._filter_by_instrument_range(notes, "Bass")
        assert len(result) == 3
        assert 70 not in [n.pitch for n in result]

    def test_guitar_drop_d_min_pitch_preserves_low_d2(self) -> None:
        """Drop D min_pitch=38 preserves D2 (38) and D#2 (39) while filtering below 38."""
        notes = [
            self._make_note(36),  # below D2
            self._make_note(38),  # D2 — valid in Drop D
            self._make_note(39),  # D#2 — valid in Drop D
            self._make_note(64),  # E4 — valid
        ]
        result = self.ear._filter_by_instrument_range(notes, "Guitar", min_pitch=38)
        assert [n.pitch for n in result] == [38, 39, 64]


class TestEarAgentHumanize(unittest.TestCase):
    """Tests for humanize_and_clean."""

    def setUp(self) -> None:
        self.ear = EarAgent(device="cpu")

    def _make_note(self, pitch, start=0.0, end=0.5, velocity=80):
        return note_seq.NoteSequence.Note(
            pitch=pitch,
            start_time=start,
            end_time=end,
            velocity=velocity,
        )

    def test_removes_ultrashort_notes(self) -> None:
        """Notes shorter than 0.05s are removed."""
        notes = [
            self._make_note(60, start=0.0, end=0.02),  # too short
            self._make_note(64, start=0.0, end=0.5),  # valid
        ]
        result = self.ear.humanize_and_clean(notes)
        assert len(result) == 1

    def test_removes_duplicate_notes_same_time(self) -> None:
        """Duplicate pitches at the same time are removed."""
        notes = [
            self._make_note(60, start=0.0, end=0.5),
            self._make_note(60, start=0.0, end=0.5),  # duplicate
        ]
        result = self.ear.humanize_and_clean(notes)
        assert len(result) == 1

    def test_bass_enforces_upper_limit(self) -> None:
        """Bass notes above pitch 67 are removed."""
        notes = [
            self._make_note(40, start=0.0, end=0.5),
            self._make_note(72, start=0.5, end=1.0),  # above bass range
        ]
        result = self.ear.humanize_and_clean(notes, is_bass=True)
        assert len(result) == 1

    def test_transcribe_stem_failure_records_failed_metric(self) -> None:
        """Failed transcription must record an error in default_metrics.stage_errors."""
        from unittest.mock import patch

        from monitoring import default_metrics

        before_errors = default_metrics.stage_errors.get("basic_pitch_transcription", 0)
        with (
            patch.object(
                self.ear,
                "_transcribe_with_basic_pitch",
                side_effect=RuntimeError("bp boom"),
            ),
            self.assertRaises(RuntimeError),
        ):
            self.ear.transcribe_stem("nonexistent.wav")
        after_errors = default_metrics.stage_errors.get("basic_pitch_transcription", 0)
        assert after_errors == before_errors + 1


if __name__ == "__main__":
    unittest.main()
