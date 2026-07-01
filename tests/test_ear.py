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


class TestEarAgentNoteConversion(unittest.TestCase):
    """Tests for _convert_to_noteseq and _convert_prettymidi_to_noteseq."""

    def setUp(self):
        self.ear = EarAgent(model_id="mimbres/YourMT3", device="cpu", prefer_yourmt3=False)

    def test_mt3_token_parsing_note_on_off(self):
        """Parse NOTE_ON/NOTE_OFF/TIME_SHIFT token format."""
        events = (
            "TIME_SHIFT 0.0 NOTE_ON 60 VELOCITY 80 "
            "TIME_SHIFT 0.5 NOTE_OFF 60 "
            "TIME_SHIFT 0.1 NOTE_ON 64 VELOCITY 70 "
            "TIME_SHIFT 0.3 NOTE_OFF 64"
        )
        notes = self.ear._convert_to_noteseq(events)
        self.assertEqual(len(notes), 2)
        self.assertEqual(notes[0].pitch, 60)
        self.assertAlmostEqual(notes[0].start_time, 0.0)
        self.assertAlmostEqual(notes[0].end_time, 0.5)
        self.assertEqual(notes[0].velocity, 80)
        self.assertEqual(notes[1].pitch, 64)

    def test_kv_format_parsing(self):
        """Parse key:value output format."""
        events = "pitch:60,start:0.0,end:0.5,velocity:80 pitch:64,start:0.5,end:1.0,velocity:75"
        notes = self.ear._convert_to_noteseq(events)
        self.assertEqual(len(notes), 2)
        self.assertEqual(notes[0].pitch, 60)
        self.assertAlmostEqual(notes[0].end_time, 0.5)

    def test_empty_input_returns_empty(self):
        """Empty / whitespace input returns empty list."""
        self.assertEqual(len(self.ear._convert_to_noteseq("")), 0)
        self.assertEqual(len(self.ear._convert_to_noteseq("   ")), 0)

    def test_unparseable_format_returns_empty_with_warning(self):
        """Unknown format returns empty list (does not crash)."""
        notes = self.ear._convert_to_noteseq("garbage input 123")
        self.assertEqual(len(notes), 0)

    def test_min_duration_filter(self):
        """Notes shorter than min_duration are excluded."""
        events = "TIME_SHIFT 0.0 NOTE_ON 60 VELOCITY 80 TIME_SHIFT 0.01 NOTE_OFF 60"
        notes = self.ear._convert_to_noteseq(events, min_duration=0.05)
        self.assertEqual(len(notes), 0)


class TestEarAgentInstrumentFilter(unittest.TestCase):
    """Tests for instrument range filtering."""

    def setUp(self):
        self.ear = EarAgent(model_id="mimbres/YourMT3", device="cpu", prefer_yourmt3=False)

    def _make_note(self, pitch, start=0.0, end=0.5, velocity=80):
        return note_seq.NoteSequence.Note(
            pitch=pitch,
            start_time=start,
            end_time=end,
            velocity=velocity,
        )

    def test_guitar_range_filters_outrageous(self):
        """Notes outside standard guitar range are removed."""
        notes = [
            self._make_note(30),  # too low (below E2=40)
            self._make_note(45),  # valid
            self._make_note(90),  # too high (above E6=88)
            self._make_note(64),  # valid
        ]
        result = self.ear._filter_by_instrument_range(notes, "Guitar")
        self.assertEqual(len(result), 2)
        self.assertIn(45, [n.pitch for n in result])
        self.assertIn(64, [n.pitch for n in result])

    def test_bass_range_allows_low_notes(self):
        """5-string bass range includes B0 (23) through G4 (67)."""
        notes = [
            self._make_note(23),  # B0 — valid
            self._make_note(40),  # valid
            self._make_note(67),  # G4 — valid (upper limit)
            self._make_note(70),  # too high
        ]
        result = self.ear._filter_by_instrument_range(notes, "Bass")
        self.assertEqual(len(result), 3)
        self.assertNotIn(70, [n.pitch for n in result])


class TestEarAgentHumanize(unittest.TestCase):
    """Tests for humanize_and_clean."""

    def setUp(self):
        self.ear = EarAgent(model_id="mimbres/YourMT3", device="cpu", prefer_yourmt3=False)

    def _make_note(self, pitch, start=0.0, end=0.5, velocity=80):
        return note_seq.NoteSequence.Note(
            pitch=pitch,
            start_time=start,
            end_time=end,
            velocity=velocity,
        )

    def test_removes_ultrashort_notes(self):
        """Notes shorter than 0.05s are removed."""
        notes = [
            self._make_note(60, start=0.0, end=0.02),  # too short
            self._make_note(64, start=0.0, end=0.5),  # valid
        ]
        result = self.ear.humanize_and_clean(notes)
        self.assertEqual(len(result), 1)

    def test_removes_duplicate_notes_same_time(self):
        """Duplicate pitches at the same time are removed."""
        notes = [
            self._make_note(60, start=0.0, end=0.5),
            self._make_note(60, start=0.0, end=0.5),  # duplicate
        ]
        result = self.ear.humanize_and_clean(notes)
        self.assertEqual(len(result), 1)

    def test_bass_enforces_upper_limit(self):
        """Bass notes above pitch 67 are removed."""
        notes = [
            self._make_note(40, start=0.0, end=0.5),
            self._make_note(72, start=0.5, end=1.0),  # above bass range
        ]
        result = self.ear.humanize_and_clean(notes, is_bass=True)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
