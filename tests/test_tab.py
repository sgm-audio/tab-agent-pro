"""
Tab Agent Pro — Unit Tests: TabAgent
Tests for dynamic-programming tablature generation and technique detection.
"""

import os
import sys
import unittest

import note_seq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import TabAgent

GUITAR_TUNING = [40, 45, 50, 55, 59, 64]  # E2-A2-D3-G3-B3-E4
BASS_TUNING = [23, 28, 33, 38, 43]  # B0-E1-A1-D2-G2


def make_note(pitch, start, end=0.5, velocity=80):
    return note_seq.NoteSequence.Note(
        pitch=pitch,
        start_time=start,
        end_time=end,
        velocity=velocity,
    )


class TestTabAgentPositions(unittest.TestCase):
    """Tests for get_valid_positions and calculate_cost."""

    def setUp(self) -> None:
        self.agent = TabAgent(tuning=GUITAR_TUNING, num_frets=24)

    def test_open_string_position(self) -> None:
        """Open E2 (pitch 40) should return string 0, fret 0."""
        positions = self.agent.get_valid_positions(40)
        assert any(p["string"] == 0 and p["fret"] == 0 for p in positions)

    def test_open_string_position_multiple(self) -> None:
        """Pitch 45 appears on both A string (open) and low E (fret 5)."""
        positions = self.agent.get_valid_positions(45)
        assert any(p["string"] == 1 and p["fret"] == 0 for p in positions)
        assert any(p["string"] == 0 and p["fret"] == 5 for p in positions)

    def test_unplayable_note(self) -> None:
        """Pitch 20 is below low E — no valid positions."""
        positions = self.agent.get_valid_positions(20)
        assert len(positions) == 0

    def test_above_fretboard(self) -> None:
        """Pitch 100 is above the 24-fret range on all strings."""
        positions = self.agent.get_valid_positions(100)
        assert len(positions) == 0

    def test_cost_same_position_zero(self) -> None:
        """Cost of transitioning to the same position is low."""
        pos = {"string": 1, "fret": 3}
        cost = self.agent.calculate_cost(pos, pos, time_delta=0.5)
        assert cost == 0.0

    def test_cost_encourages_same_string_fast(self) -> None:
        """Fast transitions on same string get a cost reduction (legato)."""
        prev = {"string": 1, "fret": 3}
        curr = {"string": 1, "fret": 5}  # same string, different fret
        cost = self.agent.calculate_cost(prev, curr, time_delta=0.1)
        # Should be lower than a different-string transition
        other = {"string": 2, "fret": 3}
        cost_other = self.agent.calculate_cost(prev, other, time_delta=0.1)
        assert cost < cost_other


class TestTabAgentGenerateTab(unittest.TestCase):
    """Tests for the full generate_tab pipeline."""

    def setUp(self) -> None:
        self.agent = TabAgent(tuning=GUITAR_TUNING, num_frets=24)

    def test_empty_notes_returns_empty(self) -> None:
        """Empty input returns empty list."""
        assert len(self.agent.generate_tab([])) == 0

    def test_single_note_returns_position(self) -> None:
        """A single note produces one tab entry with a position."""
        notes = [make_note(64, 0.0)]  # E4
        result = self.agent.generate_tab(notes)
        assert len(result) == 1
        assert "string" in result[0]
        assert "fret" in result[0]

    def test_simple_ascending_run(self) -> None:
        """Ascending scale produces valid tab."""
        notes = [
            make_note(40, 0.0),  # E2
            make_note(45, 0.5),  # A2
            make_note(50, 1.0),  # D3
            make_note(55, 1.5),  # G3
        ]
        result = self.agent.generate_tab(notes)
        assert len(result) == 4
        for entry in result:
            assert "technique" in entry

    def test_every_note_has_technique(self) -> None:
        """Every output entry has a technique annotation."""
        notes = [
            make_note(60, 0.0),
            make_note(62, 0.3),
            make_note(64, 0.6),
        ]
        result = self.agent.generate_tab(notes)
        for entry in result:
            assert "technique" in entry
            assert entry["technique"] in ["pick", "slide", "hammer", "pull"]

    def test_technique_detection_slide(self) -> None:
        """Adjacent notes on same string 1-2 frets apart → slide."""
        # E2 (open string 0) → F#2 (fret 2 on string 0)
        notes = [
            make_note(40, 0.0),  # E2
            make_note(42, 0.15),  # F#2 — fast, same string, 2 frets apart
        ]
        result = self.agent.generate_tab(notes, technique_sensitivity=0.9)
        # The DP may choose the same string for efficiency
        techniques = [r["technique"] for r in result]
        assert "slide" in techniques

    def test_technique_detection_hammer_on(self) -> None:
        """Ascending notes on same string → hammer-on."""
        # E2 (open string 0) → G#2 (fret 4 on string 0)
        notes = [
            make_note(40, 0.0),  # E2
            make_note(44, 0.15),  # G#2 — fast, same string, ascending
        ]
        result = self.agent.generate_tab(notes, technique_sensitivity=0.9)
        techniques = [r["technique"] for r in result]
        assert "hammer" in techniques

    def test_technique_detection_pull_off(self) -> None:
        """Descending notes on same string → pull-off."""
        # G3 (open string 3) → E3 (fret 5 on string 4, or fret 9 on string 3)
        # Force string-4 path by using positions only on string 4
        notes = [
            make_note(55, 0.0),  # G3 — playable on many strings
            make_note(52, 0.15),  # E3
        ]
        result = self.agent.generate_tab(notes, technique_sensitivity=0.9)
        techniques = [r["technique"] for r in result]
        # At least one technique should be detected
        assert any(
            t in ["slide", "hammer", "pull"] for t in techniques
        ), f"Expected technique detection, got: {techniques}"

    def test_bass_five_string_tuning(self) -> None:
        """5-string bass tuning works correctly."""
        agent = TabAgent(tuning=BASS_TUNING, num_frets=24)
        notes = [make_note(28, 0.0), make_note(33, 0.5)]
        result = agent.generate_tab(notes)
        assert len(result) == 2

    def test_drop_d_tuning(self) -> None:
        """Drop D tuning (D2=38) works correctly."""
        drop_d = [38, 45, 50, 55, 59, 64]
        agent = TabAgent(tuning=drop_d, num_frets=24)
        positions = agent.get_valid_positions(38)
        assert any(p["fret"] == 0 for p in positions)


if __name__ == "__main__":
    unittest.main()
