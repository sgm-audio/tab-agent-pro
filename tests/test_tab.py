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


def make_note(pitch, start, end=None, velocity=80):
    if end is None:
        end = start + 0.5
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

    def test_cost_non_negative_on_negative_time_delta(self) -> None:
        """Unsorted notes with negative time_delta must not produce negative fret penalties."""
        prev = {"string": 0, "fret": 1}
        curr = {"string": 0, "fret": 12}
        cost = self.agent.calculate_cost(prev, curr, time_delta=-0.5)
        assert cost > 0.0

    def test_prefer_low_strings_penalizes_high_frets_on_upper_strings(self) -> None:
        """prefer_low_strings=True adds penalty for high frets (>=10) on upper strings (>=2)."""
        agent_pref = TabAgent(tuning=BASS_TUNING, num_frets=24, prefer_low_strings=True)
        agent_default = TabAgent(tuning=BASS_TUNING, num_frets=24, prefer_low_strings=False)
        prev = {"string": 2, "fret": 8}
        curr = {"string": 3, "fret": 12}
        assert (
            agent_pref.calculate_cost(prev, curr, time_delta=0.5)
            == agent_default.calculate_cost(prev, curr, time_delta=0.5) + 1.0
        )


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

    def test_every_note_has_the_unmarked_technique_value(self) -> None:
        """Keep the legacy field, without inferring technique from notes."""
        notes = [
            make_note(60, 0.0),
            make_note(62, 0.3),
            make_note(64, 0.6),
        ]
        result = self.agent.generate_tab(notes)
        assert [entry["technique"] for entry in result] == ["pick"] * len(notes)

    def test_contiguous_notes_do_not_claim_slide(self) -> None:
        """Close same-string notes do not prove a slide."""
        # E2 (open string 0) → F#2 (fret 2 on string 0)
        notes = [
            make_note(40, 0.0, 0.15),  # E2 releases at the next onset
            make_note(42, 0.15),  # F#2 — fast, same string, 2 frets apart
        ]
        result = self.agent.generate_tab(notes, technique_sensitivity=0.9)
        assert [r["technique"] for r in result] == ["pick", "pick"]

    def test_ascending_notes_do_not_claim_hammer_on(self) -> None:
        """Even contiguous ascending notes may be picked separately."""
        # E2 (open string 0) → G#2 (fret 4 on string 0)
        notes = [
            make_note(40, 0.0, 0.15),  # E2 releases at the next onset
            make_note(44, 0.15),  # G#2 — fast, same string, ascending
        ]
        result = self.agent.generate_tab(notes, technique_sensitivity=0.9)
        assert [r["technique"] for r in result] == ["pick", "pick"]

    def test_descending_notes_do_not_claim_pull_off(self) -> None:
        """Descending notes on same string → pull-off."""
        notes = [make_note(44, 0.0, 0.15), make_note(40, 0.15)]
        result = self.agent.generate_tab(notes, technique_sensitivity=0.9)
        assert [r["technique"] for r in result] == ["pick", "pick"]

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
