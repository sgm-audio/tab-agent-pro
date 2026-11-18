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


class TestTabAgentChordFeasibility(unittest.TestCase):
    """Simultaneous pitches must occupy distinct strings in playable chords."""

    def setUp(self) -> None:
        self.agent = TabAgent(tuning=GUITAR_TUNING, num_frets=24)

    def _assert_no_same_string_chord(self, tab: list) -> None:
        """No time-group may contain two notes on the same string."""
        groups: dict[float, list[int]] = {}
        for pos in tab:
            key = pos["start_time"]
            groups.setdefault(key, []).append(pos["string"])
        for strings in groups.values():
            assert len(strings) == len(set(strings)), f"chord collides on a string: {strings}"

    def test_impossible_chord_is_feasible(self) -> None:
        """E2 + E6 + B3 simultaneous must not emit two frets on one string.

        E2(40) uses open low E, E6(88) uses high E fret 24, and B3(59)
        can stay open on the B string without exceeding the fret span.
        Without a chord-feasibility constraint B3 was placed on the low E
        string (fret 19), colliding with E2 (fret 0). The DP must keep all
        three notes and place them on distinct strings.
        """
        notes = [
            make_note(40, 0.0),  # E2
            make_note(88, 0.0),  # E6
            make_note(59, 0.0),  # B3
        ]
        tab = self.agent.generate_tab(notes)
        assert len(tab) == 3, f"all three chord notes must survive: {tab}"
        self._assert_no_same_string_chord(tab)
        # Three simultaneous notes with no collision must occupy three strings.
        assert (
            len({e["string"] for e in tab}) == 3
        ), "chord must use three distinct strings: " + str([(e["string"], e["fret"]) for e in tab])

    def test_same_string_chord_cost_is_penalised(self) -> None:
        """A simultaneous same-string transition carries the chord penalty."""
        prev = {"string": 0, "fret": 0}
        curr_same = {"string": 0, "fret": 19}
        curr_diff = {"string": 1, "fret": 14}
        cost_same = self.agent.calculate_cost(prev, curr_same, time_delta=0.0)
        cost_diff = self.agent.calculate_cost(prev, curr_diff, time_delta=0.0)
        assert cost_same > cost_diff
        # The penalty dominates the legato bonus, so same-string is never cheaper.
        assert cost_same >= 1000.0

    def test_rapid_sequential_motion_has_no_chord_penalty(self) -> None:
        """A 20 ms gap between onsets is not evidence of a chord or a slide."""
        prev = {"string": 0, "fret": 0}
        curr = {"string": 0, "fret": 2}
        cost = self.agent.calculate_cost(prev, curr, time_delta=0.02)
        assert cost == 3.0 - 5.0  # two-fret motion, favor economical string reuse

    def test_normal_chord_still_renders(self) -> None:
        """A normal playable chord is unaffected by the constraint."""
        # C major: C3(48) E3(52) G3(55) — spread across different strings.
        notes = [make_note(48, 0.0), make_note(52, 0.0), make_note(55, 0.0)]
        tab = self.agent.generate_tab(notes)
        assert len(tab) == 3
        self._assert_no_same_string_chord(tab)

    def test_realistic_chord_voicings_are_feasible(self) -> None:
        """Common chord shapes never emit a same-string collision."""
        # Power chord (root + fifth), major triad, and a dense 4-note chord,
        # each as a single simultaneous onset.
        voicings = [
            [40, 47],  # E2 + B2 (power chord)
            [48, 52, 55],  # C3 + E3 + G3 (C major)
            [40, 45, 52, 55],  # E2 + A2 + E3 + G3 (E major shape)
            [55, 59, 64, 67],  # G3 + B3 + E4 + G4 (G major shape)
        ]
        for pitches in voicings:
            tab = self.agent.generate_tab([make_note(p, 0.0) for p in pitches])
            # Every note must survive (these are all playable chords).
            assert len(tab) == len(
                pitches
            ), f"chord {pitches} lost notes: {len(tab)}/{len(pitches)}"
            self._assert_no_same_string_chord(tab)

    def test_rendered_output_has_no_collision_markers(self) -> None:
        """The rendered ASCII tab must never join two frets with '/'.

        render_tab_text joins two same-string positions in one column with
        'x/y'; a collision marker in the output means the constraint leaked.
        """
        import main

        notes = [make_note(40, 0.0), make_note(88, 0.0), make_note(59, 0.0)]
        tab = self.agent.generate_tab(notes)
        text = main.render_tab_text(tab)
        assert text is not None
        # Split on the column separator and check no cell contains '/'.
        for line in text.splitlines():
            if "|" in line and not line.startswith("Legend"):
                for cell in line.split("|")[1:]:
                    assert "/" not in cell, f"collision marker '/' in rendered cell: {cell!r}"


if __name__ == "__main__":
    unittest.main()
