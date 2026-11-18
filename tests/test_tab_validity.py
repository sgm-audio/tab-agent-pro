"""
Regression tests for tablature physicality.

These lock in the three defects that the static review found and that produced
unplayable tabs:

  * P0-1 — chords were fingered one note at a time, so a single column could
    place two notes on the same string and stretch far beyond a hand span;
  * P0-2 — the renderer chained its 50 ms grouping, so a fast run collapsed
    into one column;
  * P0-7 — string labels came from the string *count*, so Drop D printed ``E``
    for a D string.

They also cover ``tab_validation`` itself, which is what makes the invariants
enforceable rather than aspirational.
"""

import os
import sys
import unittest
from pathlib import Path

import note_seq
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import TabAgent
from main import render_tab_text, tuning_string_labels
from tab_validation import TabValidationError, validate_tab

GUITAR_TUNING = [40, 45, 50, 55, 59, 64]  # E2-A2-D3-G3-B3-E4
DROP_D_TUNING = [38, 45, 50, 55, 59, 64]  # D2-A2-D3-G3-B3-E4
BASS_TUNING = [23, 28, 33, 38, 43]  # B0-E1-A1-D2-G2


def make_note(pitch, start, duration=0.4, velocity=80):
    return note_seq.NoteSequence.Note(
        pitch=pitch,
        start_time=start,
        end_time=start + duration,
        velocity=velocity,
    )


def chord(*pitches, start=0.0):
    return [make_note(p, start) for p in pitches]


def columns_of(tab):
    """Map column index -> list of rows in that column."""
    columns = {}
    for row in tab:
        columns.setdefault(row["column"], []).append(row)
    return columns


class TestChordFingering(unittest.TestCase):
    """A chord must be fingered as a chord."""

    def setUp(self) -> None:
        self.agent = TabAgent(tuning=GUITAR_TUNING, num_frets=24)

    def test_no_column_repeats_a_string(self) -> None:
        """The reported defect: (40, 59, 64) put two notes on string 4."""
        tab = self.agent.generate_tab(chord(40, 59, 64))
        for rows in columns_of(tab).values():
            strings = [r["string"] for r in rows]
            assert len(set(strings)) == len(strings), rows
        assert sorted(r["fret"] for r in tab) == [0, 0, 0]

    def test_open_chord_uses_open_strings(self) -> None:
        """Open E + B + high E are frets 0 on three different strings."""
        tab = self.agent.generate_tab(chord(40, 59, 64))
        assert {(r["string"], r["fret"]) for r in tab} == {(0, 0), (4, 0), (5, 0)}

    def test_power_chord_is_one_column(self) -> None:
        tab = self.agent.generate_tab(chord(40, 47, 52))
        assert len(columns_of(tab)) == 1
        assert len(tab) == 3

    def test_hand_span_is_bounded(self) -> None:
        """No column may require a stretch wider than the hand span."""
        tab = self.agent.generate_tab(chord(40, 63))
        for rows in columns_of(tab).values():
            fretted = [r["fret"] for r in rows if r["fret"] > 0]
            if len(fretted) > 1:
                assert max(fretted) - min(fretted) <= self.agent.MAX_HAND_SPAN

    def test_random_chords_are_physical(self) -> None:
        """Property check over voicings that previously produced collisions."""
        import random

        random.seed(11)
        for _ in range(60):
            pitches = sorted(random.sample(range(40, 89), 3))
            tab = self.agent.generate_tab(chord(*pitches))
            for rows in columns_of(tab).values():
                strings = [r["string"] for r in rows]
                assert len(set(strings)) == len(strings), (pitches, rows)

    def test_first_note_is_not_up_the_neck(self) -> None:
        """A lone E4 used to be fingered at fret 24 on the low E string."""
        tab = self.agent.generate_tab(chord(64))
        assert [(r["string"], r["fret"]) for r in tab] == [(5, 0)]

    def test_unplayable_note_in_chord_keeps_the_others(self) -> None:
        """One impossible note must not destroy its column."""
        tab = self.agent.generate_tab(chord(40, 59, 21))
        assert len(tab) == 2
        assert self.agent.last_skipped_notes == [21]
        assert "below the lowest open string" in self.agent.last_skipped_columns[0]["reason"]

    def test_duplicate_pitch_in_one_chord_is_single_note(self) -> None:
        tab = self.agent.generate_tab(chord(40, 40))
        assert len(tab) == 1


class TestColumnGrouping(unittest.TestCase):
    """Sequential notes must not be merged into one chord."""

    def setUp(self) -> None:
        self.agent = TabAgent(tuning=GUITAR_TUNING, num_frets=24)

    def test_fast_run_on_one_string_is_not_merged(self) -> None:
        """Six notes 12 ms apart are a run, not a chord (the P0-2 case)."""
        notes = [make_note(40 + i, i * 0.012) for i in range(6)]
        tab = self.agent.generate_tab(notes)
        assert len(columns_of(tab)) == 6
        assert [r["fret"] for r in tab] == [0, 1, 2, 3, 4, 5]

    def test_run_at_40ms_produces_one_column_per_note(self) -> None:
        notes = [make_note(40 + i, i * 0.04) for i in range(5)]
        tab = self.agent.generate_tab(notes)
        assert len(columns_of(tab)) == 5
        assert self.agent.last_skipped_notes == []

    def test_grouping_compares_to_column_start_not_previous_note(self) -> None:
        """A chain of 15 ms gaps must not become one 75 ms 'chord'."""
        agent = TabAgent(tuning=GUITAR_TUNING, num_frets=24)
        # 40, 45, 50, 55 can be fingered as a chord (4 strings), so use pitches
        # that cannot: five distinct pitches all needing the low E string.
        notes = [make_note(40 + i, i * 0.015) for i in range(5)]
        columns = agent.group_notes(notes)
        assert len(columns) == 5


class TestToleranceParity(unittest.TestCase):
    """The agent and the renderer must agree on what "simultaneous" means."""

    def test_column_tolerance_matches_the_agent(self) -> None:
        """Two different tolerances made the two stages disagree about chords.

        The agent grouped onsets with one number and the renderer with another,
        so a tab could be fingered as a chord and then printed as a run (or the
        reverse). One constant, asserted here.
        """
        import main

        assert main.TAB_COLUMN_TOLERANCE_S == TabAgent.GROUP_TOLERANCE_S

    def test_tolerance_is_under_one_basic_pitch_hop(self) -> None:
        """A wider window would merge consecutive frames into one chord."""
        assert TabAgent.GROUP_TOLERANCE_S <= 0.012 + 1e-9

    def test_agent_and_renderer_group_a_run_identically(self) -> None:
        import main

        agent = TabAgent(tuning=GUITAR_TUNING, num_frets=24)
        notes = [make_note(40 + i, i * 0.012) for i in range(6)]
        agent_columns = len(columns_of(agent.generate_tab(notes)))

        rendered = [
            {"string": r["string"], "fret": r["fret"], "start_time": r["start_time"]}
            for r in agent.generate_tab(notes)
        ]
        assert len(main._group_tab_columns(rendered)) == agent_columns


class TestTuningLabels(unittest.TestCase):
    """String names must come from the tuning, not the string count."""

    def test_standard_guitar(self) -> None:
        assert tuning_string_labels(GUITAR_TUNING) == ["E", "B", "G", "D", "A", "E"]

    def test_drop_d_low_string_is_d(self) -> None:
        assert tuning_string_labels(DROP_D_TUNING) == ["E", "B", "G", "D", "A", "D"]

    def test_five_string_bass(self) -> None:
        assert tuning_string_labels(BASS_TUNING) == ["G", "D", "A", "E", "B"]

    def test_four_string_bass_is_not_five(self) -> None:
        assert tuning_string_labels([28, 33, 38, 43]) == ["G", "D", "A", "E"]

    def test_drop_d_renders_d_on_bottom_line(self) -> None:
        """The P0-7 symptom: a Drop-D tab used to print E| for its D string."""
        tab = self.agent_tab()
        text = render_tab_text(tab, "Guitar", tuning=DROP_D_TUNING)
        lines = [ln for ln in text.splitlines() if "|" in ln]
        assert lines[-1].startswith("D|"), lines
        assert lines[0].startswith("E|")

    def test_grid_width_follows_tuning(self) -> None:
        tab = self.agent_tab()
        text = render_tab_text(tab, "Bass", tuning=BASS_TUNING)
        lines = [ln for ln in text.splitlines() if "|" in ln]
        assert len(lines) == 5

    def agent_tab(self):
        agent = TabAgent(tuning=DROP_D_TUNING, num_frets=24)
        return agent.generate_tab([make_note(38, 0.0)])


class TestRenderAlignment(unittest.TestCase):
    """The renderer must not invent unplayable chords."""

    def test_same_string_same_onset_is_split_by_the_renderer(self) -> None:
        tab = [
            {"string": 3, "fret": 15, "technique": "pick", "start_time": 0.0},
            {"string": 3, "fret": 8, "technique": "pull", "start_time": 0.01},
        ]
        text = render_tab_text(tab, "Guitar", tuning=GUITAR_TUNING)
        assert "/" not in text
        assert "+" not in text

    def test_true_chord_stays_aligned(self) -> None:
        tab = [
            {"string": 0, "fret": 0, "technique": "pick", "start_time": 0.0},
            {"string": 5, "fret": 0, "technique": "pick", "start_time": 0.0},
        ]
        lines = [
            ln
            for ln in render_tab_text(tab, "Guitar", tuning=GUITAR_TUNING).splitlines()
            if "|" in ln
        ]
        # Both rows show their note in the first column, so the cells line up.
        assert lines[0].split("|")[1].strip() == "0"
        assert lines[-1].split("|")[1].strip() == "0"

    def test_generated_tab_has_no_slash_cells(self) -> None:
        agent = TabAgent(tuning=GUITAR_TUNING, num_frets=24)
        tab = agent.generate_tab(chord(40, 45, 50, 55, 59))
        text = render_tab_text(tab, "Guitar", tuning=GUITAR_TUNING)
        assert "/" not in text


class TestValidateTab(unittest.TestCase):
    """The validator is the gate that keeps the invariants honest."""

    def test_accepts_a_clean_tab(self) -> None:
        tab = [
            {"string": 0, "fret": 0, "pitch": 40, "technique": "pick", "column": 0},
            {"string": 5, "fret": 0, "pitch": 64, "technique": "pick", "column": 0},
        ]
        report = validate_tab(tab, GUITAR_TUNING, 24)
        assert report.columns == 1
        assert report.notes == 2

    def test_rejects_two_notes_on_one_string(self) -> None:
        tab = [
            {"string": 4, "fret": 0, "pitch": 59, "technique": "pick", "column": 0},
            {"string": 4, "fret": 5, "pitch": 64, "technique": "pick", "column": 0},
        ]
        with pytest.raises(TabValidationError, match="more than one note"):
            validate_tab(tab, GUITAR_TUNING, 24)

    def test_rejects_pitch_string_mismatch(self) -> None:
        """A Drop-D tab that calls its D string E is caught here."""
        tab = [{"string": 0, "fret": 0, "pitch": 40, "technique": "pick", "column": 0}]
        with pytest.raises(TabValidationError, match="but the note is pitch"):
            validate_tab(tab, DROP_D_TUNING, 24)

    def test_rejects_unknown_string(self) -> None:
        tab = [{"string": 6, "fret": 0, "technique": "pick", "column": 0}]
        with pytest.raises(TabValidationError, match="instrument has"):
            validate_tab(tab, GUITAR_TUNING, 24)

    def test_rejects_fret_out_of_range(self) -> None:
        tab = [{"string": 0, "fret": 25, "technique": "pick", "column": 0}]
        with pytest.raises(TabValidationError, match="outside"):
            validate_tab(tab, GUITAR_TUNING, 24)

    def test_rejects_hand_span(self) -> None:
        tab = [
            {"string": 0, "fret": 2, "pitch": 42, "technique": "pick", "column": 0},
            {"string": 5, "fret": 15, "pitch": 79, "technique": "pick", "column": 0},
        ]
        with pytest.raises(TabValidationError, match="stretch"):
            validate_tab(tab, GUITAR_TUNING, 24)

    def test_rejects_unknown_technique(self) -> None:
        tab = [{"string": 0, "fret": 0, "pitch": 40, "technique": "tapping", "column": 0}]
        with pytest.raises(TabValidationError, match="unknown technique"):
            validate_tab(tab, GUITAR_TUNING, 24)

    def test_generate_tab_validates_its_own_output(self) -> None:
        agent = TabAgent(tuning=GUITAR_TUNING, num_frets=24)
        agent.generate_tab(chord(40, 59, 64))
        assert agent.last_validation is not None
        assert agent.last_validation.columns == 1
        assert agent.last_validation.notes == 3


class TestTechniqueHonesty(unittest.TestCase):
    """Technique labels must describe something that happened."""

    def setUp(self) -> None:
        self.agent = TabAgent(tuning=GUITAR_TUNING, num_frets=24)

    def test_simultaneous_notes_are_never_legato(self) -> None:
        """A chord is one attack: no hammer/pull/slide inside a column."""
        tab = self.agent.generate_tab(chord(40, 47, 52))
        assert all(r["technique"] == "pick" for r in tab)

    def test_one_fret_move_is_not_a_slide(self) -> None:
        tab = self.agent.generate_tab([make_note(40, 0.0), make_note(41, 0.12)])
        assert [r["technique"] for r in tab] == ["pick", "hammer"]

    def test_three_fret_move_is_a_slide(self) -> None:
        tab = self.agent.generate_tab([make_note(40, 0.0), make_note(43, 0.12)])
        assert [r["technique"] for r in tab] == ["pick", "slide"]

    def test_descending_move_is_a_pull_off(self) -> None:
        tab = self.agent.generate_tab([make_note(43, 0.0), make_note(42, 0.12)])
        assert tab[-1]["technique"] == "pull"

    def test_slow_move_is_a_pick(self) -> None:
        tab = self.agent.generate_tab([make_note(40, 0.0), make_note(43, 1.2)])
        assert [r["technique"] for r in tab] == ["pick", "pick"]

    def test_only_the_new_note_carries_the_technique(self) -> None:
        """The first note of a legato pair is picked, not hammered."""
        tab = self.agent.generate_tab([make_note(40, 0.0), make_note(43, 0.12)])
        assert tab[0]["technique"] == "pick"
        assert tab[1]["technique"] == "slide"


class TestSkippedNoteReporting(unittest.TestCase):
    """Skipping must be reported, not silent."""

    def test_skipped_columns_report_time_and_reason(self) -> None:
        agent = TabAgent(tuning=GUITAR_TUNING, num_frets=24)
        agent.generate_tab([make_note(40, 0.0), make_note(10, 0.5)])
        assert len(agent.last_skipped_columns) == 1
        entry = agent.last_skipped_columns[0]
        assert entry["time"] == 0.5
        assert entry["dropped"] == [10]
        assert entry["reason"]

    def test_skipped_notes_are_pitches_not_indices(self) -> None:
        agent = TabAgent(tuning=GUITAR_TUNING, num_frets=24)
        agent.generate_tab([make_note(120, 0.0)])
        assert agent.last_skipped_notes == [120]
        assert agent.last_validation is None


class TestReadmeSample(unittest.TestCase):
    """The README's sample tab must be a tab no hand would have to fake."""

    def _sample_block(self) -> list[str]:
        readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
        block = readme.split("=== Lead Guitar Tablature ===")[1].split("```")[0]
        return [ln for ln in block.splitlines() if "|" in ln]

    def test_sample_has_six_labelled_rows(self) -> None:
        rows = self._sample_block()
        assert len(rows) == 6
        assert [ln[0] for ln in rows] == list("EBGDAE")

    def test_sample_never_stacks_notes_on_one_string(self) -> None:
        """`15/19h` was real output from the old grouping: unplayable."""
        for row in self._sample_block():
            cells = row.split("|")[1].split()
            for cell in cells:
                assert "/" not in cell, row
                # `a+b` is only emitted for genuinely distinct strings.
                assert "+" not in cell, row

    def test_sample_parses_back_into_a_playable_tab(self) -> None:
        """Read the advertised grid back and validate it as tablature.

        If the README ever shows a chord with a 12-fret stretch or two notes on
        one string, this fails — the sample is a factual claim about output.
        """
        tab = []
        for string, row in enumerate(self._sample_block()):
            cells = row.split("|")[1].split()
            for column, cell in enumerate(cells):
                if cell == "-":
                    continue
                frank = "".join(ch for ch in cell if ch.isdigit())
                if not frank:
                    continue
                fret = int(frank)
                tab.append(
                    {
                        "string": string,
                        "fret": fret,
                        "pitch": GUITAR_TUNING[string] + fret,
                        "technique": "pick",
                        "column": column,
                    }
                )
        assert tab, "sample parsed to nothing"
        report = validate_tab(tab, GUITAR_TUNING, 24, allow_techniques=False)
        assert report.columns == 12
        assert report.widest_span <= 5


if __name__ == "__main__":
    unittest.main()
