"""Regressions for the impossible voicings in the evidence audit."""

import itertools
import json
import random
from types import SimpleNamespace

from agents import EarAgent, TabAgent
from main import export_tab_to_json, render_tab_text

STANDARD = [40, 45, 50, 55, 59, 64]


def note(pitch: int, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(pitch=pitch, start_time=start, end_time=end)


def assert_feasible(tab: list[dict], agent: TabAgent) -> None:
    for onset in {row["start_time"] for row in tab}:
        active = [
            row
            for row in tab
            if row["start_time"] == onset or row["start_time"] < onset < row["end_time"]
        ]
        assert len(active) == len({row["string"] for row in active})
        frets = [row["fret"] for row in active if row["fret"]]
        assert not frets or max(frets) - min(frets) <= agent.MAX_FRET_SPAN


def test_open_e_major_retains_six_strings_independent_of_input_order() -> None:
    agent = TabAgent(STANDARD)
    pitches = [40, 47, 52, 56, 59, 64]
    notes = [note(p, 0.0, 0.6) for p in pitches]
    tab = agent.generate_tab(notes)
    assert len(tab) == 6
    assert_feasible(tab, agent)
    assert [agent.tuning[row["string"]] + row["fret"] for row in tab] == pitches
    assert all(row["technique"] == "pick" for row in tab)
    assert agent.generate_tab(list(reversed(notes))) == tab
    assert agent.last_skipped_events == []


def test_search_keeps_a_string_free_for_a_note_arriving_during_sustain() -> None:
    agent = TabAgent(STANDARD)
    tab = agent.generate_tab([note(45, 0, 0.65), note(40, 0.15, 0.45)])
    assert [(row["string"], row["fret"], row["end_time"]) for row in tab] == [
        (1, 0, 0.65),
        (0, 0, 0.45),
    ]
    assert_feasible(tab, agent)


def test_seventh_simultaneous_note_is_reported_instead_of_overbooking_strings() -> None:
    agent = TabAgent(STANDARD)
    tab = agent.generate_tab([note(p, 0, 1) for p in (40, 47, 52, 56, 59, 64, 67)])
    assert len(tab) == 6
    assert_feasible(tab, agent)
    (omitted,) = agent.last_skipped_events
    assert omitted["start_time"] == 0 and omitted["end_time"] == 1
    assert omitted["pitch"] == agent.last_skipped_notes[0]
    assert omitted["reason"] == "fingering_conflict"


def test_overlapping_frets_outside_hand_span_are_not_combined() -> None:
    agent = TabAgent([40, 70])
    # Each note has a unique string; frets 1 and 10 cannot share the hand span.
    tab = agent.generate_tab([note(41, 0, 0.5), note(80, 0.1, 0.3)])
    assert len(tab) == 1
    assert agent.last_skipped_events[0]["reason"] == "fingering_conflict"


def test_instant_notes_at_the_same_onset_also_need_distinct_strings() -> None:
    agent = TabAgent(STANDARD)
    assert len(agent.generate_tab([note(40, 0, 0), note(42, 0, 0)])) == 1
    assert len(agent.last_skipped_events) == 1


def test_note_can_start_at_release_but_empty_gap_is_not_labeled_legato() -> None:
    agent = TabAgent(STANDARD)
    contiguous = agent.generate_tab([note(40, 0, 0.12), note(42, 0.12, 0.2)])
    gapped = agent.generate_tab([note(40, 0, 0.05), note(42, 0.12, 0.2)])
    assert len(contiguous) == len(gapped) == 2
    assert contiguous[1]["technique"] == "slide"
    assert gapped[1]["technique"] == "pick"


def test_small_polyphonic_search_matches_exhaustive_oracle() -> None:
    rng = random.Random(4814)
    agent = TabAgent([40, 45, 50], num_frets=8)
    for _ in range(24):
        events = sorted(
            (time, rng.randint(40, 58), time + rng.choice((0.125, 0.25)))
            for time in (0.0, 0.0, 0.125, 0.25)
        )
        choices = [
            [None] + [(s, p - base) for s, base in enumerate(agent.tuning) if 0 <= p - base <= 8]
            for _, p, _ in events
        ]
        best = (0, 0.0)
        for candidate in itertools.product(*choices):
            assigned = [
                (t, end, pos)
                for (t, _, end), pos in zip(events, candidate, strict=True)
                if pos is not None
            ]
            valid = True
            for t, _, _ in assigned:
                simultaneous = [pos for start, end, pos in assigned if start <= t < end]
                fretted = [f for _, f in simultaneous if f]
                if len({s for s, _ in simultaneous}) < len(simultaneous) or (
                    fretted and max(fretted) - min(fretted) > 4
                ):
                    valid = False
                    break
            if valid:
                cost = sum(
                    agent.calculate_cost(
                        {"string": s, "fret": f}, {"string": s2, "fret": f2}, t2 - t
                    )
                    for (t, _, (s, f)), (t2, _, (s2, f2)) in itertools.pairwise(assigned)
                )
                best = min(best, (-len(assigned), cost))
        tab = agent.generate_tab([note(p, t, end) for t, p, end in events])
        cost = sum(
            agent.calculate_cost(p, q, q["start_time"] - p["start_time"])
            for p, q in itertools.pairwise(tab)
        )
        assert_feasible(tab, agent)
        assert (-len(tab), cost) == best


def test_exports_retain_drop_d_timing_and_omissions(tmp_path) -> None:
    tuning = [38, 45, 50, 55, 59, 64]
    agent = TabAgent(tuning)
    tab = agent.generate_tab([note(38, 0.2, 0.4), note(37, 0.6, 0.8), note(38, 2.0, 2.4)])
    text = render_tab_text(
        tab, tuning=tuning, skipped_notes=agent.last_skipped_events, separation_mode="raw_fallback"
    )
    assert text is not None
    assert "D2 A2 D3 G3 B3 E4" in text
    d_rows = [line.strip() for line in text.splitlines() if line.lstrip().startswith("D|")]
    assert d_rows[-1].startswith("D|0")
    assert "t(s)|0.200 2.000" in text
    assert "1 detected note(s) omitted" in text
    assert "unseparated audio" in text
    path = tmp_path / "tab.json"
    export_tab_to_json(
        tab,
        path,
        tuning=tuning,
        skipped_notes=agent.last_skipped_events,
        separation_mode="raw_fallback",
    )
    data = json.loads(path.read_text())
    assert data["num_strings"] == 6 and data["tuning"] == tuning
    assert data["tablature"][0]["end_time"] == 0.4
    assert data["skipped_notes"][0]["reason"] == "out_of_range"
    assert data["technique_basis"] == "heuristic" and data["separation_mode"] == "raw_fallback"


def test_range_filter_uses_both_tuning_boundaries() -> None:
    ear = EarAgent(device="cpu")
    notes = [note(p, 0, 0.5) for p in (41, 42, 90, 91)]
    assert [n.pitch for n in ear._filter_by_instrument_range(notes, "Guitar", 42, 90)] == [42, 90]
    bass = [note(p, 0, 0.5) for p in (67, 69, 70)]
    assert [n.pitch for n in ear.humanize_and_clean(bass, is_bass=True, max_pitch=69)] == [67, 69]
