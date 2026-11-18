"""Fretboard invariants and explicit open polyphony regressions."""

from itertools import product

import pytest

from agents import TabAgent
from tests.audio_fixtures import BASS_4, BASS_5, DROP_D, STANDARD, make_note, tab_pitches


def test_viterbi_matches_exhaustive_global_cost():
    agent = TabAgent(STANDARD)
    # The cheapest second note from the low E string is open B, but the G
    # string's fourth fret makes the path as a whole cheaper.
    pitches = [41, 59, 69]
    starts = [0.0, 0.125, 0.5]
    notes = [make_note(p, t, t + 0.10) for p, t in zip(pitches, starts, strict=True)]
    candidates = [
        [
            {"string": i, "fret": p - open_pitch}
            for i, open_pitch in enumerate(STANDARD)
            if 0 <= p - open_pitch <= 24
        ]
        for p in pitches
    ]
    deltas = [starts[1] - starts[0], starts[2] - starts[1]]

    def path_cost(path):
        return sum(
            agent.calculate_cost(prev, curr, dt)
            for prev, curr, dt in zip(path[:-1], path[1:], deltas, strict=True)
        )

    greedy = [candidates[0][0]]
    for options, dt in zip(candidates[1:], deltas, strict=True):
        greedy.append(min(options, key=lambda curr: agent.calculate_cost(greedy[-1], curr, dt)))
    paths = list(product(*candidates))
    optimum = min(path_cost(path) for path in paths)
    assert optimum < path_cost(greedy), "fixture must distinguish Viterbi from greedy"

    def last_transition(path):
        return agent.calculate_cost(path[-2], path[-1], deltas[-1])

    best_last = min(map(last_transition, paths))
    assert optimum < min(path_cost(path) for path in paths if last_transition(path) == best_last)

    tab = agent.generate_tab(notes)
    assert tab_pitches(tab, STANDARD, 24) == pitches
    assert [p["start_time"] for p in tab] == starts
    assert path_cost(tab) == pytest.approx(optimum)


@pytest.mark.parametrize(
    ("tuning", "frets"),
    [(STANDARD, 19), (DROP_D, 24), (BASS_4, 24), (BASS_5, 24)],
    ids=["classical", "drop-d", "bass-4", "bass-5"],
)
def test_fret_boundaries_keep_playable_notes_after_a_rejected_event(tuning, frets):
    low, high = min(tuning), max(tuning) + frets
    pitches = [low - 1, low, high + 1, high, low + 1]
    notes = [make_note(p, i / 2, i / 2 + 0.25) for i, p in enumerate(pitches)]
    agent = TabAgent(tuning, num_frets=frets)
    tab = agent.generate_tab(notes[::-1])  # exercise chronology, not just range filtering

    assert tab_pitches(tab, tuning, frets) == [low, high, low + 1]
    assert [p["start_time"] for p in tab] == [0.5, 1.5, 2.0]
    assert [(p["string"], p["fret"]) for p in tab[:2]] == [(0, 0), (len(tuning) - 1, frets)]
    assert agent.last_skipped_notes == [low - 1, high + 1]


@pytest.mark.parametrize(
    "events",
    [
        [(41, 0, 0.10), (42, 0.19, 0.40)],  # a real rest, but inside onset window
        [(41, 0, 0.40), (42, 0, 0.40)],  # simultaneity, not a slide
        [(41, 0, 0.15), (39, 0.1, 0.30), (42, 0.16, 0.40)],  # rejected intermediary
        [(41, 0, 0.07), (41, 0.08, 0.40)],  # same pitch reattack
    ],
    ids=["rest", "simultaneous", "skipped-intermediary", "repeated-pick"],
)
def test_no_legato_annotation_without_a_plausible_transition(events):
    agent = TabAgent(STANDARD)
    tab = agent.generate_tab([make_note(*event) for event in events], technique_sensitivity=0.9)
    assert tab_pitches(tab, STANDARD, 24) == [p for p, _, _ in events if p >= 40]
    assert [p["technique"] for p in tab] == ["pick", "pick"]


class StringOccupancyError(AssertionError):
    """Expected only for a known string-occupancy defect, not missing or wrong notes."""


@pytest.mark.xfail(
    strict=True, raises=StringOccupancyError, reason="sequential DP ignores chord occupancy"
)
def test_playable_chord_keeps_all_pitches_on_distinct_strings():
    # All three notes have an easy simultaneous assignment: the open E, A, D strings.
    pitches = [40, 45, 50]
    tab = TabAgent(STANDARD).generate_tab([make_note(p, 0, 0.5) for p in pitches])
    assert sorted(tab_pitches(tab, STANDARD, 24)) == pitches
    if len({p["string"] for p in tab}) != 3:
        raise StringOccupancyError(tab)


@pytest.mark.xfail(
    strict=True, raises=StringOccupancyError, reason="sequential DP ignores active sustains"
)
def test_sustained_string_cannot_be_reused_during_a_later_note():
    # A2 can be played open, freeing the only string that can play E2 later.
    notes = [make_note(45, 0, 0.7), make_note(40, 0.18, 0.5)]
    tab = TabAgent(STANDARD).generate_tab(notes)
    assert tab_pitches(tab, STANDARD, 24) == [45, 40]
    if tab[0]["string"] == tab[1]["string"]:
        raise StringOccupancyError(tab)
    assert tab[1]["technique"] == "pick"


@pytest.mark.xfail(
    strict=True, raises=StringOccupancyError, reason="no rejection policy for impossible chords"
)
def test_impossible_same_string_dyad_reports_the_conflict():
    # Neither pitch can be moved off low E. At most one can be represented as tab.
    agent = TabAgent(STANDARD)
    tab = agent.generate_tab([make_note(p, 0, 0.5) for p in (40, 41)])
    assigned = tab_pitches(tab, STANDARD, 24)
    assert sorted(assigned + agent.last_skipped_notes) == [40, 41]
    if len(assigned) != 1:
        raise StringOccupancyError(tab)
