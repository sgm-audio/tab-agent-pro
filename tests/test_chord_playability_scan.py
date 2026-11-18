"""Exhaustive scan: no two simultaneous notes may share a string.

The chord fix is expressed upstream as a large cost penalty rather than a hard
constraint, so it holds only while the penalty dominates every other term in
`calculate_cost`. These scans make that assumption checkable, and would catch a
regression if the cost weights are ever retuned.
"""

from itertools import combinations

import pytest

from agents import TabAgent
from tests.audio_fixtures import STANDARD, make_note, tab_pitches

FRETBOARD = range(40, 65)  # everything reachable on a 24-fret standard guitar


def _simultaneous(notes: list) -> list:
    """Notes sharing an onset, i.e. a chord that must occupy distinct strings."""
    assert notes, "fixture produced no tab"
    return notes


def test_no_dyad_ever_shares_a_string():
    collisions = []
    silent = []
    for low, high in combinations(FRETBOARD, 2):
        agent = TabAgent(STANDARD)
        tab = _simultaneous(agent.generate_tab([make_note(low, 0, 0.5), make_note(high, 0, 0.5)]))
        if len({pos["string"] for pos in tab}) != len(tab):
            collisions.append((low, high))
        played = tab_pitches(tab, STANDARD, 24)
        reported = {event["pitch"] for event in getattr(agent, "last_skipped_events", [])}
        unexplained = [p for p in (low, high) if p not in played and p not in reported]
        if unexplained:
            silent.append((low, high, unexplained))
    assert collisions == [], f"{len(collisions)} dyads stacked on one string"
    assert silent == [], f"{len(silent)} dyads lost a pitch without reporting it"


@pytest.mark.parametrize("shape", [(0, 4, 7), (0, 3, 7), (0, 5, 9)], ids=["major", "minor", "6th"])
def test_no_triad_ever_shares_a_string(shape):
    collisions = []
    for root in range(40, 61):
        pitches = [root + interval for interval in shape]
        if pitches[-1] > max(STANDARD) + 24:
            continue
        agent = TabAgent(STANDARD)
        tab = _simultaneous(agent.generate_tab([make_note(p, 0, 0.5) for p in pitches]))
        if len({pos["string"] for pos in tab}) != len(tab):
            collisions.append(tuple(pitches))
    assert collisions == [], f"{len(collisions)} triads stacked on one string: {collisions[:3]}"


def test_over_full_chord_never_exceeds_the_string_count():
    # Seven simultaneous pitches on a six-string guitar: six may be written, and
    # the seventh must be reported rather than stacked onto an occupied string.
    agent = TabAgent(STANDARD)
    tab = agent.generate_tab([make_note(p, 0, 0.5) for p in [40, 45, 50, 55, 59, 64, 67]])

    assert len(tab) == 6
    assert len({pos["string"] for pos in tab}) == 6
    dropped = [event["pitch"] for event in getattr(agent, "last_skipped_events", [])]
    played = tab_pitches(tab, STANDARD, 24)
    assert sorted(played + dropped) == [40, 45, 50, 55, 59, 64, 67]
