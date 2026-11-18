"""Check the solver's occupancy and omission contracts across chord inputs.

The hard string-occupancy rule, not the cost of a transition, must prevent
simultaneous notes from using one string twice.
"""

from itertools import combinations

import pytest

from agents import TabAgent
from tests.audio_fixtures import STANDARD, make_note, tab_pitches

PITCH_RANGE = range(40, 65)  # standard guitar's lowest note to highest open string


def _check_chord(pitches: tuple[int, ...]) -> int:
    agent = TabAgent(STANDARD)
    tab = agent.generate_tab([make_note(pitch, 0, 0.5) for pitch in pitches])
    played = tab_pitches(tab, STANDARD, 24)
    assert 0 < len(tab) == len({pos["string"] for pos in tab}), (pitches, tab)
    assert all(event["reason"] == "fingering_conflict" for event in agent.last_skipped_events)
    reported = [event["pitch"] for event in agent.last_skipped_events]
    assert sorted(played + reported) == sorted(pitches), (pitches, played, reported)
    return len(tab)


def test_no_dyad_ever_shares_a_string():
    for pitches in combinations(PITCH_RANGE, 2):
        _check_chord(pitches)


@pytest.mark.parametrize(
    "shape", [(0, 4, 7), (0, 3, 7), (0, 5, 9)], ids=["major", "minor", "inversion"]
)
def test_no_triad_ever_shares_a_string(shape):
    for root in range(40, 61):
        _check_chord(tuple(root + interval for interval in shape))


def test_over_full_chord_never_exceeds_the_string_count():
    # Six strings: keep six pitches and explicitly account for the remaining one.
    assert _check_chord((40, 45, 50, 55, 59, 64, 67)) == 6
