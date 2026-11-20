#!/usr/bin/env python3
"""Reproduce the tab-accuracy probes in docs/SUBSTANTIATION_LOG.md.

`logic` uses known note events (no audio or inference). `fixtures DIR` writes
new synthetic WAVs and their reference events. `score` compares a CLI tab JSON
with a supplied synthetic reference or an AlpacaGuitarSet GM-token track.
None of these is a certification or a replacement for a held-out benchmark.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

TUNINGS = {
    "guitar": (40, 45, 50, 55, 59, 64),
    "drop-d": (38, 45, 50, 55, 59, 64),
    "bass4": (28, 33, 38, 43),
    "bass5": (23, 28, 33, 38, 43),
}


def note(pitch: int, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(pitch=pitch, start_time=start, end_time=end, velocity=80)


def check_logic() -> None:
    from agents import EarAgent, TabAgent
    from main import render_tab_text

    seed = 4814
    rng = random.Random(seed)
    trials = 0
    for tuning in TUNINGS.values():
        for prefer_low_strings in (False, True):
            agent = TabAgent(list(tuning), prefer_low_strings=prefer_low_strings)
            for _ in range(16):
                times = [0.0]
                for _ in range(rng.randrange(1, 5)):
                    times.append(times[-1] + rng.choice((0.0625, 0.125, 0.25, 0.5)))
                notes = [note(rng.randint(tuning[0], tuning[-1] + 24), t, t + 0.05) for t in times]
                # Independent enumeration of feasible positions and complete paths;
                # use the *production scoring function* to test optimality, not ergonomics.
                candidates = [
                    [
                        {"string": i, "fret": n.pitch - base}
                        for i, base in enumerate(tuning)
                        if 0 <= n.pitch - base <= 24
                    ]
                    for n in notes
                ]

                def cost(path, *, agent=agent, times=tuple(times)):
                    return sum(
                        agent.calculate_cost(a, b, t1 - t0)
                        for a, b, t0, t1 in zip(
                            path[:-1], path[1:], times[:-1], times[1:], strict=True
                        )
                    )

                tab = agent.generate_tab(notes)
                assert len(tab) == len(notes)
                assert all(
                    0 <= p["string"] < len(tuning)
                    and 0 <= p["fret"] <= 24
                    and tuning[p["string"]] + p["fret"] == n.pitch
                    for p, n in zip(tab, notes, strict=True)
                )
                assert cost(tab) == min(map(cost, itertools.product(*candidates)))
                trials += 1
    print(f"Exhaustive per-note cost oracle: {trials}/{trials} passed (seed {seed})")

    def display(name, observations, agent):
        tab = agent.generate_tab(observations)
        print(
            name,
            [
                (agent.tuning[p["string"]] + p["fret"], p["string"], p["fret"], p["technique"])
                for p in tab
            ],
        )
        return tab

    guitar = TabAgent(list(TUNINGS["guitar"]))
    pitches = (40, 47, 52, 56, 59, 64)  # E major: playable as 0-2-2-1-0-0 (low to high)
    chord = display("E major", [note(p, 0, 0.6) for p in pitches], guitar)
    display("Same E major, reversed input", [note(p, 0, 0.6) for p in reversed(pitches)], guitar)
    print(render_tab_text(chord, num_strings=6))
    seven = display("Seven simultaneous notes", [note(p, 0, 0.6) for p in (*pitches, 67)], guitar)
    print(f"Over-capacity group: {len(seven)} emitted, {len(guitar.last_skipped_notes)} skipped")
    display(
        "Overlapping A2+B2 (can use distinct strings)",
        [note(45, 0, 0.65), note(47, 0.15, 0.45)],
        guitar,
    )
    display(
        "All four are picked, with gaps",
        [note(p, i * 0.15, i * 0.15 + 0.08) for i, p in enumerate((40, 42, 46, 42))],
        guitar,
    )

    early = guitar.generate_tab([note(40, 0, 0.1), note(42, 0.5, 0.6)])
    late = guitar.generate_tab([note(40, 0, 0.4), note(42, 5, 5.5)])
    early_text = render_tab_text(early, num_strings=6)
    late_text = render_tab_text(late, num_strings=6)
    assert early_text is not None and late_text is not None
    print(
        "Different rhythms render identically:",
        early_text.splitlines()[:-1] == late_text.splitlines()[:-1],
    )

    drop = TabAgent(list(TUNINGS["drop-d"]))
    d2 = display("Drop D, open D2", [note(38, 0, 0.5)], drop)
    print(render_tab_text(d2, num_strings=6))
    raised = TabAgent([midi + 2 for midi in TUNINGS["guitar"]])
    high_note = note(90, 0, 0.5)
    print("Raised tuning: MIDI 90 positions", raised.get_valid_positions(90))
    ear = EarAgent(device="cpu")
    print(
        "Same pitch after EarAgent range filter",
        ear._filter_by_instrument_range([high_note], "Guitar", min_pitch=42),
    )


def make_fixtures(folder: Path) -> None:
    import numpy as np
    import soundfile as sf

    folder.mkdir(parents=True, exist_ok=True)
    sample_rate = 44100
    for name, pitches in {"drop-d": (38, 45, 38, 42), "low-b": (23, 28, 23, 35)}.items():
        audio = np.zeros(3 * sample_rate)
        reference = []
        for index, pitch in enumerate(pitches):
            start = 0.25 + index * 0.625
            duration = 0.44
            frequency = 440 * 2 ** ((pitch - 69) / 12)
            t = np.arange(round(duration * sample_rate)) / sample_rate
            envelope = (
                np.minimum(t / 0.008, 1) * np.exp(-t * 5) * np.clip((duration - t) / 0.02, 0, 1)
            )
            tone = sum(np.sin(2 * np.pi * frequency * k * t) / k**1.2 for k in range(1, 9))
            offset = round(start * sample_rate)
            audio[offset : offset + len(t)] = 0.25 * envelope * tone
            reference.append({"pitch": pitch, "start_time": start})
        sf.write(folder / f"{name}.wav", audio, sample_rate)
        (folder / f"{name}.ref.json").write_text(json.dumps(reference, indent=2) + "\n")
        print(f"Wrote {name}.wav and {name}.ref.json to {folder}")


@dataclass(frozen=True)
class Event:
    time: float
    pitch: int
    string: int | None = None


def load_reference(ref: Path, num_strings: int) -> list[Event]:
    if ref.suffix == ".json":
        return [Event(x["start_time"], x["pitch"]) for x in json.loads(ref.read_text())]
    if ref.suffix != ".txt":
        raise ValueError("Reference must be synthetic .ref.json or a GM-token track (.txt)")
    out = []
    for line in ref.read_text().splitlines():
        start, _end, *tokens, _note_type = line.split()
        if len(tokens) != num_strings:
            raise ValueError(f"Unexpected string count in {ref}: {line}")
        for i, token in enumerate(tokens):
            pitch = int(token.split("-", 1)[0])
            if pitch:
                out.append(Event(float(start), pitch, num_strings - i - 1))
    return out


def group_counts(tab: list[dict], window: float = 0.05) -> tuple[int, int, int, int]:
    """Use 50 ms buckets to compare against the audited version's renderer."""
    groups: list[list[dict]] = []
    for row in tab:
        if groups and abs(row["start_time"] - groups[-1][0]["start_time"]) < window:
            groups[-1].append(row)
        else:
            groups.append([row])
    collisions = sum(len({row["string"] for row in group}) < len(group) for group in groups)
    exact_onset_collisions = len(tab) - len({(p["start_time"], p["string"]) for p in tab})
    return len(groups), collisions, exact_onset_collisions, max(map(len, groups), default=0)


def count_matches(
    ref: list[Event], pred: list[Event], tolerance: float, *, strings: bool = False
) -> int:
    """One-to-one fixed-window onset matching per pitch (or pitch+string)."""
    ref_by_pitch: dict[tuple[int, int | None], list[float]] = defaultdict(list)
    pred_by_pitch: dict[tuple[int, int | None], list[float]] = defaultdict(list)
    for events, by_pitch in ((ref, ref_by_pitch), (pred, pred_by_pitch)):
        for event in events:
            by_pitch[(event.pitch, event.string if strings else None)].append(event.time)
    matched = 0
    for key, times in ref_by_pitch.items():
        right = iter(sorted(pred_by_pitch.get(key, ())))
        found = next(right, None)
        for time in sorted(times):
            while found is not None and found < time - tolerance:
                found = next(right, None)
            if found is not None and found <= time + tolerance:
                matched += 1
                found = next(right, None)
    return matched


def score(path: Path, tuning: tuple[int, ...], ref_file: Path | None, onset_ms: float) -> None:
    data = json.loads(path.read_text())
    if data.get("num_strings") != len(tuning):
        raise ValueError(f"Selected tuning ({len(tuning)} strings) does not match {path}")
    tab = data["tablature"]
    pred = [Event(x["start_time"], tuning[x["string"]] + x["fret"], x["string"]) for x in tab]
    groups, collisions, exact, largest = group_counts(tab)
    print(
        f"{path.name}: notes={len(tab)}, onset_groups={groups}, largest_group={largest}, "
        f"groups_with_string_collisions={collisions}, excess_events_on_same_string/exact_onset={exact}"
    )
    print("  technique counts:", dict(sorted(Counter(x["technique"] for x in tab).items())))
    if ref_file is not None:
        ref = load_reference(ref_file, len(tuning))
        hits = count_matches(ref, pred, onset_ms / 1000)
        denom = len(pred) + len(ref)
        print(
            f"  reference={len(ref)}, matches={hits}, extra={len(pred) - hits}, "
            f"missed={len(ref) - hits}, F1={2 * hits / denom if denom else 0:.3f} (±{onset_ms:g} ms)"
        )
        if ref and all(x.string is not None for x in ref):
            correct_string = count_matches(ref, pred, onset_ms / 1000, strings=True)
            print(f"  matches additionally requiring the reference string: {correct_string}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("logic")
    fixture = modes.add_parser("fixtures")
    fixture.add_argument("folder", type=Path)
    scoring = modes.add_parser("score")
    scoring.add_argument("tab", type=Path)
    scoring.add_argument("--tuning", choices=TUNINGS, required=True)
    scoring.add_argument("--reference", type=Path)
    scoring.add_argument("--onset-ms", type=float, default=50.0)
    args = parser.parse_args()
    if args.mode == "logic":
        check_logic()
    elif args.mode == "fixtures":
        make_fixtures(args.folder)
    else:
        score(args.tab, TUNINGS[args.tuning], args.reference, args.onset_ms)


if __name__ == "__main__":
    main()
