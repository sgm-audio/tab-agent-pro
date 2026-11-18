# ruff: noqa
"""
Reference replacement for TabAgent.generate_tab (see RELIABILITY_REVIEW.md, T-1/T-2).

Differences from the shipped implementation:
  * notes are sorted deterministically by (onset, pitch);
  * notes that start within CHORD_EPS are treated as ONE chord and assigned to
    *distinct strings* with a feasible hand span, instead of being run through a
    linear chain that is allowed to put two simultaneous notes on one string;
  * there is an absolute hand-position term and no negative cost, so the chosen
    voicing does not drift to fret 19 when the open position is available;
  * technique annotation only fires between *single* successive notes that
    actually overlap in time (legato), is opt-in (annotate_techniques=False by
    default), and never emits "slide": Basic Pitch is called with
    multiple_pitch_bends=False, so there is no bend evidence to support one.
"""

from itertools import product

CHORD_EPS = 0.05
TECHNIQUE_MAX_INTERVAL = 4  # frets for a hammer/pull
MAX_VOICINGS = 24
MAX_SPAN = 5


def _positions(agent, pitch):
    return agent.get_valid_positions(pitch)


def _voicings(agent, pitches, max_span=MAX_SPAN):
    """All playable assignments: one position per pitch, distinct strings,
    frets within ``max_span`` of each other (open strings are always allowed)."""
    per_note = [_positions(agent, p) for p in pitches]
    if any(not p for p in per_note):
        return []
    out = []
    for combo in product(*per_note):
        strings = [c["string"] for c in combo]
        if len(set(strings)) != len(strings):
            continue
        fretted = [c["fret"] for c in combo if c["fret"] > 0]
        if fretted and max(fretted) - min(fretted) > max_span:
            continue
        out.append(list(combo))
    return out


def _hand(v):
    fretted = [p["fret"] for p in v if p["fret"] > 0]
    return sum(fretted) / len(fretted) if fretted else 0.0


def _voicing_score(v, prev):
    # Prefer the open position and low frets, then compact spans.
    score = _hand(v) * 1.0
    if prev is not None:
        score += abs(_hand(v) - _hand(prev)) * 1.5
        moved = sum(1 for a, b in zip(prev, v) if a["string"] != b["string"])
        score += moved * 0.75
    return score


def generate_tab(agent, midi_notes, technique_sensitivity=0.7, annotate_techniques=False):
    agent.last_skipped_notes = []
    if not midi_notes:
        return []

    notes = sorted(
        (
            {"pitch": int(n.pitch), "start": float(n.start_time), "end": float(n.end_time)}
            for n in midi_notes
        ),
        key=lambda n: (round(n["start"], 3), n["pitch"]),
    )

    # 1. group simultaneous notes into chords
    groups = []
    for note in notes:
        if groups and abs(note["start"] - groups[-1][0]["start"]) <= CHORD_EPS:
            groups[-1].append(note)
        else:
            groups.append([note])

    # 2. enumerate playable voicings per chord; drop unplayable notes and report
    layers = []
    for group in groups:
        pitches = [n["pitch"] for n in group]
        voicings = _voicings(agent, pitches)
        if not voicings:
            # drop only the notes that are unplayable, keep the rest of the chord
            keep = [n for n in group if _positions(agent, n["pitch"])]
            agent.last_skipped_notes += [n["pitch"] for n in group if n not in keep]
            if not keep:
                continue
            group = keep
            voicings = _voicings(agent, [n["pitch"] for n in group])
            if not voicings:
                continue
        layers.append((group, voicings[:MAX_VOICINGS]))

    if not layers:
        return []

    # 3. Viterbi over chord voicings (one state per playable voicing)
    costs = [[_voicing_score(v, None) for v in layers[0][1]]]
    back = []
    for i in range(1, len(layers)):
        prev_v, prev_costs = layers[i - 1][1], costs[-1]
        row, ptr = [], []
        for v in layers[i][1]:
            best, best_j = float("inf"), 0
            for j, pv in enumerate(prev_v):
                c = prev_costs[j] + _voicing_score(v, pv)
                if c < best:
                    best, best_j = c, j
            row.append(best)
            ptr.append(best_j)
        costs.append(row)
        back.append(ptr)

    best = min(range(len(costs[-1])), key=lambda i: costs[-1][i])
    chosen = [best]
    for i in range(len(layers) - 2, -1, -1):
        best = back[i][best]
        chosen.append(best)
    chosen.reverse()

    # 4. flatten to positions, annotate only genuine single-note legato
    window = 0.05 + technique_sensitivity * 0.25
    out = []
    prev_single = None
    for (group, voicings), idx in zip(layers, chosen):
        v = voicings[idx]
        is_single = len(group) == 1
        for note, pos in zip(group, v):
            item = {
                "string": pos["string"],
                "fret": pos["fret"],
                "technique": "pick",
                "start_time": note["start"],
            }
            if annotate_techniques and is_single and prev_single is not None:
                prev_note, prev_pos = prev_single
                gap = note["start"] - prev_note["start"]
                legato = prev_note["end"] >= note["start"] - 0.02
                interval = abs(pos["fret"] - prev_pos["fret"])
                same_string = pos["string"] == prev_pos["string"]
                if same_string and 0 <= gap < window and legato and interval:
                    # No "slide": Basic Pitch runs with multiple_pitch_bends=False,
                    # so no bend evidence exists to support one.
                    if interval <= TECHNIQUE_MAX_INTERVAL:
                        item["technique"] = "hammer" if pos["fret"] > prev_pos["fret"] else "pull"
            out.append(item)
        prev_single = (group[0], v[0]) if is_single else None
    return out
