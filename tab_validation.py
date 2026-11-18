"""
Tablature validity checks.

A tab is *valid* when every position it prints can actually be played on the
instrument it names, and when two notes that are printed in one column can
sound at the same time. Those are not stylistic preferences — a tab that
violates them is wrong, and a player cannot tell it is wrong by reading it.

These checks exist so that the invariants the fingering model relies on are
enforced in code rather than in a comment. They are cheap enough to run on
every generated tab, and :func:`validate_tab` is wired into
``TabAgent.generate_tab`` so a regression fails loudly instead of shipping a
tab nobody can play.

Every check raises :class:`TabValidationError` with the offending entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Techniques a rendered fret suffix is derived from.
VALID_TECHNIQUES = frozenset({"pick", "slide", "hammer", "pull"})


class TabValidationError(ValueError):
    """Raised when a tab position is not physically playable."""


@dataclass
class TabValidationReport:
    """Summary of a successful validation, for tests and diagnostics."""

    columns: int = 0
    notes: int = 0
    widest_span: int = 0
    techniques: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "columns": self.columns,
            "notes": self.notes,
            "widest_span": self.widest_span,
            "techniques": dict(self.techniques),
        }


def validate_tab(
    tab_data: list[dict],
    tuning: list[int],
    num_frets: int,
    *,
    max_hand_span: int = 5,
    allow_techniques: bool = True,
) -> TabValidationReport:
    """
    Assert that a tab is playable on ``tuning``.

    Checks, per position:
      * ``string`` indexes a real string on the instrument;
      * ``0 <= fret <= num_frets``;
      * when the entry carries a ``pitch``, ``tuning[string] + fret == pitch``
        (this is the check that catches a wrong string/fret assignment, and the
        one that would have caught the Drop-D mislabelling);
      * ``technique`` is one of the four known values.

    Checks, per column (positions sharing a ``column`` value, or sharing an
    onset when ``column`` is absent):
      * no two notes are on the same string (a string sounds one note at a
        time, so this is the check that catches chord-as-arpeggio fingering);
      * the fretted notes fit within ``max_hand_span`` frets.

    Args:
        tab_data: positions from ``TabAgent.generate_tab``
        tuning: open-string MIDI notes, lowest string first
        num_frets: highest fret on the instrument
        max_hand_span: widest stretch a hand can hold, in frets
        allow_techniques: set False to ignore technique values (for callers
            that pass position-only dicts)

    Returns:
        TabValidationReport describing what was checked.

    Raises:
        TabValidationError: on the first violation, naming the entry.
    """
    if not tab_data:
        return TabValidationReport()

    if not tuning:
        raise TabValidationError("cannot validate a tab with an empty tuning")

    num_strings = len(tuning)
    report = TabValidationReport(notes=len(tab_data))

    for index, position in enumerate(tab_data):
        if not isinstance(position, dict):
            raise TabValidationError(f"entry {index} is not a dict: {position!r}")
        for key in ("string", "fret"):
            if key not in position:
                raise TabValidationError(f"entry {index} has no {key!r}: {position!r}")

        string = int(position["string"])
        fret = int(position["fret"])

        if not 0 <= string < num_strings:
            raise TabValidationError(
                f"entry {index} uses string {string}, but the instrument has "
                f"{num_strings} strings (0-{num_strings - 1})"
            )
        if not 0 <= fret <= num_frets:
            raise TabValidationError(f"entry {index} uses fret {fret}, outside 0-{num_frets}")
        if position.get("pitch") is not None:
            expected = tuning[string] + fret
            if int(position["pitch"]) != expected:
                raise TabValidationError(
                    f"entry {index} says string {string} fret {fret}, which is "
                    f"pitch {expected}, but the note is pitch {position['pitch']}"
                )
        if allow_techniques and "technique" in position:
            technique = str(position["technique"])
            if technique not in VALID_TECHNIQUES:
                raise TabValidationError(
                    f"entry {index} has unknown technique {technique!r}; "
                    f"expected one of {sorted(VALID_TECHNIQUES)}"
                )
            report.techniques[technique] = report.techniques.get(technique, 0) + 1

    for column_key, members in _columns(tab_data).items():
        report.columns += 1
        strings = [int(p["string"]) for p in members]
        if len(set(strings)) != len(strings):
            repeated = sorted({s for s in strings if strings.count(s) > 1})
            raise TabValidationError(
                f"column {column_key} puts more than one note on string(s) {repeated}: "
                f"a string cannot sound two notes at once"
            )
        fretted = [int(p["fret"]) for p in members if int(p["fret"]) > 0]
        if len(fretted) > 1:
            span = max(fretted) - min(fretted)
            report.widest_span = max(report.widest_span, span)
            if span > max_hand_span:
                raise TabValidationError(
                    f"column {column_key} needs a {span}-fret stretch "
                    f"(frets {min(fretted)}-{max(fretted)}), wider than the "
                    f"{max_hand_span}-fret hand span"
                )

    return report


def _columns(tab_data: list[dict]) -> dict[object, list[dict]]:
    """Group positions into columns by explicit ``column``, else by onset."""
    columns: dict[object, list[dict]] = {}
    for position in tab_data:
        key = position.get("column")
        if key is None:
            key = round(float(position.get("start_time", 0.0)), 3)
        columns.setdefault(key, []).append(position)
    return columns
