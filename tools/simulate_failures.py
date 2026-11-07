#!/usr/bin/env python3
"""
Failure-mode simulation harness (phase 4).

Exercises the degradation paths that static analysis cannot see, and asserts
the documented policy for each: every failure must be logged, typed, and either
recovered from or escalated — never swallowed.

Usage:
    python tools/simulate_failures.py      # exit 0 = all degradations behaved
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    RESULTS.append((ok, name, detail))


def _fresh_cwd() -> str:
    return tempfile.mkdtemp(prefix="tab-fail-")


# ---------------------------------------------------------------------------
# 1. Missing / corrupt configuration
# ---------------------------------------------------------------------------


def simulate_corrupt_user_memory() -> None:
    import main

    cwd = os.getcwd()
    tmp = _fresh_cwd()
    os.chdir(tmp)
    try:
        (Path(tmp) / "user_memory").mkdir()
        (Path(tmp) / "user_memory" / "user_preferences.json").write_text("{not json")
        _path, config = main.load_user_memory()
        check(
            config["guitar_tuning"] == [40, 45, 50, 55, 59, 64],
            "corrupt user_preferences.json falls back to defaults",
            str(config["guitar_tuning"]),
        )
    finally:
        os.chdir(cwd)


def simulate_invalid_thresholds() -> None:
    import main

    rejected = 0
    for onset, frame in ((99.0, 0.3), (0.5, -1.0), (float("nan"), 0.3)):
        try:
            main.validate_thresholds(onset, frame)
        except ValueError:
            rejected += 1
    check(rejected == 3, "out-of-range thresholds rejected", f"{rejected}/3")


def simulate_missing_env_file() -> None:
    import main

    with patch.dict(os.environ, {}, clear=False):
        main._load_dotenv(str(Path(_fresh_cwd()) / "absent.env"))
    check(True, "missing .env is a no-op (not a crash)")


# ---------------------------------------------------------------------------
# 2. Bad user input
# ---------------------------------------------------------------------------


def simulate_bad_audio() -> None:
    import app as appmod

    tmp = _fresh_cwd()
    bad = Path(tmp) / "bad.wav"
    bad.write_bytes(b"NOT-A-WAV")
    msg = appmod._validate_audio(bad)
    check(bool(msg), "undecodable audio is rejected", str(msg))
    check(
        "Cannot read audio" in (msg or ""),
        "undecodable audio message is actionable (not empty)",
        str(msg),
    )


def simulate_oversize_and_format() -> None:
    import app as appmod

    tmp = _fresh_cwd()
    big = Path(tmp) / "big.wav"
    big.write_bytes(b"\0" * (51 * 1024 * 1024))
    check("File too large" in (appmod._validate_audio(big) or ""), "50 MB cap enforced")

    txt = Path(tmp) / "song.txt"
    txt.write_bytes(b"hello")
    check("Unsupported format" in (appmod._validate_audio(txt) or ""), "extension allowlist")

    check(appmod._validate_audio(Path("/nonexistent/x.wav")) == "File not found", "missing file")


# ---------------------------------------------------------------------------
# 3. Backend availability
# ---------------------------------------------------------------------------


def simulate_no_backend_health() -> None:
    from monitoring import HealthTracker

    h = HealthTracker()
    h.set_status("up")
    h.set_component("basic_pitch", "unavailable")
    h.set_component("demucs", "unavailable")
    check(h.effective_status() == "down", "no backend -> status down", h.effective_status())
    check(h.is_ready() is False, "no backend -> not ready")


def simulate_transcription_backend_down() -> None:
    """transcribe_stem must raise (escalate) when no model is available."""
    import agents

    ear = agents.EarAgent.__new__(agents.EarAgent)
    ear.model = None
    ear.processor = None
    with patch.object(agents, "BASIC_PITCH_AVAILABLE", False), patch.object(agents, "health"):
        try:
            ear.transcribe_stem(str(ROOT / "examples" / "guitar_solo.wav"))
        except RuntimeError as e:
            check("No transcription models" in str(e), "no backend -> RuntimeError escalated")
        else:
            check(False, "no backend -> RuntimeError escalated", "no exception raised")


def simulate_demucs_offline() -> None:
    """Demucs failing must degrade to raw audio, not crash the pipeline."""
    import agents

    splitter = agents.SplitterAgent.__new__(agents.SplitterAgent)
    tmp = _fresh_cwd()
    splitter.output_dir = tmp
    splitter.log = agents.get_logger("splitter")
    audio = ROOT / "examples" / "guitar_solo.wav"
    with (
        patch.object(agents, "DEMUCS_API_AVAILABLE", False),
        patch.object(agents, "health"),
        patch.object(agents, "metrics"),
        patch.object(splitter, "_separate_with_subprocess", side_effect=OSError("demucs missing")),
    ):
        stems = splitter.separate_stems(str(audio))
    check(
        set(stems) == {"guitar", "bass"} and all(Path(p).exists() for p in stems.values()),
        "demucs unavailable -> raw-audio passthrough",
        str(list(stems)),
    )


# ---------------------------------------------------------------------------
# 4. Disk / filesystem
# ---------------------------------------------------------------------------


def simulate_unwritable_output() -> None:
    import agents

    try:
        with patch("agents.os.makedirs", side_effect=OSError("read-only filesystem")):
            agents.SplitterAgent(output_dir="/proc/definitely-not-writable")
    except OSError as e:
        check("read-only" in str(e), "unwritable output dir raises OSError (not silent)")
    else:
        check(False, "unwritable output dir raises OSError (not silent)", "no exception")


# ---------------------------------------------------------------------------
# 5. Malformed model output
# ---------------------------------------------------------------------------


def simulate_malformed_mt3_tokens(capsys) -> None:
    import agents

    ear = agents.EarAgent.__new__(agents.EarAgent)
    notes = ear._parse_mt3_tokens(
        ["NOTE_ON", "sixty", "TIME_SHIFT", "0.5", "NOTE_OFF", "60"], min_duration=0.05
    )
    out = capsys.readouterr().out
    check(notes == [], "malformed MT3 tokens yield no notes")
    check("mt3_token_skipped" in out, "malformed MT3 tokens are logged", out.strip()[:120])


def main() -> int:
    simulations = [
        simulate_corrupt_user_memory,
        simulate_invalid_thresholds,
        simulate_missing_env_file,
        simulate_bad_audio,
        simulate_oversize_and_format,
        simulate_no_backend_health,
        simulate_transcription_backend_down,
        simulate_demucs_offline,
        simulate_unwritable_output,
    ]
    for sim in simulations:
        try:
            sim()
        except Exception as e:
            check(False, sim.__name__, f"simulation raised {type(e).__name__}: {e}")

    # Needs capsys; run separately through pytest-compatible capture.
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            simulate_malformed_mt3_tokens(type("C", (), {"readouterr": lambda self: type("R", (), {"out": buf.getvalue()})()})())  # type: ignore[arg-type]
        except Exception as e:
            check(False, "simulate_malformed_mt3_tokens", str(e))

    failures = [r for r in RESULTS if not r[0]]
    for ok, name, detail in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail and not ok else ""))
    print(f"\n{len(RESULTS) - len(failures)}/{len(RESULTS)} failure-mode checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
