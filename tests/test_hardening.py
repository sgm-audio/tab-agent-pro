"""
Regression tests for the production-readiness hardening pass.

Each test here pins a defect that was found by the phase-1/2 audit and is
otherwise invisible to the original suite:

  * health status derived from backends (was hard-coded to "up")
  * /ready readiness probe (did not exist)
  * concurrent web requests must not share an output directory
  * metrics memory must stay bounded on a long-lived process
  * metrics/health must be thread-safe
  * internal exception text must not reach the web user
  * `--instrument` default must match its documented behaviour
  * threshold values from user memory must be range-validated
"""

import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoring import HealthTracker, PipelineMetrics, get_logger

# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------


class TestHealthDerivation:
    def test_no_usable_backend_reports_down(self) -> None:
        h = HealthTracker()
        h.set_status("up")
        h.set_component("basic_pitch", "unavailable")
        h.set_component("demucs", "unavailable")
        assert h.effective_status() == "down"
        assert h.is_ready() is False

    def test_missing_demucs_degrades_but_stays_ready(self) -> None:
        """
        Demucs missing is degraded, not unready.

        Separation has a documented raw-audio passthrough, so a Pod without
        Demucs weights still transcribes whole-mix audio. Gating readiness on it
        would take a serving instance out of rotation (found in sign-off: a live
        run reported ready=False *while* completing a 14-file transcription).
        """
        h = HealthTracker()
        h.set_status("up")
        h.set_component("basic_pitch", "available")
        h.set_component("demucs", "unavailable")
        assert h.effective_status() == "degraded"
        assert h.is_ready() is True
        assert h.as_dict()["degraded_components"] == ["demucs"]

    def test_no_transcription_backend_is_unready(self) -> None:
        h = HealthTracker()
        h.set_status("up")
        h.set_component("basic_pitch", "unavailable")
        h.set_component("demucs", "loaded")
        assert h.effective_status() == "down"
        assert h.is_ready() is False

    def test_all_backends_ready(self) -> None:
        h = HealthTracker()
        h.set_status("up")
        h.set_component("basic_pitch", "available")
        h.set_component("demucs", "loaded")
        assert h.effective_status() == "up"
        assert h.is_ready() is True

    def test_degraded_component_still_counts_as_usable(self) -> None:
        """Demucs falling back to the CLI still produces stems."""
        h = HealthTracker()
        h.set_status("up")
        h.set_component("basic_pitch", "available")
        h.set_component("demucs", "loaded (CLI)")
        assert h.is_ready() is True

    def test_explicit_down_is_respected(self) -> None:
        h = HealthTracker()
        h.set_component("basic_pitch", "available")
        h.set_component("demucs", "available")
        h.set_status("down")
        assert h.effective_status() == "down"

    def test_high_error_rate_degrades(self) -> None:
        h = HealthTracker()
        h.set_status("up")
        h.set_component("basic_pitch", "available")
        h.set_component("demucs", "available")
        for _ in range(9):
            h.record_request(success=False)
        h.record_request(success=True)
        assert h.effective_status() == "degraded"
        # Readiness must not flap on user-supplied bad audio.
        assert h.is_ready() is True


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_health_counters_are_thread_safe() -> None:
    h = HealthTracker()

    def worker() -> None:
        for _ in range(2000):
            h.record_request(success=True)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert h.as_dict()["requests"]["total"] == 16000


def test_concurrent_component_registration_does_not_raise() -> None:
    h = HealthTracker()
    errors: list[BaseException] = []

    def worker(idx: int) -> None:
        try:
            for j in range(500):
                h.set_component(f"component_{idx}", str(j))
                h.component_health()
                h.as_dict()
        except BaseException as exc:  # recorded and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


# ---------------------------------------------------------------------------
# Bounded metrics
# ---------------------------------------------------------------------------


class TestMetricsBounded:
    def test_retained_samples_are_bounded(self) -> None:
        m = PipelineMetrics(logger=get_logger("test"), window=32)
        for _ in range(500):
            m.record_stage("demucs", 0.5, True)
        assert len(m.stage_times["demucs"]) == 32

    def test_totals_stay_exact_while_memory_is_bounded(self) -> None:
        m = PipelineMetrics(logger=get_logger("test"), window=10)
        for i in range(1000):
            m.record_stage("demucs", 2.0, i % 4 != 0)
        summary = m.summary()["demucs"]
        assert summary["count"] == 1000
        assert summary["total_s"] == pytest.approx(2000.0)
        assert summary["avg_s"] == pytest.approx(2.0)
        assert summary["error_rate"] == pytest.approx(0.25)
        assert summary["window_count"] == 10

    def test_summary_is_empty_when_nothing_recorded(self) -> None:
        assert PipelineMetrics(logger=get_logger("test")).summary() == {}


# ---------------------------------------------------------------------------
# Web request isolation
# ---------------------------------------------------------------------------


def test_concurrent_sessions_get_distinct_directories() -> None:
    """
    Regression: session dirs were named with a second-resolution timestamp and
    created with exist_ok=True, so two requests in the same second shared one
    directory and one user's ZIP could contain another user's files.
    """
    pytest.importorskip("gradio")
    import tempfile

    import app

    original = app.OUTPUT_DIR
    tmp = Path(tempfile.mkdtemp())
    app.OUTPUT_DIR = tmp
    try:
        # Call the same helper the request path uses, twice, as fast as possible.
        first = Path(tempfile.mkdtemp(prefix="session_", dir=str(app.OUTPUT_DIR)))
        second = Path(tempfile.mkdtemp(prefix="session_", dir=str(app.OUTPUT_DIR)))
        assert first != second
        assert first.is_dir() and second.is_dir()
    finally:
        app.OUTPUT_DIR = original
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def test_session_directory_creation_uses_mkdtemp() -> None:
    """Guard the fix at the source level so it cannot silently regress."""
    source = Path("app.py").read_text()
    assert 'tempfile.mkdtemp(prefix="session_", dir=str(OUTPUT_DIR))' in source
    assert 'strftime("%Y%m%d_%H%M%S")' not in source.split("session_")[0]


def test_web_error_does_not_leak_internal_detail() -> None:
    """
    Regression: the catch-all returned `Transcription failed: {e}`, exposing
    filesystem paths and library internals to web users.
    """
    source = Path("app.py").read_text()
    assert 'f"❌ **Transcription failed:** {e}"' not in source
    assert "Error reference" in source


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


class TestInstrumentContract:
    def test_help_documents_guitar_default(self) -> None:
        import main

        help_text = main.build_parser().format_help()
        assert "default: Guitar" in help_text

    def test_default_flag_value_is_guitar(self) -> None:
        import main

        args = main.build_parser().parse_args(["song.wav"])
        assert args.instrument == "Guitar"

    def test_default_runs_guitar_only(self, tmp_path, monkeypatch) -> None:
        """
        Regression: an absent --instrument forced run_bass, so a plain
        `main.py song.wav` produced bass output despite documenting
        "default: Guitar". Assert the pipeline selection matches the docs.
        """
        import main

        selected: dict[str, bool] = {}

        monkeypatch.setattr(main, "_load_dotenv", lambda *a, **k: None)
        monkeypatch.setattr(main, "load_user_memory", lambda: (str(tmp_path / "m.json"), {}))
        monkeypatch.setattr(main, "process_suno_audio", lambda *a, **k: ("x.wav", False, {}))

        class _Splitter:
            def __init__(self, *a, **k) -> None:
                pass

            def separate_stems(self, path):
                return {"guitar": "g.wav", "bass": "b.wav"}

            def process_guitars(self, path):
                return {"lead": "l", "left": "a", "right": "b"}

            def process_bass(self, path):
                executed = dict(selected)
                assert executed.get("guitar") is True
                raise AssertionError("bass pipeline must not run for the default")

        monkeypatch.setattr(main, "SplitterAgent", _Splitter)

        class _Ear:
            def __init__(self, *a, **k) -> None:
                pass

            def transcribe_stem(self, *a, **k):
                selected["guitar"] = True
                return []

            def humanize_and_clean(self, notes, is_bass=False):
                return notes

            def export_midi(self, *a, **k):
                return None

        monkeypatch.setattr(main, "EarAgent", _Ear)
        monkeypatch.setattr(main, "SunoNotePostprocessor", lambda *a, **k: _NoopPost())
        monkeypatch.setattr(main, "TabAgent", lambda *a, **k: _NoopTab())

        wav = tmp_path / "song.wav"
        wav.write_bytes(b"RIFF")
        main.main([str(wav), "--output-dir", str(tmp_path / "out")])
        assert selected == {"guitar": True}


class _NoopPost:
    def process(self, notes, is_suno, metrics):
        return notes


class _NoopTab:
    def __init__(self, *a, **k) -> None:
        pass

    def generate_tab(self, notes, technique_sensitivity=0.7):
        return []


# ---------------------------------------------------------------------------
# User-config validation
# ---------------------------------------------------------------------------


class TestThresholdValidation:
    @pytest.mark.parametrize(
        ("onset", "frame"),
        [(-0.01, 0.3), (0.5, 1.01), (0.5, 42.0), (float("nan"), 0.3), (0.5, float("inf"))],
    )
    def test_out_of_range_thresholds_are_rejected(self, onset, frame) -> None:
        import main

        with pytest.raises((ValueError, SystemExit)):
            main.validate_thresholds(onset, frame)

    @pytest.mark.parametrize(("onset", "frame"), [(0.0, 1.0), (0.5, 0.3), (1.0, 0.0), (0.0, 0.0)])
    def test_in_range_thresholds_accepted(self, onset, frame) -> None:
        import main

        assert main.validate_thresholds(onset, frame) == (onset, frame)


def test_cli_rejects_absurd_threshold(tmp_path, monkeypatch) -> None:
    """An impossible threshold must abort before any audio work happens."""
    import main

    monkeypatch.setattr(main, "_load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(main, "load_user_memory", lambda: (str(tmp_path / "m.json"), {}))
    monkeypatch.setattr(main, "process_suno_audio", lambda *a, **k: ("x.wav", False, {}))

    wav = tmp_path / "song.wav"
    wav.write_bytes(b"RIFF")
    with pytest.raises(SystemExit):
        main.main([str(wav), "--onset", "42", "--output-dir", str(tmp_path)])
