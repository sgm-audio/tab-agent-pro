import json
import sys

import pytest

from monitoring import PipelineLogger, default_metrics, health


def test_health_tracker_payload_shape() -> None:
    d = health.as_dict()
    assert "status" in d
    assert "ready" in d
    assert "uptime_s" in d
    assert "components" in d
    assert "requests" in d
    assert "last_error" in d
    assert "last_success" in d
    assert "python_version" in d
    assert "platform" in d


def test_health_status_is_derived_not_hardcoded() -> None:
    """
    Status must reflect backend reality.

    Regression: `health` was initialised to "up" at import and never updated,
    so a container with no working backend passed its healthcheck.
    """
    from monitoring import HealthTracker

    tracker = HealthTracker()
    tracker.set_status("up")
    # Nothing registered yet: not "broken", but definitely not ready.
    assert tracker.is_ready() is False

    tracker.set_component("basic_pitch", "unavailable")
    tracker.set_component("demucs", "unavailable")
    assert tracker.effective_status() == HealthTracker.STATUS_DOWN
    assert tracker.as_dict()["status"] == "down"
    assert tracker.as_dict()["ready"] is False

    # Only the hard-required backend is left: serving, but quality-degraded.
    # (Demucs has a passthrough fallback, so it must not gate readiness -
    # otherwise a working instance takes itself out of rotation.)
    tracker.set_component("basic_pitch", "available")
    assert tracker.effective_status() == HealthTracker.STATUS_DEGRADED
    assert tracker.is_ready() is True
    assert tracker.as_dict()["degraded_components"] == ["demucs"]

    # Both backends usable -> up and ready.
    tracker.set_component("demucs", "loaded (CLI)")
    assert tracker.effective_status() == HealthTracker.STATUS_UP
    assert tracker.is_ready() is True
    assert tracker.as_dict()["degraded_components"] == []


def test_health_global_exposes_backends_that_gate_readiness() -> None:
    """
    The process-wide tracker publishes the components that gate readiness.

    Values are mutated by other tests in the session (agents.py records them at
    import time, and several tests patch the availability flags), so this
    asserts the shape/type contract rather than the ambient values.
    """
    component_health = health.component_health()
    assert set(component_health) == {"basic_pitch"}
    assert all(isinstance(v, bool) for v in component_health.values())
    assert isinstance(health.is_ready(), bool)


def test_default_metrics() -> None:
    assert hasattr(default_metrics, "track_stage")
    assert callable(default_metrics.track_stage)


def test_pipeline_logger() -> None:
    log = PipelineLogger("test_module")
    # Capture stdout
    from io import StringIO

    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        log.info("test_event", key="value")
        log.warning("warn_event")
        log.error("error_event", exc=ValueError("bad"))
    finally:
        sys.stdout = old_stdout

    lines = buf.getvalue().strip().splitlines()
    assert len(lines) == 3

    for line in lines:
        record = json.loads(line)
        assert "ts" in record
        assert "module" in record
        assert "event" in record

    assert json.loads(lines[0])["level"] == "INFO"
    assert json.loads(lines[0])["event"] == "test_event"
    assert json.loads(lines[1])["level"] == "WARN"
    assert json.loads(lines[2])["level"] == "ERROR"
    assert json.loads(lines[2])["error_type"] == "ValueError"


def test_create_ui() -> None:
    pytest.importorskip("gradio")
    import app

    demo = app.create_ui()
    import gradio as gr

    assert isinstance(demo, gr.Blocks)
