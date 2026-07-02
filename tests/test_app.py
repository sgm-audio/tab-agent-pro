import json
import sys

import pytest

from monitoring import PipelineLogger, default_metrics, health


def test_health_tracker() -> None:
    d = health.as_dict()
    assert "status" in d
    assert "uptime_s" in d
    assert "components" in d
    assert "requests" in d
    assert "last_error" in d
    assert "last_success" in d
    assert "python_version" in d
    assert "platform" in d
    assert d["status"] == "up"


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
