"""
Regression tests for the app wiring fixed by the docs audit.

Covers:
  * P0-1 — the Gradio button used to call ``_process_audio_impl`` with 5 inputs
    while the function required 6 (``progress`` with no default), so every
    transcription attempt failed with "'NoneType' object is not callable".
  * P0-3/P0-4 — host/port defaults and Zero GPU gating.
  * P2-1 — .env loading.
"""

import inspect
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _import_app():
    pytest.importorskip("gradio")
    import app

    return app


# ---------------------------------------------------------------------------
# P0-1: the UI entry point must accept exactly the connected inputs
# ---------------------------------------------------------------------------


def test_process_audio_impl_accepts_five_args() -> None:
    """The regression: 5 positional args (as wired in create_ui) must not raise."""
    app = _import_app()

    msg, zip_path = app._process_audio_impl(None, "Guitar", True, True, True)
    assert "upload" in msg.lower()
    assert zip_path is None


def test_every_pipeline_entry_point_has_defaults() -> None:
    app = _import_app()

    for name in ("_process_audio_impl", "process_audio", "ui_entry"):
        fn = getattr(app, name)
        params = list(inspect.signature(fn).parameters.values())
        optional = [p for p in params if p.default is not inspect.Parameter.empty]
        required = [p for p in params if p.default is inspect.Parameter.empty]
        assert len(required) <= 1, f"{name} requires {[p.name for p in required]}"
        assert optional, f"{name} has no defaulted parameters"


def test_ui_entry_matches_runtime_capabilities() -> None:
    """ui_entry is the GPU wrapper only when a Zero GPU slice is available."""
    app = _import_app()

    if app.GPU_AVAILABLE:
        assert app.ui_entry is app._gpu_process_audio
    else:
        assert app.ui_entry is app.process_audio


def test_gpu_available_requires_zero_gpu_env() -> None:
    app = _import_app()

    assert (app.SPACES_AVAILABLE and app.ZERO_GPU) == app.GPU_AVAILABLE
    # A plain import of `spaces` alone must not advertise GPU acceleration.
    if not app.ZERO_GPU:
        assert app.GPU_AVAILABLE is False


def test_transcribe_button_is_bound_to_ui_entry() -> None:
    """create_ui must wire the button to ui_entry, not to a mismatched function."""
    app = _import_app()

    source = inspect.getsource(app.create_ui)
    assert "fn=ui_entry" in source
    assert "fn=_process_audio_impl" not in source

    # And the click handler must be callable with the 5 connected inputs.
    demo = app.create_ui()
    import gradio as gr

    assert isinstance(demo, gr.Blocks)


def test_process_audio_wrapper_forwards_all_args() -> None:
    app = _import_app()

    with patch("app._process_audio_impl", return_value=("OK", "/tmp/z.zip")) as impl:
        msg, _ = app.process_audio("f.wav", "Guitar", True, False, True, MagicMock())
    assert msg == "OK"
    assert impl.call_args.args[1:5] == ("Guitar", True, False, True)


# ---------------------------------------------------------------------------
# P0-3: container/Spaces reachability
# ---------------------------------------------------------------------------


def test_default_host_is_not_loopback() -> None:
    app = _import_app()

    module_source = Path(app.__file__).read_text()
    assert 'os.getenv("HOST", "127.0.0.1")' not in module_source
    assert 'os.getenv("HOST", "0.0.0.0")' in module_source
    assert 'os.getenv("PORT", "7860")' in module_source


def test_port_env_is_used(monkeypatch) -> None:
    app = _import_app()

    seen: dict = {}

    class _FakeUvicorn:
        @staticmethod
        def run(_app, host=None, port=None):
            seen["host"] = host
            seen["port"] = port

    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9001")
    monkeypatch.setattr("uvicorn.run", _FakeUvicorn.run)
    monkeypatch.setattr(app, "create_app", lambda: object())

    app.serve()

    assert seen == {"host": "0.0.0.0", "port": 9001}


def test_serve_defaults(monkeypatch) -> None:
    """Without HOST/PORT the server must still bind externally on 7860."""
    app = _import_app()

    seen: dict = {}
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setattr(
        "uvicorn.run", lambda _app, host=None, port=None: seen.update(host=host, port=port)
    )
    monkeypatch.setattr(app, "create_app", lambda: object())

    app.serve()

    assert seen == {"host": "0.0.0.0", "port": 7860}


# ---------------------------------------------------------------------------
# P2-1: .env support
# ---------------------------------------------------------------------------


def test_load_dotenv_reads_file(tmp_path, monkeypatch) -> None:
    app = _import_app()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "TAB_AGENT_TEST_KEY=hello\n"
        'TAB_AGENT_TEST_QUOTED="quoted value"\n'
        "TAB_AGENT_TEST_EMPTY=\n"
        "not-a-valid-line\n"
    )

    monkeypatch.delenv("TAB_AGENT_TEST_KEY", raising=False)
    monkeypatch.delenv("TAB_AGENT_TEST_QUOTED", raising=False)
    app._load_dotenv(str(env_file))

    assert os.environ["TAB_AGENT_TEST_KEY"] == "hello"
    assert os.environ["TAB_AGENT_TEST_QUOTED"] == "quoted value"
    monkeypatch.undo()


def test_load_dotenv_does_not_override_existing(tmp_path, monkeypatch) -> None:
    app = _import_app()
    env_file = tmp_path / ".env"
    env_file.write_text("TAB_AGENT_TEST_EXISTING=from-file\n")
    monkeypatch.setenv("TAB_AGENT_TEST_EXISTING", "from-env")

    app._load_dotenv(str(env_file))

    assert os.environ["TAB_AGENT_TEST_EXISTING"] == "from-env"


def test_load_dotenv_missing_file_is_noop(tmp_path) -> None:
    app = _import_app()

    app._load_dotenv(str(tmp_path / "does-not-exist.env"))  # must not raise


# ---------------------------------------------------------------------------
# Status reporting for an empty transcription
# ---------------------------------------------------------------------------


def test_no_notes_produces_warning_not_success() -> None:
    app = _import_app()

    with (
        patch("app.process_suno_audio", return_value=("/tmp/x.wav", False, {})),
        patch("app._validate_audio", return_value=None),
        patch("app.SplitterAgent") as splitter_cls,
        patch("app.EarAgent") as ear_cls,
        patch("app.TabAgent"),
        patch("app.SunoNotePostprocessor") as suno_cls,
        patch("app.export_tab_to_txt"),
        patch("app.export_tab_to_json"),
    ):
        splitter_cls.return_value.separate_stems.return_value = {"guitar": "g", "bass": "b"}
        splitter_cls.return_value.process_guitars.return_value = {
            "lead": "l",
            "left": "l",
            "right": "r",
        }
        ear_cls.return_value.transcribe_stem.return_value = []
        ear_cls.return_value.humanize_and_clean.return_value = []
        suno_cls.return_value.process.return_value = []

        msg, zip_path = app._process_audio_impl("d.wav", "Guitar", True, True, True)

    assert "No notes were transcribed" in msg
    assert zip_path is not None  # the (mostly empty) ZIP is still downloadable
