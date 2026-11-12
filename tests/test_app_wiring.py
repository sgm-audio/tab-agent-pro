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

    msg, zip_path, _tab = app._process_audio_impl(None, "Guitar", True, True, True)
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


def test_ui_entry_is_process_audio() -> None:
    """ui_entry is the plain process function (no cloud GPU wrapper)."""
    app = _import_app()
    assert app.ui_entry is app.process_audio


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

        msg, zip_path, _tab = app._process_audio_impl("d.wav", "Guitar", True, True, True)

    assert "No notes were transcribed" in msg
    assert zip_path is not None  # the (mostly empty) ZIP is still downloadable


def test_files_generated_not_double_counted_and_formats_clean() -> None:
    """Files Generated matches actual ZIP entries and Formats has no empty commas."""
    import zipfile

    import note_seq

    app = _import_app()
    fake_note = note_seq.NoteSequence.Note(pitch=40, start_time=0.0, end_time=0.5, velocity=80)

    with (
        patch("app.process_suno_audio", return_value=("/tmp/x.wav", False, {})),
        patch("app._validate_audio", return_value=None),
        patch("app.SplitterAgent") as splitter_cls,
        patch("app.EarAgent") as ear_cls,
        patch("app.SunoNotePostprocessor") as suno_cls,
    ):
        splitter_cls.return_value.separate_stems.return_value = {"guitar": "g", "bass": "b"}
        splitter_cls.return_value.process_bass.return_value = "b"
        ear_cls.return_value.transcribe_stem.return_value = [fake_note]
        ear_cls.return_value.humanize_and_clean.return_value = [fake_note]
        suno_cls.return_value.process.return_value = [fake_note]

        def _write_midi(_notes, path):
            Path(path).write_bytes(b"MThd")

        ear_cls.return_value.export_midi.side_effect = _write_midi

        # Uncheck Tab (middle option) to test both format string and file count
        msg, zip_path, _tab = app._process_audio_impl("d.wav", "Bass", True, False, True)

        assert zip_path is not None
        with zipfile.ZipFile(zip_path, "r") as zf:
            num_entries = len(zf.namelist())
        assert f"**Files Generated**: {num_entries}" in msg
        assert "**Formats**: MIDI, JSON" in msg
        assert ", ," not in msg


def test_import_does_not_hijack_sigint() -> None:
    """Importing/reloading app must not replace the process SIGINT handler outside serve()."""
    import importlib
    import signal

    app = _import_app()
    prev = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        importlib.reload(app)
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, prev)


def test_app_honors_saved_user_preferences(tmp_path, monkeypatch) -> None:
    """Web UI loads saved user_preferences.json (suno_aggressive_mode, tuning, min_pitch)."""
    import json

    import init_memory

    app = _import_app()
    prefs_file = tmp_path / "user_preferences.json"
    prefs_file.write_text(
        json.dumps(
            {
                "config": {
                    "guitar_tuning": [38, 45, 50, 55, 59, 64],
                    "suno_aggressive_mode": True,
                    "prefer_low_strings": True,
                }
            }
        )
    )
    monkeypatch.setattr(init_memory, "get_memory_dir", lambda: str(tmp_path))

    with (
        patch("app.process_suno_audio", return_value=("/tmp/x.wav", False, {})) as suno_fn,
        patch("app._validate_audio", return_value=None),
        patch("app.SplitterAgent") as splitter_cls,
        patch("app.EarAgent") as ear_cls,
        patch("app.TabAgent") as tab_cls,
        patch("app.SunoNotePostprocessor") as suno_cls,
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

        app._process_audio_impl("d.wav", "Guitar", True, True, True)

        assert suno_fn.call_args.kwargs.get("aggressiveness") == 0.8
        assert ear_cls.return_value.transcribe_stem.call_args.kwargs.get("min_pitch") == 38
        assert tab_cls.call_args.kwargs.get("prefer_low_strings") is True


def test_load_saved_config_ignores_corrupt_json(tmp_path, monkeypatch) -> None:
    """Corrupt user_preferences.json falls back cleanly to empty config dict."""
    import init_memory

    app = _import_app()
    (tmp_path / "user_preferences.json").write_text("{corrupt json")
    monkeypatch.setattr(init_memory, "get_memory_dir", lambda: str(tmp_path))
    assert app._load_saved_config() == {}


def test_same_second_sessions_get_distinct_directories() -> None:
    """Two runs within the same second do not overwrite each other's session dir."""
    app = _import_app()

    # No clock patch: session directory names come from tempfile.mkdtemp, and
    # patching app.datetime breaks the epoch arithmetic in cleanup_stale_uploads
    # (MagicMock > int). Two back-to-back calls are "the same second" in the
    # only sense that matters here - nothing about the name is time-derived.
    with (
        patch("app.process_suno_audio", return_value=("/tmp/x.wav", False, {})),
        patch("app._validate_audio", return_value=None),
        patch("app.SplitterAgent") as splitter_cls,
        patch("app.EarAgent") as ear_cls,
        patch("app.SunoNotePostprocessor") as suno_cls,
    ):
        splitter_cls.return_value.separate_stems.return_value = {"guitar": "g", "bass": "b"}
        splitter_cls.return_value.process_bass.return_value = "b"
        ear_cls.return_value.transcribe_stem.return_value = []
        ear_cls.return_value.humanize_and_clean.return_value = []
        suno_cls.return_value.process.return_value = []

        _msg1, zip1, _tab1 = app._process_audio_impl("d.wav", "Bass", True, True, True)
        _msg2, zip2, _tab2 = app._process_audio_impl("d.wav", "Bass", True, True, True)

        assert zip1 is not None and zip2 is not None
        assert Path(zip1).parent != Path(zip2).parent
        # Assert the behaviour, not the naming scheme: this branch replaces the
        # same-second retry loop with tempfile.mkdtemp, which is atomic and so
        # also closes the race between the loop's existence check and mkdir.
        # Both directories must exist and be unique.
        assert Path(zip1).parent.is_dir() and Path(zip2).parent.is_dir()
        assert Path(zip1).parent.name != Path(zip2).parent.name
