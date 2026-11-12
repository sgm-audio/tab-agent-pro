"""Coverage tests for app.py."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


def _import_app():
    pytest.importorskip("gradio")
    import app

    return app


def test_create_app_health() -> None:
    app = _import_app()
    from httpx import ASGITransport, AsyncClient

    fastapi_app = app.create_app()
    transport = ASGITransport(app=fastapi_app)

    async def _test():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/health")

    resp = asyncio.run(_test())
    assert resp.status_code == 200
    data = resp.json()
    # /health is liveness: always 200, and the status must be *derived* from
    # component health (never hard-coded to "up").
    assert data["status"] in {"up", "degraded", "down", "starting"}
    assert isinstance(data["ready"], bool)
    assert "uptime_s" in data
    assert "components" in data
    assert "requests" in data
    assert "python_version" in data
    assert "platform" in data


def test_ready_endpoint_reports_backend_availability() -> None:
    """/ready must be 200 only when a usable transcription backend exists."""
    app = _import_app()
    from httpx import ASGITransport, AsyncClient

    from monitoring import health

    fastapi_app = app.create_app()
    transport = ASGITransport(app=fastapi_app)

    async def _probe():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/ready"), health.is_ready()

    resp, ready = asyncio.run(_probe())
    assert resp.status_code == (200 if ready else 503)
    body = resp.json()
    assert body["ready"] is ready
    assert set(body["components"]) == {"basic_pitch"}


def test_create_app_metrics() -> None:
    app = _import_app()
    from httpx import ASGITransport, AsyncClient

    fastapi_app = app.create_app()
    transport = ASGITransport(app=fastapi_app)

    async def _test():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/health/metrics")

    resp = asyncio.run(_test())
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)


def test_create_ui_returns_blocks() -> None:
    app = _import_app()
    import gradio as gr

    demo = app.create_ui()
    assert isinstance(demo, gr.Blocks)


def test_process_audio_impl_none_audio() -> None:
    app = _import_app()

    msg, zip_path, _tab = app._process_audio_impl(None, "Guitar", True, True, True, None)
    assert "upload" in msg.lower()
    assert zip_path is None


def test_process_audio_non_gpu_wrapper() -> None:
    app = _import_app()

    with patch("app._process_audio_impl", return_value=("OK", "/tmp/z.zip", None)) as mock_impl:
        msg, _zp, _tab = app.process_audio("f.wav", "Guitar", True, True, True, MagicMock())
        assert msg == "OK"
        mock_impl.assert_called_once()


def test_process_audio_impl_error() -> None:
    app = _import_app()

    mock_progress = MagicMock()
    with (
        patch("app.process_suno_audio", side_effect=ValueError("boom")),
        patch("app._validate_audio", return_value=None),
    ):
        msg, zip_path, _tab = app._process_audio_impl(
            "dummy.wav", "Guitar", True, True, True, mock_progress
        )
        assert "failed" in msg
        assert zip_path is None


def _mock_notes():
    """Two note objects shaped like note_seq notes (app only reads pitch/times)."""
    import note_seq

    return [
        note_seq.NoteSequence.Note(pitch=64, start_time=0.0, end_time=0.4, velocity=80),
        note_seq.NoteSequence.Note(pitch=67, start_time=0.5, end_time=0.9, velocity=80),
    ]


def _make_mock_stems():
    return {"lead": "/tmp/lead.wav", "left": "/tmp/left.wav", "right": "/tmp/right.wav"}


def _make_mock_bass_stems():
    return {"bass": "/tmp/bass.wav"}


def _run_mocked_pipeline(app, instrument, stems_fn, is_suno=False):
    mock_progress = MagicMock()
    mock_suno = patch(
        "app.process_suno_audio",
        return_value=("/tmp/test.wav", is_suno, {"sample_rate": 44100}),
    )
    mock_export_txt = patch("app.export_tab_to_txt")
    mock_export_json = patch("app.export_tab_to_json")
    mock_validate = patch("app._validate_audio", return_value=None)

    with (
        mock_suno,
        mock_export_txt,
        mock_export_json,
        mock_validate,
        patch("app.SplitterAgent") as splitter_cls,
        patch("app.EarAgent") as ear_cls,
        patch("app.SunoNotePostprocessor") as sgp_cls,
        patch("app.TabAgent") as tab_cls,
    ):
        splitter_inst = splitter_cls.return_value
        splitter_inst.separate_stems.return_value = {
            "guitar": "/tmp/g.wav",
            "bass": "/tmp/b.wav",
        }
        splitter_inst.process_guitars.return_value = stems_fn()
        splitter_inst.process_bass.return_value = "/tmp/bass_clean.wav"

        ear_inst = ear_cls.return_value
        ear_inst.transcribe_stem.return_value = _mock_notes()
        ear_inst.humanize_and_clean.return_value = _mock_notes()

        sgp_inst = sgp_cls.return_value
        sgp_inst.process.return_value = _mock_notes()

        tab_inst = tab_cls.return_value
        tab_inst.generate_tab.return_value = [
            {"string": 0, "fret": 3, "technique": "pick", "start_time": 0.0}
        ]

        msg, zip_path, _tab = app._process_audio_impl(
            "dummy.wav", instrument, True, True, True, mock_progress
        )
    return msg, zip_path, _tab


def test_process_audio_impl_guitar_full_pipeline() -> None:
    app = _import_app()

    msg, zip_path, _tab = _run_mocked_pipeline(app, "Guitar", _make_mock_stems)
    assert "Complete!" in msg
    assert zip_path is not None
    assert zip_path.endswith(".zip")


def test_process_audio_impl_bass_full_pipeline() -> None:
    app = _import_app()

    msg, zip_path, _tab = _run_mocked_pipeline(app, "Bass", _make_mock_bass_stems)
    assert "Complete!" in msg
    assert zip_path is not None
    assert zip_path.endswith(".zip")


def test_process_audio_impl_suno_true() -> None:
    app = _import_app()

    msg, zip_path, _tab = _run_mocked_pipeline(app, "Guitar", _make_mock_stems, is_suno=True)
    assert "Complete!" in msg
    assert zip_path is not None


def test_process_audio_impl_zip_with_files() -> None:
    """
    A real run writes its artifacts into a fresh session directory.

    The directory name used to be `session_<second-resolution timestamp>`, so
    two requests in the same second shared it; it is now a unique tempdir.
    """
    app = _import_app()

    before = {p.name for p in app.OUTPUT_DIR.iterdir() if p.is_dir()}
    mock_progress = MagicMock()
    with (
        patch("app.process_suno_audio", return_value=("/tmp/test.wav", False, {})),
        patch("app._validate_audio", return_value=None),
        patch("app.SplitterAgent") as sc,
        patch("app.EarAgent") as ec,
        patch("app.SunoNotePostprocessor") as sgc,
        patch("app.TabAgent") as tc,
        patch("app.export_tab_to_txt"),
        patch("app.export_tab_to_json"),
    ):
        sc.return_value.separate_stems.return_value = {"guitar": "/tmp/g.wav"}
        sc.return_value.process_guitars.return_value = _make_mock_stems()
        ec.return_value.transcribe_stem.return_value = _mock_notes()
        ec.return_value.humanize_and_clean.return_value = _mock_notes()
        sgc.return_value.process.return_value = _mock_notes()
        tc.return_value.generate_tab.return_value = [
            {"string": 0, "fret": 3, "technique": "pick", "start_time": 0.0}
        ]

        msg, _zp, _tab = app._process_audio_impl("d.wav", "Guitar", True, True, True, mock_progress)

    assert "Complete!" in msg
    after = {p.name for p in app.OUTPUT_DIR.iterdir() if p.is_dir()}
    created = after - before
    assert created, "a new session directory must be created for each request"
    assert len(created) == 1
    # Unique, collision-resistant names (mkdtemp), not a shared timestamp.
    assert created.pop().startswith("session_")
