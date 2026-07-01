"""Coverage tests for app.py."""

import asyncio
from datetime import datetime as _real_dt
from unittest.mock import MagicMock, patch

import pytest


def _import_app():
    pytest.importorskip("gradio")
    import app

    return app


def test_create_app_health():
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
    assert data["status"] == "up"
    assert "uptime_s" in data
    assert "components" in data
    assert "requests" in data
    assert "python_version" in data
    assert "platform" in data


def test_create_app_metrics():
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


def test_create_ui_returns_blocks():
    app = _import_app()
    import gradio as gr

    demo = app.create_ui()
    assert isinstance(demo, gr.Blocks)


def test_process_audio_impl_none_audio():
    app = _import_app()

    msg, zip_path = app._process_audio_impl(None, "Guitar", True, True, True, None)
    assert "upload" in msg.lower()
    assert zip_path is None


def test_process_audio_non_gpu_wrapper():
    app = _import_app()

    with patch("app._process_audio_impl", return_value=("OK", "/tmp/z.zip")) as mock_impl:
        msg, zp = app.process_audio("f.wav", "Guitar", True, True, True, MagicMock())
        assert msg == "OK"
        mock_impl.assert_called_once()


def test_process_audio_impl_error():
    app = _import_app()

    mock_progress = MagicMock()
    with patch("app.process_suno_audio", side_effect=ValueError("boom")):
        msg, zip_path = app._process_audio_impl(
            "dummy.wav", "Guitar", True, True, True, mock_progress
        )
        assert "Error" in msg
        assert zip_path is None


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

    with mock_suno, mock_export_txt, mock_export_json:
        with patch("app.SplitterAgent") as splitter_cls:
            splitter_inst = splitter_cls.return_value
            splitter_inst.separate_stems.return_value = {
                "guitar": "/tmp/g.wav",
                "bass": "/tmp/b.wav",
            }
            splitter_inst.process_guitars.return_value = stems_fn()
            splitter_inst.process_bass.return_value = "/tmp/bass_clean.wav"

            with patch("app.EarAgent") as ear_cls:
                ear_inst = ear_cls.return_value
                ear_inst.transcribe_stem.return_value = []
                ear_inst.humanize_and_clean.return_value = []

                with patch("app.SunoNotePostprocessor") as sgp_cls:
                    sgp_inst = sgp_cls.return_value
                    sgp_inst.process.return_value = []

                    with patch("app.TabAgent") as tab_cls:
                        tab_inst = tab_cls.return_value
                        tab_inst.generate_tab.return_value = []

                        msg, zip_path = app._process_audio_impl(
                            "dummy.wav", instrument, True, True, True, mock_progress
                        )
    return msg, zip_path


def test_process_audio_impl_guitar_full_pipeline():
    app = _import_app()

    msg, zip_path = _run_mocked_pipeline(app, "Guitar", _make_mock_stems)
    assert "Complete!" in msg
    assert zip_path is not None
    assert zip_path.endswith(".zip")


def test_process_audio_impl_bass_full_pipeline():
    app = _import_app()

    msg, zip_path = _run_mocked_pipeline(app, "Bass", _make_mock_bass_stems)
    assert "Complete!" in msg
    assert zip_path is not None
    assert zip_path.endswith(".zip")


def test_process_audio_impl_suno_true():
    app = _import_app()

    msg, zip_path = _run_mocked_pipeline(app, "Guitar", _make_mock_stems, is_suno=True)
    assert "Complete!" in msg
    assert zip_path is not None


def test_process_audio_impl_zip_with_files():
    app = _import_app()

    fixed = _real_dt(2024, 6, 15, 10, 30, 0)
    with patch("app.datetime") as mock_dt:
        mock_dt.now.return_value = fixed

        ts = "20240615_103000"
        session_dir = app.OUTPUT_DIR / f"session_{ts}"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "test_output.mid").write_text("fake midi")

        mock_progress = MagicMock()
        with patch("app.process_suno_audio", return_value=("/tmp/test.wav", False, {})):
            with patch("app.SplitterAgent") as sc:
                sc.return_value.separate_stems.return_value = {"guitar": "/tmp/g.wav"}
                sc.return_value.process_guitars.return_value = _make_mock_stems()
                with patch("app.EarAgent") as ec:
                    ec.return_value.transcribe_stem.return_value = []
                    ec.return_value.humanize_and_clean.return_value = []
                    with patch("app.SunoNotePostprocessor") as sgc:
                        sgc.return_value.process.return_value = []
                        with patch("app.TabAgent") as tc:
                            tc.return_value.generate_tab.return_value = []
                            with patch("app.export_tab_to_txt"), patch("app.export_tab_to_json"):
                                msg, zp = app._process_audio_impl(
                                    "d.wav", "Guitar", True, True, True, mock_progress
                                )
        assert "Complete!" in msg
        assert session_dir.exists()
