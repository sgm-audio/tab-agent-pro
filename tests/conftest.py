"""Keep app startup/shutdown in tests away from real uploads and other runs."""

import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_app_tempdir(tmp_path_factory):
    root = tmp_path_factory.mktemp("tab-agent-runtime")
    with pytest.MonkeyPatch.context() as patch:
        # app caches its output root on import; isolate test sessions and uploads.
        patch.setattr(tempfile, "tempdir", str(root))
        patch.setenv("GRADIO_TEMP_DIR", str(root / "gradio"))
        yield
