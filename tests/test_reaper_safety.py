"""Execute both Lua scripts against a REAPER shim, not just string searches."""

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

import main
from tools import check_reaper_settings as guard

ROOT = Path(__file__).resolve().parent.parent


def test_settings_gui_to_lua_command_to_cli_parser(tmp_path, monkeypatch):
    lupa = pytest.importorskip("lupa")
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    lua.execute("EXTSTATE_INIT = {}")
    lua.globals().INSTALL_PATH = str(tmp_path / "install path")
    lua.globals().AUDIO_PATH = str(tmp_path / "song name.wav")
    lua.execute(guard.STUB)
    lua.globals().GUI_INPUT = (
        f"2,{tmp_path / 'install path'},Bass,0.62,0.31,{tmp_path / 'out dir'},0,1,0"
    )
    lua.execute("reaper.GetUserInputs = function(...) return true, GUI_INPUT end")
    lua.execute((ROOT / "reaper/Settings.lua").read_text())
    lua.execute((ROOT / "reaper/TabAgent.lua").read_text())
    cmd = lua.globals().CAPTURED_COMMAND
    argv = shlex.split(cmd)
    index = argv.index("main.py")
    parsed = main.parse_args(argv[index + 1 : -1])  # remove shell redirection 2>&1
    assert parsed.audio == lua.globals().AUDIO_PATH
    assert parsed.profile == "rock_drop_d"
    assert parsed.instrument == "Bass"
    assert main.resolve_thresholds(parsed, {"onset_threshold": 0.1, "frame_threshold": 0.2}) == (
        0.62,
        0.31,
    )
    assert parsed.output_dir == str(tmp_path / "out dir")
    assert parsed.no_midi and not parsed.no_tab and parsed.no_json


@pytest.mark.xfail(strict=True, reason="Lua double quotes permit POSIX shell command substitution")
def test_shell_metacharacters_in_audio_path_remain_literal(tmp_path, monkeypatch):
    pytest.importorskip("lupa")
    install = tmp_path / "install"
    install.mkdir()
    # The substitution is intentionally harmless and confined to pytest's tmp dir.
    audio = str(tmp_path / "$(touch injected).wav")
    (install / "main.py").write_text(
        "import json,sys\nopen('argv.json','w').write(json.dumps(sys.argv[1:]))\n"
    )
    monkeypatch.setattr(guard, "INSTALL_PATH", str(install))
    monkeypatch.setattr(guard, "AUDIO_PATH", audio)
    cmd = guard.run_lua({"install_path": str(install)})
    assert cmd
    subprocess.run(cmd, shell=True, cwd=tmp_path, check=True, env=os.environ.copy())
    assert not (install / "injected").exists()
    assert json.loads((install / "argv.json").read_text())[0] == audio


@pytest.mark.xfail(
    strict=True, reason="failed pipeline imports stale MIDI before checking exit status"
)
def test_failed_pipeline_never_imports_stale_midi(tmp_path):
    lupa = pytest.importorskip("lupa")
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    lua.execute("EXTSTATE_INIT = {}")
    lua.globals().INSTALL_PATH = str(tmp_path)
    lua.globals().AUDIO_PATH = str(tmp_path / "song.wav")
    lua.execute(guard.STUB)
    lua.execute("""
      EXTSTATE_INIT.install_path = INSTALL_PATH
      IMPORTED = 0
      reaper.file_exists = function(path) return true end
      reaper.InsertMedia = function(path, mode) IMPORTED = IMPORTED + 1 end
      reaper.defer = function(callback) callback() end
      io.popen = function(cmd, mode)
        return {read = function() return nil end, close = function() return nil, "exit", 1 end}
      end
    """)
    lua.execute((ROOT / "reaper/TabAgent.lua").read_text())
    assert "Error" in lua.globals().MB_TITLES
    assert lua.globals().IMPORTED == 0
