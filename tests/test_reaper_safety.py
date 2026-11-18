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
    assert Path(parsed.output_dir).parent == tmp_path / "out dir"
    assert Path(parsed.output_dir).name.startswith("tab-agent-")
    assert parsed.fresh_output_dir
    assert parsed.no_midi and not parsed.no_tab and parsed.no_json


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
      UNDO_ENDS = 0
      reaper.Undo_EndBlock = function(...) UNDO_ENDS = UNDO_ENDS + 1 end
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
    assert lua.globals().UNDO_ENDS == 1


def test_all_posix_command_fields_are_literal(tmp_path):
    lupa = pytest.importorskip("lupa")
    install = tmp_path / "install 'quotes' $(touch injected)"
    install.mkdir()
    audio = str(tmp_path / "a'\"`touch injected`$HOME\\song.wav")
    output = str(tmp_path / "output 'quote' $(touch injected)")
    profile = "rock_'\"`touch injected`$HOME"
    (install / "main.py").write_text(
        "import json,sys\nopen('argv.json','w').write(json.dumps(sys.argv[1:]))\n"
    )
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    lua.execute("EXTSTATE_INIT = {}")
    lua.globals().INSTALL_PATH = str(install)
    lua.globals().AUDIO_PATH = audio
    lua.execute(guard.STUB)
    lua.globals().EXTSTATE_INIT.install_path = str(install)
    lua.globals().EXTSTATE_INIT.output_dir = output
    lua.globals().EXTSTATE_INIT.profile = profile
    lua.execute((ROOT / "reaper/TabAgent.lua").read_text())
    cmd = lua.globals().CAPTURED_COMMAND
    subprocess.run(cmd, shell=True, cwd=tmp_path, check=True, env=os.environ.copy())
    args = json.loads((install / "argv.json").read_text())
    assert args[0] == audio
    assert args[args.index("--profile") + 1] == profile
    assert Path(args[args.index("--output-dir") + 1]).parent == Path(output)
    assert "--fresh-output-dir" in args
    assert not (install / "injected").exists()
    assert not (tmp_path / "injected").exists()


def test_success_imports_only_new_session_files(tmp_path):
    lupa = pytest.importorskip("lupa")
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    lua.execute("EXTSTATE_INIT = {}")
    lua.globals().INSTALL_PATH = str(tmp_path / "install")
    lua.globals().AUDIO_PATH = str(tmp_path / "song.wav")
    lua.execute(guard.STUB)
    root = tmp_path / "outputs"
    root.mkdir()
    (root / "song_bass.mid").touch()  # stale file must not be imported
    lua.globals().EXTSTATE_INIT.install_path = str(tmp_path / "install")
    lua.globals().EXTSTATE_INIT.output_dir = str(root)
    lua.globals().exists_on_disk = lambda path: Path(path).is_file()
    lua.execute("""
      imports = {}
      local original_file_exists = reaper.file_exists
      reaper.file_exists = function(path) return original_file_exists(path) or exists_on_disk(path) end
      reaper.InsertMedia = function(path, mode) table.insert(imports, path) end
      reaper.defer = function(cb) POLL = cb end
    """)
    lua.execute((ROOT / "reaper/TabAgent.lua").read_text())
    argv = shlex.split(lua.globals().CAPTURED_COMMAND)
    session = Path(argv[argv.index("--output-dir") + 1])
    assert session.parent == root and not session.exists()
    session.mkdir()
    fresh = session / "song_lead_guitar.mid"
    fresh.touch()
    lua.globals().POLL()
    assert list(lua.globals().imports.values()) == [str(fresh)]
    assert "Error" not in lua.globals().MB_TITLES


@pytest.mark.parametrize("key", ["install_path", "audio", "profile", "output_dir"])
def test_windows_rejects_command_expansion_characters(tmp_path, key):
    lupa = pytest.importorskip("lupa")
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    lua.execute("EXTSTATE_INIT = {}")
    lua.globals().INSTALL_PATH = "C:/safe"
    lua.globals().AUDIO_PATH = "C:/safe/song.wav"
    lua.execute(guard.STUB)
    lua.globals().EXTSTATE_INIT.install_path = "C:/safe"
    lua.execute('reaper.GetOS = function() return "Win64" end')
    if key == "audio":
        lua.globals().AUDIO_PATH = "C:/safe/%EVIL%/song.wav"
    elif key == "install_path":
        lua.globals().INSTALL_PATH = "C:/safe/%EVIL%"
        lua.globals().EXTSTATE_INIT.install_path = lua.globals().INSTALL_PATH
    else:
        lua.globals().EXTSTATE_INIT[key] = "%EVIL%"
    lua.execute((ROOT / "reaper/TabAgent.lua").read_text())
    assert lua.globals().CAPTURED_COMMAND is None
    assert "Error" in lua.globals().MB_TITLES


def test_fresh_output_dir_rejects_existing_before_pipeline(tmp_path, monkeypatch):
    source = tmp_path / "song.wav"
    source.touch()
    output = tmp_path / "out"
    output.mkdir()
    stale = output / "song_bass.mid"
    stale.write_bytes(b"old")
    called = []
    monkeypatch.setattr(main, "process_suno_audio", lambda *a, **kw: called.append(True))
    with pytest.raises(FileExistsError):
        main.main([str(source), "--output-dir", str(output), "--fresh-output-dir"])
    assert stale.read_bytes() == b"old"
    assert not called


def test_missing_selection_cancels_without_start_and_closes_undo(tmp_path):
    lupa = pytest.importorskip("lupa")
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    lua.execute("EXTSTATE_INIT = {}")
    lua.globals().INSTALL_PATH = str(tmp_path)
    lua.globals().AUDIO_PATH = str(tmp_path / "song.wav")
    lua.execute(guard.STUB)
    lua.execute("""
      EXTSTATE_INIT.install_path = INSTALL_PATH
      UNDO_ENDS = 0
      reaper.Undo_EndBlock = function(...) UNDO_ENDS = UNDO_ENDS + 1 end
      reaper.GetSelectedMediaItem = function(...) return nil end
    """)
    lua.execute((ROOT / "reaper/TabAgent.lua").read_text())
    assert lua.globals().CAPTURED_COMMAND is None
    assert lua.globals().UNDO_ENDS == 1
