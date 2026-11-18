"""Run both real Lua scripts against the documented REAPER API shapes."""

import json
import os
import shlex
import subprocess
from pathlib import Path

import pretty_midi
import pytest
import soundfile as sf
from lupa import LuaRuntime

import agents
import init_memory
import main
from tests.audio_fixtures import SAMPLE_RATE, plucks, tab_pitches

REAPER = Path(__file__).resolve().parents[1] / "reaper"

STUB = r"""
EVENTS = { dialogs = {}, console = {}, inserted = {}, renamed = {}, undo = {} }
local callbacks, tracks = {}, { "existing-selected", "existing-last" }
local function add(list, value) table.insert(list, value) end
reaper = {
  GetOS = function() return PLATFORM end,
  GetExtState = function(section, key)
    assert(section == "TabAgent")
    return EXTSTATE[key] or ""
  end,
  SetExtState = function(section, key, value, persist)
    assert(section == "TabAgent" and persist and type(value) == "string")
    EXTSTATE[key] = value
  end,
  GetUserInputs = function(_, fields, _, defaults)
    assert(fields == 9 and type(defaults) == "string")
    return DIALOG_INPUT ~= nil, DIALOG_INPUT
  end,
  GetResourcePath = function() return "/no-reaper-install-here" end,
  GetSelectedMediaItem = function() return HAS_ITEM and 1 or nil end,
  GetActiveTake = function() return HAS_TAKE and 1 or nil end,
  GetMediaItemTake_Source = function() return 1 end,
  -- REAPER Lua returns the filename as its only value (no success flag).
  GetMediaSourceFileName = function() return AUDIO end,
  file_exists = function(path)
    assert(type(path) == "string", "file_exists expects a filename")
    return EXISTS[path] == true
  end,
  EnumProjects = function(index) assert(index == -1); return CURRENT_PROJECT, "" end,
  MB = function(message, title) add(EVENTS.dialogs, title .. ": " .. message) end,
  ShowConsoleMsg = function(message) add(EVENTS.console, message) end,
  Undo_BeginBlock = function() add(EVENTS.undo, "begin") end,
  Undo_EndBlock = function() add(EVENTS.undo, "end") end,
  defer = function(callback) add(callbacks, callback) end,
  InsertMedia = function(path, mode)
    add(EVENTS.inserted, { path, mode })
    if not INSERT_OK then return 0 end
    if mode == 1 then
      -- New track need not be the last track. Mode 0 modifies the selected one.
      table.insert(tracks, 2, "new-" .. #EVENTS.inserted)
    end
    return 1
  end,
  GetNumTracks = function() return #tracks end,
  GetTrack = function(_, i) return tracks[i + 1] end,
  GetTrackGUID = function(track) return track end,
  GetSetMediaTrackInfo_String = function(track, key, value, set)
    assert(key == "P_NAME" and set)
    add(EVENTS.renamed, { track, value })
  end,
}

function drain_callbacks()
  local count = 0
  while #callbacks > 0 do
    count = count + 1
    assert(count < 100, "polling never stopped")
    table.remove(callbacks, 1)()
  end
end

io.popen = function(command, mode)
  assert(mode == "r")
  EVENTS.command = command
  if not START_OK then return nil end
  local line = 0
  return {
    read = function(_, fmt)
      assert(fmt == "*l")
      line = line + 1
      return CHILD_LINES[line]
    end,
    close = function()
      assert(not EVENTS.closed, "the pipe was closed twice")
      EVENTS.closed = true
      if EXIT_CODE == 0 then return true, "exit", 0 end
      return nil, "exit", EXIT_CODE
    end,
  }
end
local session_number = 0
os.tmpname = function()
  if not TEMP_OK then error("temporary directory unavailable") end
  session_number = session_number + 1
  return "/tmp/session-" .. session_number
end
os.remove = function(path)
  assert(path:match("^/tmp/session%-%d+$"))
  return true
end
os.execute = function() error("unexpected shell execution") end
"""


class ReaperHarness:
    def __init__(self, install, audio, *, platform="Linux_x86_64"):
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        self.globals = self.lua.globals()
        g = self.globals
        g.PLATFORM = platform
        g.AUDIO = str(audio)
        g.EXTSTATE = self.lua.table_from({"install_path": str(install)})
        g.EXISTS = self.lua.table_from({f"{install}/main.py": True, str(audio): True})
        g.CURRENT_PROJECT = "proj-a"
        g.CHILD_LINES = self.lua.table()
        g.EXIT_CODE = 0
        g.HAS_ITEM = g.HAS_TAKE = g.INSERT_OK = g.START_OK = g.TEMP_OK = True
        self.lua.execute(STUB)

    def run(self, script="TabAgent.lua"):
        self.lua.execute((REAPER / script).read_text(encoding="utf-8"), name=script)

    def finish(self, *lines):
        self.globals.CHILD_LINES = self.lua.table_from(lines)
        self.globals.drain_callbacks()

    @property
    def command(self):
        return self.globals.EVENTS.command

    def cli_argv(self):
        tokens = shlex.split(self.command)
        return tokens[tokens.index("main.py") + 1 : -1]

    def events(self, key):
        events = self.globals.EVENTS[key]
        return [events[i] for i in range(1, len(events) + 1)]


@pytest.mark.parametrize("include_midi", [True, False], ids=["midi-json", "tab-only"])
def test_reaper_settings_dialog_reaches_real_cli_and_controls_outputs(
    tmp_path, monkeypatch, capsys, include_midi
):
    install = tmp_path / "Tab Agent install"
    install.mkdir()
    audio = tmp_path / "bass rehearsal.take.wav"
    sf.write(audio, plucks([(28, 0.25, 0.8)]), SAMPLE_RATE)
    host = ReaperHarness(install, audio)
    exports = "yes,0,1" if include_midi else "0,true,0"
    frame = "" if include_midi else "0.26"
    host.globals.DIALOG_INPUT = f"5,{install},bass,0.62,{frame},mix exports,{exports}"
    host.run("Settings.lua")
    settings_dialogs = host.events("dialogs")
    assert len(settings_dialogs) == 1 and "Settings saved" in settings_dialogs[0]
    assert dict(host.globals.EXTSTATE) == {
        "profile": "bass_4_string",
        "install_path": str(install),
        "instrument": "bass",
        "onset": "0.62",
        "frame": frame,
        "output_dir": "mix exports",
        "export_midi": "1" if include_midi else "0",
        "export_tab": "0" if include_midi else "1",
        "export_json": "1" if include_midi else "0",
    }
    host.run()
    tokens = shlex.split(host.command)
    assert tokens[:5] == ["cd", str(install), "&&", "python3", "main.py"]
    assert tokens[-1] == "2>&1"
    assert ("--frame" in tokens) == bool(frame)

    # Exercise actual Python parsing, saved profile, DSP, instrument range and
    # exporters. Only the two model boundaries are replaced; no weights needed.
    monkeypatch.chdir(install)
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setattr(init_memory, "get_memory_dir", lambda: str(tmp_path / "memory"))
    monkeypatch.setattr(agents, "DEMUCS_API_AVAILABLE", False)

    def offline(*args, **kwargs):
        raise FileNotFoundError("Demucs not installed in this scenario")

    monkeypatch.setattr(agents.SplitterAgent, "_separate_with_subprocess", offline)
    monkeypatch.setattr(agents, "BASIC_PITCH_AVAILABLE", True)
    calls = []

    def predict(path, **kwargs):
        calls.append((Path(path).name, kwargs["onset_threshold"], kwargs["frame_threshold"]))
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(0)
        instrument.notes = [pretty_midi.Note(80, p, 0.25, 0.8) for p in (23, 28)]
        midi.instruments.append(instrument)
        return None, midi, []

    monkeypatch.setattr(agents, "basic_pitch_predict", predict)
    output_root = install / "mix exports"
    output_root.mkdir()
    # This stem belongs to a previous guitar run and should never be imported.
    stale = output_root / f"{audio.stem}_lead_guitar.mid"
    stale.write_bytes(b"previous file")
    host.globals.EXISTS[str(stale)] = True
    opts = main.parse_args(tokens[5:-1])
    out = Path(opts.output_dir)
    assert opts.fresh_output_dir and out.parent == output_root and not out.exists()
    assert out.name.startswith("tab-agent-")
    assert main.main(tokens[5:-1]) == 0
    assert calls == [("processed_bass_clean.wav", 0.62, 0.26 if frame else 0.35)]
    assert stale.read_bytes() == b"previous file"
    fresh = {p.name for p in out.iterdir() if p.is_file()}
    extensions = ("mid", "json") if include_midi else ("tab",)
    assert fresh == {f"{audio.stem}_bass.{ext}" for ext in extensions}
    if include_midi:
        data = json.loads((out / f"{audio.stem}_bass.json").read_text())
        assert data["instrument"] == "4-String Bass" and data["num_strings"] == 4
        assert tab_pitches(data["tablature"], [28, 33, 38, 43], 24) == [28]
    else:
        tab = (out / f"{audio.stem}_bass.tab").read_text()
        assert "=== 4-String Bass Tablature ===" in tab
        assert "t(s)|0.250" in tab
        assert any(line.strip() == "E|0" for line in tab.splitlines())

    for path in out.iterdir():
        host.globals.EXISTS[str(path)] = path.is_file()
    host.finish(*capsys.readouterr().out.splitlines())
    imported = [[item[1], item[2]] for item in host.events("inserted")]
    bass_midi = str(out / f"{audio.stem}_bass.mid")
    assert imported == ([[bass_midi, 1]] if include_midi else [])
    assert host.events("undo") == (["begin", "end"] if include_midi else [])
    assert host.events("dialogs") == settings_dialogs


def test_settings_gui_to_lua_command_to_cli_parser(tmp_path):
    install = tmp_path / "install path"
    audio = tmp_path / "song name.wav"
    out = tmp_path / "out dir"
    host = ReaperHarness(install, audio)
    host.globals.DIALOG_INPUT = f"2,{install},Bass,0.62,0.31,{out},0,1,0"
    host.run("Settings.lua")
    host.run()
    tokens = shlex.split(host.command)
    parsed = main.parse_args(tokens[tokens.index("main.py") + 1 : -1])
    assert parsed.audio == str(audio)
    assert parsed.profile == "rock_drop_d"
    assert parsed.instrument == "Bass"
    assert main.resolve_thresholds(parsed, {"onset_threshold": 0.1, "frame_threshold": 0.2}) == (
        0.62,
        0.31,
    )
    assert parsed.fresh_output_dir and Path(parsed.output_dir).parent == out
    assert parsed.no_midi and not parsed.no_tab and parsed.no_json


def test_reaper_posix_literal_paths_reach_the_child_unchanged(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("requires a real POSIX shell")
    monkeypatch.setenv("TAB_AGENT_LITERAL", "wrong-if-expanded")
    install = tmp_path / "Project's $TAB_AGENT_LITERAL"
    install.mkdir()
    main_file = install / "main.py"
    main_file.write_text("import json, os, sys\nprint(json.dumps([os.getcwd(), sys.argv[1:]]))\n")
    audio = tmp_path / 'take $TAB_AGENT_LITERAL "雪".wav'
    host = ReaperHarness(install, audio)
    host.globals.EXTSTATE.output_dir = "export's $TAB_AGENT_LITERAL"
    host.run()

    # Replay only our inert test paths, against a tiny argv recorder, not the model.
    result = subprocess.run(
        ["/bin/sh", "-c", host.command], capture_output=True, text=True, check=True, timeout=10
    )
    cwd, argv = json.loads(result.stdout)
    assert Path(cwd).resolve() == install.resolve()
    opts = main.parse_args(argv)
    assert opts.audio == str(audio)
    assert opts.fresh_output_dir
    assert Path(opts.output_dir).parent == install / "export's $TAB_AGENT_LITERAL"


def test_reaper_windows_uses_drive_switch_and_rejects_expansion():
    install, audio = r"D:\Program Files (x86)\Tab Agent", r"C:\Songs\Take 01.wav"
    host = ReaperHarness(install, audio, platform="Win64")
    host.run()
    assert host.command.startswith(f'cd /d "{install}" && python main.py ')
    assert host.command.endswith(f'-- "{audio}" 2>&1')

    rejected = ReaperHarness(install, audio, platform="Win64")
    rejected.globals.EXTSTATE.output_dir = r"D:\mixed %TEMP%"
    rejected.run()
    assert rejected.command is None
    assert "unsupported by the Windows" in "".join(rejected.events("console"))
    assert any("Error" in dialog for dialog in rejected.events("dialogs"))


@pytest.mark.parametrize(
    "bad_input",
    [
        "no item",
        "no take",
        "no source file",
        "nil source filename",
        "start failure",
        "temp failure",
    ],
)
def test_reaper_no_project_mutation_on_selection_or_start_failure(bad_input):
    host = ReaperHarness("/repo", "/audio/take.wav")
    if bad_input == "no item":
        host.globals.HAS_ITEM = False
    elif bad_input == "no take":
        host.globals.HAS_TAKE = False
    elif bad_input == "no source file":
        host.globals.AUDIO = ""  # e.g. in-project MIDI
    elif bad_input == "nil source filename":
        host.globals.AUDIO = None
    elif bad_input == "temp failure":
        host.globals.TEMP_OK = False
    else:
        host.globals.START_OK = False
    host.run()
    host.finish()
    assert (host.command is not None) == (bad_input == "start failure")
    assert len(host.events("dialogs")) == 1
    assert host.events("inserted") == host.events("renamed") == host.events("undo") == []


@pytest.mark.parametrize(
    "abort", ["nonzero exit", "nonzero exit with partial output", "project changed", "stale file"]
)
def test_reaper_never_imports_failed_or_stale_midi(abort):
    host = ReaperHarness("/repo", "/audio/take.wav")
    host.globals.EXISTS["/repo/output/take_bass.mid"] = True
    host.run()
    assert host.events("undo") == []  # not open during inference
    if abort in ("nonzero exit with partial output", "project changed"):
        out = Path(main.parse_args(host.cli_argv()).output_dir)
        host.globals.EXISTS[str(out / "take_bass.mid")] = True
    if abort.startswith("nonzero exit"):
        host.globals.EXIT_CODE = 2
    elif abort == "project changed":
        host.globals.CURRENT_PROJECT = "proj-b"
    lines = ["[tab-agent] Transcribing"]
    if abort in ("nonzero exit with partial output", "project changed"):
        lines.append("[tab-agent-midi] bass")  # a partial or wrong-project export exists
    host.finish(*lines)
    assert host.globals.EVENTS.closed is True
    assert host.events("inserted") == host.events("renamed") == host.events("undo") == []
    dialogs = host.events("dialogs")
    assert len(dialogs) == 1
    if abort.startswith("nonzero exit"):
        assert "Error" in dialogs[0]
    assert "Done" not in "".join(host.events("console"))


@pytest.mark.parametrize("succeed", [True, False], ids=["track-created", "insert-failed"])
def test_reaper_renames_only_a_successfully_created_track(succeed):
    host = ReaperHarness("/repo", "/audio/part.multi.wav")
    host.globals.INSERT_OK = succeed
    host.run()
    out = Path(main.parse_args(host.cli_argv()).output_dir)
    midi_path = str(out / "part.multi_rhythm_R.mid")
    host.globals.EXISTS[midi_path] = True
    host.finish("[tab-agent-midi] rhythm_R")
    assert host.events("undo") == ["begin", "end"]
    assert [[entry[1], entry[2]] for entry in host.events("inserted")] == [[midi_path, 1]]
    assert [[entry[1], entry[2]] for entry in host.events("renamed")] == (
        [["new-1", "part.multi - Rhythm Guitar R"]] if succeed else []
    )
    assert len(host.events("dialogs")) == (0 if succeed else 1)


def test_reaper_invalid_threshold_is_not_persisted_or_silently_dropped():
    host = ReaperHarness("/repo", "/audio/take.wav")
    host.globals.EXTSTATE.frame = "0.35"
    host.globals.DIALOG_INPUT = "5,/repo,Bass,,bad-threshold,,1,1,1"
    host.run("Settings.lua")
    assert dict(host.globals.EXTSTATE) == {"install_path": "/repo", "frame": "0.35"}
    assert "Settings not saved" in host.events("dialogs")[-1]
    host.globals.EXTSTATE.frame = "1.5"  # corrupt persisted state also fails before launch
    host.run()
    assert host.command is None
    assert "Invalid frame threshold" in "".join(host.events("console"))


def test_reaper_uses_a_distinct_output_dir_for_each_run():
    host = ReaperHarness("/repo", "/audio/take.wav")
    host.run()
    first = main.parse_args(host.cli_argv())
    host.finish()
    host.globals.EVENTS.closed = None
    host.run()
    second = main.parse_args(host.cli_argv())
    host.finish()
    assert first.fresh_output_dir and second.fresh_output_dir
    assert first.output_dir != second.output_dir
    assert Path(first.output_dir).parent == Path(second.output_dir).parent == Path("/repo/output")


def test_reaper_output_collision_does_not_process_or_import(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "take.wav"
    audio.touch()
    host = ReaperHarness(tmp_path / "repo", audio)
    host.run()
    argv = host.cli_argv()
    out = Path(main.parse_args(argv).output_dir)
    out.mkdir(parents=True)
    stale = out / "take_bass.mid"
    stale.write_bytes(b"stale MIDI")
    host.globals.EXISTS[str(stale)] = True
    monkeypatch.setattr(main, "load_user_memory", lambda: pytest.fail("processed an occupied path"))

    with pytest.raises(SystemExit) as exc:
        main.main(argv)
    assert exc.value.code != 0
    child_output = capsys.readouterr()
    assert "Output directory already exists" in child_output.err
    assert "[tab-agent-midi]" not in child_output.out
    assert stale.read_bytes() == b"stale MIDI"
    host.globals.EXIT_CODE = exc.value.code
    host.finish(*child_output.out.splitlines())
    assert host.events("inserted") == host.events("renamed") == host.events("undo") == []
    assert len(host.events("dialogs")) == 1 and "Error" in host.events("dialogs")[0]
