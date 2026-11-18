#!/usr/bin/env python3
"""Exercise the Settings and TabAgent ReaScripts with a local Lua API fixture.

Requires ``lupa``. No shell commands or real DAW operations are performed.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
# Include spaces so removing argument quotes can't sneak through the check.
INSTALL_PATH = "/fixture/Tab Agent install"
AUDIO_PATH = f"{INSTALL_PATH}/my song.wav"
OUTPUT_DIR = "/fixture/Tab Agent output"

STUB = r"""
reaper = {}
CAPTURED_COMMAND = nil
DIALOGS = ""
IMPORTED = 0
CONSOLE = ""
local callbacks = {}

reaper.GetExtState = function(_, key) return EXTSTATE_INIT[key] or "" end
reaper.SetExtState = function(_, key, value) EXTSTATE_INIT[key] = value end
reaper.GetUserInputs = function(_, count)
  assert(count == 9)
  return true, SETTINGS_FORM
end
reaper.file_exists = function(path)
  assert(type(path) == "string", "file_exists expects a string")
  local dir = EXTSTATE_INIT.output_dir or INSTALL_PATH .. "/output"
  return path == AUDIO_PATH or path == INSTALL_PATH .. "/main.py"
    or (HAS_MIDI and path == dir .. "/my song_lead_guitar.mid")
end
reaper.GetResourcePath = function() return "/tmp/reaper-resources" end
reaper.GetOS = function() return "Linux_x86_64" end
reaper.MB = function(_, title) DIALOGS = DIALOGS .. "|" .. title end
reaper.ShowConsoleMsg = function(msg) CONSOLE = CONSOLE .. msg end
reaper.Undo_BeginBlock = function() end
reaper.Undo_EndBlock = function() end
reaper.GetSelectedMediaItem = function(_, idx) return 1 end
reaper.GetActiveTake = function(item) return 1 end
reaper.GetMediaItemTake_Source = function(take) return 1 end
-- Lua returns only the filename, not a separate success code.
reaper.GetMediaSourceFileName = function(src, buf) return SOURCE_NAME end
reaper.InsertMedia = function(path, mode) IMPORTED = IMPORTED + 1 end
reaper.GetNumTracks = function() return 1 end
reaper.GetTrack = function(_, idx) return 1 end
reaper.GetSetMediaTrackInfo_String = function() end
reaper.defer = function(callback) table.insert(callbacks, callback) end
function run_callbacks()
  for i = 1, 10 do
    if #callbacks == 0 then return end
    table.remove(callbacks, 1)()
  end
  error("pipeline did not reach completion")
end

io.popen = function(cmd, mode)
  CAPTURED_COMMAND = cmd
  local lines = {"[tab-agent] running"}
  return {
    read = function() return table.remove(lines, 1) end,
    close = function() return (EXIT_CODE == 0), "exit", EXIT_CODE end,
  }
end
os.execute = function() error("unexpected os.execute in test") end
"""


class Result(NamedTuple):
    command: str | None
    dialogs: str
    imported: int
    console: str


def run_lua(
    extstate: dict[str, str],
    *,
    settings_form: str | None = None,
    source_name: str | None = AUDIO_PATH,
    has_midi: bool = False,
    exit_code: int = 0,
) -> Result:
    from lupa import LuaRuntime

    lua = LuaRuntime()
    env = lua.globals()
    env.EXTSTATE_INIT = lua.table_from(extstate)
    env.INSTALL_PATH = INSTALL_PATH
    env.AUDIO_PATH = AUDIO_PATH
    env.SOURCE_NAME = source_name
    env.HAS_MIDI = has_midi
    env.EXIT_CODE = exit_code
    env.SETTINGS_FORM = settings_form
    lua.execute(STUB)
    if settings_form is not None:
        lua.execute((ROOT / "reaper" / "Settings.lua").read_text())
    lua.execute((ROOT / "reaper" / "TabAgent.lua").read_text())
    lua.execute("run_callbacks()")
    return Result(env.CAPTURED_COMMAND, env.DIALOGS, env.IMPORTED, env.CONSOLE)


def main() -> int:
    try:
        import lupa  # noqa: F401
    except ImportError:
        print("Install lupa to run the REAPER settings check.", file=sys.stderr)
        return 1

    defaults = {"install_path": INSTALL_PATH}
    configured = run_lua(
        defaults, settings_form=f"5,{INSTALL_PATH},Bass,0.62,0.31,{OUTPUT_DIR},0,1,0"
    )
    words = shlex.split(configured.command or "")
    # The action may add interpreter flags; check the script's argument vector.
    args = words[words.index("main.py") + 1 : -1] if "main.py" in words else []
    no_tab = run_lua({**defaults, "export_tab": "0", "profile": "", "onset": "", "frame": ""})
    normal = run_lua(defaults, has_midi=True)
    no_output = run_lua(defaults)
    failed = run_lua(defaults, exit_code=1, has_midi=True)
    missing_source = run_lua(defaults, source_name=None)
    checks = {
        "Settings.lua form reaches main.py with spaced paths and correct flags": args
        == [
            AUDIO_PATH,
            "--profile",
            "bass_4_string",
            "--instrument",
            "Bass",
            "--onset",
            "0.62",
            "--frame",
            "0.31",
            "--no-midi",
            "--no-json",
            "--output-dir",
            OUTPUT_DIR,
        ],
        "MIDI disabled: completion does not import or show No Output": bool(configured.command)
        and configured.imported == 0
        and "No Output" not in configured.dialogs,
        "blank overrides are omitted and a disabled tab adds only --no-tab": [
            word for word in shlex.split(no_tab.command or "") if word.startswith("--")
        ]
        == ["--no-tab"],
        "default settings launch with no override flags": bool(normal.command)
        and "--" not in (normal.command or ""),
        "deferred output forwarded and MIDI imported on success": normal.imported == 1
        and "[tab-agent] running" in normal.console,
        "enabled MIDI with no files shows No Output": no_output.imported == 0
        and "No Output" in no_output.dialogs,
        "failed child never imports existing MIDI or reports Done": failed.imported == 0
        and "Error" in failed.dialogs
        and "No Output" not in failed.dialogs
        and "Done" not in failed.console,
        "missing source gives a Glue Required error, without launching": missing_source.command
        is None
        and "Glue Required" in missing_source.dialogs,
    }
    for label, ok in checks.items():
        print(f"  {'ok' if ok else 'FAIL'} {label}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
