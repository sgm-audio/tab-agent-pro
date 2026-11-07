#!/usr/bin/env python3
"""
Execute reaper/TabAgent.lua with a stubbed REAPER API and verify that the
settings written by reaper/Settings.lua are turned into CLI flags for main.py.

Background (docs-audit P1-3): Settings.lua persisted profile/instrument/
thresholds/export flags but TabAgent.lua only ever read install_path, so every
setting was silently ignored.

Requires `lupa` (pip install lupa). Skips cleanly when lupa is unavailable.
"""

from __future__ import annotations

import atexit
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTALL_PATH = tempfile.mkdtemp(prefix="tab-agent-fake-install")
AUDIO_PATH = str(Path(INSTALL_PATH) / "song.wav")
OUTPUT_DIR = tempfile.mkdtemp(prefix="tab-agent-fake-output")


def _cleanup_temp_dirs() -> None:
    shutil.rmtree(INSTALL_PATH, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


atexit.register(_cleanup_temp_dirs)

FAILURES: list[str] = []
CAPTURED: list[str] = []

STUB = r"""
reaper = {}
local extstate = EXTSTATE_INIT
reaper.GetExtState = function(_, key) return extstate[key] or "" end
reaper.SetExtState = function(_, key, value) extstate[key] = value end
reaper.file_exists = function(path) return path == AUDIO_PATH or path == INSTALL_PATH .. "/main.py" end
reaper.GetResourcePath = function() return "/tmp/reaper-resources" end
reaper.GetOS = function() return "Linux_x86_64" end
MB_TITLES = ""
reaper.MB = function(msg, title, kind) MB_TITLES = MB_TITLES .. "|" .. tostring(title) end
reaper.ShowConsoleMsg = function(msg) end
reaper.Undo_BeginBlock = function() end
reaper.Undo_EndBlock = function(a, b) end
reaper.GetSelectedMediaItem = function(_, idx) return 1 end
reaper.GetActiveTake = function(item) return 1 end
reaper.GetMediaItemTake_Source = function(take) return 1 end
reaper.GetMediaSourceFileName = function(src, buf) return "", AUDIO_PATH end
reaper.InsertMedia = function(path, mode) end
reaper.GetNumTracks = function() return 1 end
reaper.GetTrack = function(_, idx) return 1 end
reaper.GetSetMediaTrackInfo_String = function(track, name, value, setnew) end

-- Intercept the shell command (the script calls os.execute(cmd))
local real_execute = os.execute
os.execute = function(cmd) CAPTURED_COMMAND = cmd; return 0 end
"""


def run_lua(extstate: dict[str, str]) -> str | None:
    from lupa import LuaRuntime

    lua = LuaRuntime(unpack_returned_tuples=True)
    lua.execute("EXTSTATE_INIT = {}")
    for key, value in extstate.items():
        lua.execute(f'EXTSTATE_INIT["{key}"] = "{value}"')
    lua.execute(f"AUDIO_PATH = '{AUDIO_PATH}'")
    lua.execute(f"INSTALL_PATH = '{INSTALL_PATH}'")
    lua.execute(STUB)
    lua.execute((ROOT / "reaper" / "TabAgent.lua").read_text())
    captured = lua.globals().CAPTURED_COMMAND
    return str(captured) if captured is not None else None


def expect(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label} {detail}")
        FAILURES.append(label)


def main() -> int:
    try:
        import lupa  # noqa: F401
    except ImportError:
        print("lupa not installed — skipping REAPER settings check")
        return 0

    print("REAPER settings → CLI flag propagation:")

    # 1. Full settings set
    extstate = {
        "install_path": INSTALL_PATH,
        "profile": "rock_drop_d",
        "instrument": "Bass",
        "onset": "0.62",
        "frame": "0.31",
        "output_dir": OUTPUT_DIR,
        "export_midi": "1",
        "export_tab": "0",
        "export_json": "1",
    }
    cmd = run_lua(extstate)
    expect("command was executed", bool(cmd))
    if cmd:
        expect("passes --profile", "--profile" in cmd and "rock_drop_d" in cmd, cmd)
        expect("passes --instrument", "--instrument" in cmd and "Bass" in cmd, cmd)
        expect("passes --onset", "--onset 0.62" in cmd, cmd)
        expect("passes --frame", "--frame 0.31" in cmd, cmd)
        expect("passes --no-tab for a disabled format", "--no-tab" in cmd, cmd)
        expect(
            "does not disable enabled formats",
            "--no-midi" not in cmd and "--no-json" not in cmd,
            cmd,
        )
        expect("passes --output-dir", "--output-dir" in cmd and OUTPUT_DIR in cmd, cmd)
        expect(
            "quotes the audio path",
            re.search(r"main\.py \"[^\"]*song\.wav\"", cmd) is not None,
            cmd,
        )

    # 2. Defaults: nothing but install_path stored
    cmd = run_lua({"install_path": INSTALL_PATH})
    expect("works with no stored settings", bool(cmd))
    if cmd:
        expect(
            "no spurious --no-* flags by default", "--no-midi" not in cmd and "--no-tab" not in cmd
        )

    # 3. Blank profile/thresholds must not add flags
    cmd = run_lua({"install_path": INSTALL_PATH, "profile": "", "onset": "", "frame": ""})
    if cmd:
        expect("blank profile omits --profile", "--profile" not in cmd, cmd)
        expect("blank thresholds omit flags", "--onset" not in cmd and "--frame" not in cmd, cmd)

    # 4. Disabled MIDI export (--no-midi) must not trigger "Tab Agent — No Output" dialog
    from lupa import LuaRuntime

    lua = LuaRuntime(unpack_returned_tuples=True)
    lua.execute("EXTSTATE_INIT = {}")
    lua.execute(f'EXTSTATE_INIT["install_path"] = "{INSTALL_PATH}"')
    lua.execute('EXTSTATE_INIT["export_midi"] = "0"')
    lua.execute(f"AUDIO_PATH = '{AUDIO_PATH}'")
    lua.execute(f"INSTALL_PATH = '{INSTALL_PATH}'")
    lua.execute(STUB)
    lua.execute((ROOT / "reaper" / "TabAgent.lua").read_text())
    no_midi_cmd = str(lua.globals().CAPTURED_COMMAND or "")
    mb_titles = str(lua.globals().MB_TITLES or "")
    expect("export_midi=0 passes --no-midi", "--no-midi" in no_midi_cmd, no_midi_cmd)
    expect("export_midi=0 does not show No Output error", "No Output" not in mb_titles, mb_titles)

    print(f"\n{'FAILED' if FAILURES else 'All REAPER settings checks passed.'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
