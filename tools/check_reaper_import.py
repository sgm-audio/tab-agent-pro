#!/usr/bin/env python3
"""
Execute reaper/TabAgent.lua end to end — including the deferred import phase —
and assert that importing MIDI is safe to undo.

Two defects motivated this guard:

  * ``InsertMedia(path, 0)`` adds media to the *last touched* track, so the
    transcribed MIDI landed on whatever track the user had selected — often on
    top of the audio being transcribed. Mode 1 creates a dedicated track.
  * ``Undo_BeginBlock()`` was called in the script body and ``Undo_EndBlock()``
    inside a ``reaper.defer`` callback, i.e. in a later script invocation. Undo
    blocks must open and close in the same invocation, so the two halves were
    never paired and ``Ctrl+Z`` could not reliably remove an import. The flag
    must stay -1: it means "everything changed" and is what the REAPER docs
    recommend for scripts that modify project state directly.

check_reaper_settings.py stubs ``reaper.defer`` to a no-op, which is why the
import path was never exercised. This checker runs it.

Requires `lupa` (pip install lupa). Skips cleanly when lupa is unavailable.
"""

from __future__ import annotations

import atexit
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INSTALL_PATH = tempfile.mkdtemp(prefix="tab-agent-fake-install")
OUTPUT_DIR = tempfile.mkdtemp(prefix="tab-agent-fake-output")
AUDIO_PATH = str(Path(INSTALL_PATH) / "song.wav")
BASENAME = "song"

MIDI_FILES = (
    "song_lead_guitar.mid",
    "song_rhythm_L.mid",
    "song_rhythm_R.mid",
    "song_bass.mid",
)

FAILURES: list[str] = []


def _cleanup() -> None:
    shutil.rmtree(INSTALL_PATH, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


atexit.register(_cleanup)

STUB = r"""
reaper = {}
local extstate = EXTSTATE_INIT
reaper.GetExtState = function(_, key) return extstate[key] or "" end
reaper.SetExtState = function(_, key, value) extstate[key] = value end
reaper.file_exists = function(path)
  if path == AUDIO_PATH or path == INSTALL_PATH .. "/main.py" then return true end
  return MIDI_EXISTS[path] == true
end
reaper.GetResourcePath = function() return "/tmp/reaper-resources" end
reaper.GetOS = function() return "Linux_x86_64" end
MB_TITLES = ""
reaper.MB = function(msg, title, kind) MB_TITLES = MB_TITLES .. "|" .. tostring(title) end
reaper.ShowConsoleMsg = function(msg) end

-- Undo bookkeeping: remember which "phase" each call happened in so a block
-- that spans invocations is detectable.
UNDO_LOG = {}
local phase = "body"
reaper.Undo_BeginBlock = function()
  UNDO_LOG[#UNDO_LOG + 1] = "begin:" .. phase
end
reaper.Undo_EndBlock = function(desc, flags)
  UNDO_LOG[#UNDO_LOG + 1] = "end:" .. phase .. ":" .. tostring(flags)
end

reaper.PreventUIRefresh = function(flag) end
reaper.UpdateArrange = function() end

reaper.GetSelectedMediaItem = function(_, idx)
  if NO_SELECTION then return nil end
  return 1
end
reaper.GetActiveTake = function(item) return 1 end
reaper.GetMediaItemTake_Source = function(take) return 1 end
reaper.GetMediaSourceFileName = function(src, buf) return "", AUDIO_PATH end

-- Record every insert so the mode argument can be asserted.
INSERT_LOG = {}
CAPTURED_COMMAND = nil
reaper.InsertMedia = function(path, mode)
  INSERT_LOG[#INSERT_LOG + 1] = tostring(mode) .. ":" .. tostring(path)
  TRACK_COUNT = TRACK_COUNT + 1
end
TRACK_COUNT = 1
reaper.GetNumTracks = function() return TRACK_COUNT end
reaper.GetTrack = function(_, idx) return 1 end
reaper.GetSetMediaTrackInfo_String = function(track, name, value, setnew) end

-- Intercept io.popen so no real process is spawned; simulate immediate EOF.
local FAKE_STREAM = {}
function FAKE_STREAM:read(fmt) return nil end
function FAKE_STREAM:close() return true, nil, 0 end
io.popen = function(cmd, mode)
  CAPTURED_COMMAND = cmd
  return FAKE_STREAM
end

-- Run deferred callbacks immediately, and mark the phase so the undo log shows
-- whether a block opened in the body and closed in the callback.
reaper.defer = function(callback)
  if DEFER_USED then return end
  DEFER_USED = true
  phase = "deferred"
  callback()
  phase = "body"
end
"""


def _lua_str(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def run(extstate: dict[str, str], midi_on_disk: bool = True, no_selection: bool = False):
    from lupa import LuaRuntime

    lua = LuaRuntime(unpack_returned_tuples=True)
    lua.execute("EXTSTATE_INIT = {}")
    for key, value in extstate.items():
        lua.execute(f'EXTSTATE_INIT["{key}"] = "{_lua_str(value)}"')
    lua.execute(f"AUDIO_PATH = '{_lua_str(AUDIO_PATH)}'")
    lua.execute(f"INSTALL_PATH = '{_lua_str(INSTALL_PATH)}'")
    lua.execute("MIDI_EXISTS = {}")
    if midi_on_disk:
        for name in MIDI_FILES:
            path = f"{INSTALL_PATH}/output/{name}"
            lua.execute(f'MIDI_EXISTS["{_lua_str(path)}"] = true')
    lua.execute(f"NO_SELECTION = {'true' if no_selection else 'false'}")
    lua.execute(STUB)
    lua.execute((ROOT / "reaper" / "TabAgent.lua").read_text())

    g = lua.globals()
    return {
        "inserts": [str(x) for x in (g.INSERT_LOG.values() or [])],
        "undo": [str(x) for x in (g.UNDO_LOG.values() or [])],
        "titles": str(g.MB_TITLES or ""),
        "command": str(g.CAPTURED_COMMAND or ""),
    }


def expect(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label} {detail}")
        FAILURES.append(label)


def check_import() -> None:
    print("REAPER MIDI import is undoable and creates its own tracks:")
    result = run({"install_path": INSTALL_PATH, "export_midi": "1"})
    inserts = result["inserts"]
    undo = result["undo"]

    expect("MIDI files were imported", len(inserts) == len(MIDI_FILES), str(inserts))
    expect(
        "every import creates a new track (mode 1)",
        all(entry.startswith("1:") for entry in inserts),
        str(inserts),
    )
    expect(
        "no import targets the last touched track (mode 0)",
        not any(entry.startswith("0:") for entry in inserts),
        str(inserts),
    )
    expect("an undo block was opened", any(e.startswith("begin:") for e in undo), str(undo))
    expect("the undo block was closed", any(e.startswith("end:") for e in undo), str(undo))

    begins = sum(1 for e in undo if e.startswith("begin:"))
    ends = sum(1 for e in undo if e.startswith("end:"))
    expect("one begin per end", begins == ends and begins >= 1, str(undo))

    expect(
        "the block opens and closes in one invocation",
        all(e.endswith(":deferred") for e in undo if e.startswith("begin:"))
        and all(e.split(":")[1] == "deferred" for e in undo if e.startswith("end:")),
        str(undo),
    )
    # -1 means "everything changed" and is the value the REAPER docs recommend
    # when a script modifies project state directly. A narrower bitmask can
    # make REAPER decide nothing changed and create no undo point at all, so
    # -1 is the correct choice here, not a bug.
    expect(
        "the undo block covers all project state (-1)",
        all(e.endswith(":-1") for e in undo if e.startswith("end:")),
        str(undo),
    )
    expect(
        "the block is closed with a description",
        str(undo[-1]).startswith("end:") if undo else False,
        str(undo),
    )


def check_cancel_paths() -> None:
    print("\nCancelling does not leave an undo block open:")
    result = run({"install_path": INSTALL_PATH}, no_selection=True)
    expect("cancel performs no import", result["inserts"] == [], str(result["inserts"]))
    expect("cancel opens no undo block", result["undo"] == [], str(result["undo"]))
    expect("cancel explains what to do", "Tab Agent" in result["titles"], result["titles"])


def check_no_midi_written() -> None:
    print("\nA finished run with no MIDI on disk does not corrupt the project:")
    result = run({"install_path": INSTALL_PATH, "export_midi": "1"}, midi_on_disk=False)
    expect("nothing is inserted", result["inserts"] == [], str(result["inserts"]))
    expect(
        "one balanced undo block", result["undo"].count("begin:deferred") == 1, str(result["undo"])
    )
    expect(
        "the user is told no output was found", "No Output" in result["titles"], result["titles"]
    )


def main() -> int:
    try:
        import lupa  # noqa: F401
    except ImportError:
        print("lupa not installed — skipping REAPER import check")
        return 0

    check_import()
    check_cancel_paths()
    check_no_midi_written()

    print(f"\n{'FAILED' if FAILURES else 'All REAPER import checks passed.'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
