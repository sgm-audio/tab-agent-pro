# ruff: noqa
#!/usr/bin/env python3
"""
Drive the REAL reaper/TabAgent.lua under a stubbed REAPER API (lupa).

  1. `Undo_BeginBlock()` is called in the script body, `Undo_EndBlock()` only on
     a later defer tick (or never, if the user aborts / closes REAPER).
  2. `InsertMedia()` runs against whatever project is active when the poll
     finally completes - the script never captures the project it started in.
  3. `stream:read("*l")` inside the deferred callback blocks that callback
     (i.e. REAPER's main thread) until the child prints a newline.
  4. ExtState values are interpolated into a shell command with `"` quoting.

Usage: /tmp/pocvenv/bin/python audit_poc/reaper_probe.py
"""

import os
import tempfile
import time
from pathlib import Path

from lupa import LuaRuntime

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "reaper"

OUTPUT_DIR = tempfile.mkdtemp(prefix="tabagent-out-")
BASENAME = "song"
for suffix in ("lead_guitar", "rhythm_L", "rhythm_R", "bass"):
    Path(OUTPUT_DIR, f"{BASENAME}_{suffix}.mid").write_bytes(b"MThd")

STUB = r"""
reaper = {}
EVENTS = {}
PENDING = {}
ACTIVE_PROJECT = "PROJECT-A"
local extstate = EXTSTATE_INIT

local function log(ev, detail) EVENTS[#EVENTS+1] = ev .. (detail and (":" .. detail) or "") end

reaper.GetExtState = function(_, key) return extstate[key] or "" end
reaper.SetExtState = function(_, key, value) extstate[key] = value end
reaper.file_exists = function(path)
  if path == AUDIO_PATH or path == INSTALL_PATH .. "/main.py" then return true end
  return real_file_exists(path)
end
reaper.GetResourcePath = function() return "/tmp/reaper-resources" end
reaper.GetOS = function() return "Linux_x86_64" end
reaper.MB = function(msg, title, kind) log("MB", title); return 0 end
reaper.ShowConsoleMsg = function(msg) end
reaper.Undo_BeginBlock = function() log("Undo_BeginBlock") end
reaper.Undo_EndBlock = function(desc, flags) log("Undo_EndBlock") end
reaper.GetSelectedMediaItem = function(_, idx) return 1 end
reaper.GetActiveTake = function(item) return 1 end
reaper.GetMediaItemTake_Source = function(take) return 1 end
reaper.GetMediaSourceFileName = function(src, buf) return "", AUDIO_PATH end
reaper.InsertMedia = function(path, mode) log("InsertMedia_into_" .. ACTIVE_PROJECT) end
reaper.GetNumTracks = function() return 1 end
reaper.GetTrack = function(_, idx) return 1 end
reaper.GetSetMediaTrackInfo_String = function(track, name, value, setnew) end
reaper.EnumProjects = function(idx, projfn) return ACTIVE_PROJECT end
reaper.defer = function(cb) PENDING[#PENDING+1] = cb end
"""


def setup(settings, popen):
    lua = LuaRuntime(unpack_returned_tuples=True)
    lua.execute("EXTSTATE_INIT = {}")
    for key, value in settings.items():
        lua.execute(f'EXTSTATE_INIT["{key}"] = "{value}"')
    lua.execute("AUDIO_PATH = '/tmp/song.wav'")
    lua.execute("INSTALL_PATH = '/tmp/fake-install'")
    lua.execute(STUB)
    lua.globals().real_file_exists = os.path.isfile
    lua.globals().popen_impl = popen
    lua.execute("io.popen = popen_impl")
    lua.execute("CAPTURED = nil")
    lua.execute((SCRIPTS / "TabAgent.lua").read_text())
    return lua


def events(lua):
    ev = lua.globals().EVENTS
    return [ev[i] for i in range(1, len(ev) + 1)]


def take_callback(lua):
    p = lua.globals().PENDING
    n = len(p)
    if n == 0:
        return None
    cb = p[1]
    lua.execute("PENDING = {}")
    return cb


class FakeStream:
    def __init__(self, lines, delay=0.0):
        self.lines = list(lines)
        self.delay = delay
        self.started = None

    def read(self, fmt):
        if self.started is None:
            self.started = time.time()
        if self.delay:
            time.sleep(self.delay)
            self.delay = 0.0
        return self.lines.pop(0) if self.lines else None

    def close(self):
        return True, None, 0


LAST_CMD = {}


def factory(lines, delay=0.0):
    def _popen(cmd, mode):
        LAST_CMD["cmd"] = cmd
        return FakeStream(lines, delay)

    return _popen


print("=== A. undo block spans idle cycles + project switch ===")
lua = setup(
    {"install_path": "/tmp/fake-install", "output_dir": OUTPUT_DIR}, factory(["l1\n", "l2\n"])
)
print("script body          :", events(lua))
lua.globals().ACTIVE_PROJECT = "PROJECT-B"  # user switches project tab while it runs
tick = 0
while True:
    cb = take_callback(lua)
    if cb is None:
        break
    tick += 1
    cb()
    print(f"defer tick {tick}          :", events(lua))
print("=> one Undo_BeginBlock, Undo_EndBlock only after", tick, "defer ticks")
print("=> MIDI imported into:", [e for e in events(lua) if e.startswith("InsertMedia")])

print("\n=== B. read('*l') blocks the deferred callback ===")
lua = setup({"install_path": "/tmp/fake-install"}, factory(["late\n"], delay=3.0))
cb = take_callback(lua)
t0 = time.time()
cb()
print(f"one poll() blocked the main thread for {time.time() - t0:.2f}s while the child was quiet")

print("\n=== C. quoting of ExtState values ===")
lua = setup(
    {"install_path": "/tmp/fake-install", "profile": "$(echo INJECTED > /tmp/tabagent_pwned)"},
    factory([]),
)
print("generated command:")
print("   ", LAST_CMD["cmd"])

# Prove the substitution really happens in sh (harmless sentinel file).
marker = "/tmp/tabagent_pwned"
if os.path.exists(marker):
    os.unlink(marker)
os.makedirs("/tmp/fake-install", exist_ok=True)
os.system(LAST_CMD["cmd"].replace("python3", "true") + " >/dev/null 2>&1")
print("sentinel file created by shell substitution:", os.path.exists(marker))
