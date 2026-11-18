"""Adversarial end-to-end test of reaper/TabAgent.lua.

Drives the real Lua with a REAPER shim whose io.popen runs a real subprocess,
so the full start -> stream -> poll -> import path executes.

Covers the two REAPER P0 defects from the independent verification report:
  * a failed pipeline imported MIDI left over from an earlier session;
  * the audio path was interpolated into a /bin/sh command, so a filename
    containing $(...) executed.

The import gate under test is main's: the pipeline announces each MIDI file it
writes as a "[tab-agent-midi] <suffix>" line, poll_pipeline records those
suffixes, and imports only files from *this* run -- and only when the process
exited 0. There is no on-disk manifest to go stale.

Requires `lupa`; skips cleanly (exit 0) when it is unavailable, matching
tools/check_reaper_settings.py.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

try:
    import lupa
except ImportError:  # pragma: no cover - optional dev dependency
    lupa = None

# Derive from this file's location, not a hardcoded checkout path: CI runs from
# /home/runner/work/..., and an absolute sandbox path silently breaks there.
ROOT = Path(__file__).resolve().parent.parent
MIDI_SUFFIXES = ("lead_guitar", "rhythm_L", "rhythm_R", "bass")

STUB = r"""
reaper = {}
local extstate = {install_path = INSTALL, output_dir = OUTDIR,
                  export_midi="1", export_tab="1", export_json="1"}
IMPORTED, MB_MSGS, CONSOLE, TRACK_GUIDS = {}, {}, "", {}
DEFER_CB = nil
reaper.GetExtState = function(_, k) return extstate[k] or "" end
reaper.SetExtState = function(_, k, v) extstate[k] = v end
reaper.file_exists = function(p) return FILE_EXISTS(p) end
reaper.GetResourcePath = function() return "/tmp/rr" end
reaper.GetOS = function() return "Linux_x86_64" end
reaper.GetSelectedMediaItem = function() return 1 end
reaper.GetActiveTake = function() return 1 end
reaper.GetMediaItemTake_Source = function() return 1 end
reaper.GetMediaSourceFileName = function(_, buf) return AUDIO, buf end
reaper.MB = function(m,t,k) MB_MSGS[#MB_MSGS+1] = tostring(m) end
reaper.ShowConsoleMsg = function(m) CONSOLE = CONSOLE .. tostring(m) end
reaper.Undo_BeginBlock = function() end
reaper.Undo_EndBlock = function() end
reaper.InsertMedia = function(p, sel)
  IMPORTED[#IMPORTED+1] = p
  TRACK_GUIDS[#TRACK_GUIDS+1] = "guid-" .. tostring(#TRACK_GUIDS+1)
  return #IMPORTED
end
reaper.GetNumTracks = function() return #TRACK_GUIDS end
reaper.GetTrack = function(i, idx) return { idx = idx } end
reaper.GetTrackGUID = function(track) return TRACK_GUIDS[track.idx + 1] end
reaper.GetSetMediaTrackInfo_String = function() end
reaper.EnumProjects = function() return PROJECT end
reaper.defer = function(cb) DEFER_CB = cb end
reaper.EnumerateFiles = function(dir, idx) return ENUM(dir, idx) end
"""


class _Stream:
    """Minimal io.popen handle: reads real lines, reports a real exit status."""

    def __init__(self, cmd):
        self.proc = subprocess.Popen(
            ["/bin/sh", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        self.lines = self.proc.stdout.read().splitlines()
        self.i = 0

    def read(self, _fmt):
        if self.i < len(self.lines):
            self.i += 1
            return self.lines[self.i - 1]
        return None

    def close(self):
        return (True, "exit", self.proc.wait())


def run(
    exit_code,
    stale=False,
    name="song",
    audio_name=None,
    only=None,
):
    install = tempfile.mkdtemp(prefix="inst-")
    outdir = tempfile.mkdtemp(prefix="out-")
    audio = os.path.join(install, audio_name or (name + ".wav"))
    Path(audio).write_bytes(b"RIFF")

    if stale:
        for suffix in MIDI_SUFFIXES:
            target = Path(outdir) / f"{name}_{suffix}.mid"
            target.write_bytes(b"MThd stale")
            old = time.time() - 3600
            os.utime(target, (old, old))

    if only == "":
        writes = []  # ran cleanly but transcribed nothing
    elif only:
        writes = [f"{name}_{suffix}.mid" for suffix in (only,)]
    else:
        writes = [f"{name}_{suffix}.mid" for suffix in MIDI_SUFFIXES] if exit_code == 0 else []
    stub = (
        "import json, os, sys\n"
        "args = sys.argv\n"
        "out = (args[args.index('--output-dir') + 1]"
        f" if '--output-dir' in args else {str(Path(install) / 'output')!r})\n"
        "os.makedirs(out, exist_ok=True)\n"
        f"writes = {writes!r}\n"
        f"NAME = {name!r}\n"
        "for f in writes:\n"
        "    open(os.path.join(out, f), 'wb').write(b'MThd fresh')\n"
        "    print('[tab-agent-midi] ' + f[len(NAME) + 1:-4])\n"
        "print('[tab-agent] stub ran')\n"
        f"sys.exit({exit_code})\n"
    )
    Path(install, "main.py").write_text(stub)

    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    lua.execute(
        STUB.replace("INSTALL", json.dumps(install))
        .replace("OUTDIR", json.dumps(outdir))
        .replace("AUDIO", json.dumps(audio))
        .replace("PROJECT", "1")
    )
    lua.globals().FILE_EXISTS = lambda p: os.path.exists(p)

    def _enum(directory, index):
        try:
            entries = sorted(os.listdir(directory))
        except OSError:
            return None
        index = int(index)
        return entries[index] if index < len(entries) else None

    lua.globals().ENUM = _enum
    lua.globals().io.popen = lambda cmd, mode: _Stream(cmd)
    lua.execute((ROOT / "reaper" / "TabAgent.lua").read_text())

    for _ in range(300):
        callback = lua.globals().DEFER_CB
        if callback is None:
            break
        lua.globals().DEFER_CB = None
        callback()

    imported = [os.path.basename(p) for p in lua.globals().IMPORTED.values()]
    dialogs = [m.splitlines()[0][:70] for m in lua.globals().MB_MSGS.values()]
    console = str(lua.globals().CONSOLE)
    shutil.rmtree(install, ignore_errors=True)
    shutil.rmtree(outdir, ignore_errors=True)
    return imported, dialogs, console


def main():
    if lupa is None:
        print("lupa not installed - skipping REAPER safety check")
        return 0
    results = []

    def t1():
        imported, dialogs, _ = run(1, stale=True)
        return (
            imported == [],
            f"imported={len(imported)} dialogs={dialogs}",
        )

    def t2():
        imported, _dialogs, _ = run(0, stale=False)
        return imported == [f"song_{s}.mid" for s in MIDI_SUFFIXES], f"imported={imported}"

    def t3():
        # exit 0, produces nothing, while four stale MIDI files sit in the
        # output directory: nothing may be imported.
        imported, dialogs, console = run(0, stale=True, only="")
        return (
            imported == [] and any("No MIDI from this run" in d for d in dialogs),
            f"imported={imported} dialogs={dialogs} console={[line for line in console.splitlines() if 'ignored' in line]}",
        )

    def t4():
        imported, _dialogs, _ = run(0, stale=True, only="bass")
        return imported == ["song_bass.mid"], f"imported={imported}"

    def t5():
        # A command substitution in the filename must arrive literally and must
        # not be executed by /bin/sh. The marker is looked for in a fresh temp
        # directory owned by the test.
        audio = "evil$(touch PWNED_MARKER).wav"
        marker = Path(tempfile.mkdtemp(prefix="inject-")) / "PWNED_MARKER"
        cwd = os.getcwd()
        os.chdir(marker.parent)
        try:
            run(0, stale=False, audio_name=audio)
        finally:
            os.chdir(cwd)
        return not marker.exists(), f"injection file present={marker.exists()}"

    def t6():
        audio = "it's a song.wav"
        _imported, _dialogs, console = run(0, stale=False, audio_name=audio)
        # The basename in REAPER comes from the source filename; a quote must not
        # break the shell command (the pipeline still runs and reports).
        return "stub ran" in console, f"console={console.splitlines()[:2]}"

    def t7():
        # The P0-2 scenario in its exact original shape: exit 0, four stale MIDI
        # files on disk, and a run that legitimately produced only the bass.
        imported, _dialogs, _ = run(0, stale=True, only="bass")
        return imported == ["song_bass.mid"], f"imported={imported}"

    checks = [
        ("failed run imports nothing (stale files present)", t1),
        ("successful run imports all four tracks", t2),
        ("success with no new output imports nothing + warns", t3),
        ("success imports only the genuinely new file", t4),
        ("shell metacharacters in filename are literal", t5),
        ("single quote in filename is preserved", t6),
        ("stale files ignored when only one track is regenerated", t7),
    ]
    for label, fn in checks:
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            print(f"        {detail}")
        results.append(ok)
    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
