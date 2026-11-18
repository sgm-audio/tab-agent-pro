"""Check shipped REAPER Lua against real POSIX shell child processes.

The shim exercises command construction, deferred callbacks and the import
protocol, including fresh run directories and same-run MIDI announcements.
The child is an argv/export fixture, not Basic Pitch or a native REAPER host.
Requires Lupa (the same dev dependency as the Lua/CLI contract tests).
"""

import os
import subprocess  # nosec B404
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIDI_SUFFIXES = ("lead_guitar", "rhythm_L", "rhythm_R", "bass")

STUB = r"""
reaper = {}
local extstate = {install_path = INSTALL, output_dir = OUTDIR,
                  export_midi="1", export_tab="1", export_json="1"}
IMPORTED, MB_MSGS, CONSOLE, TRACK_GUIDS = {}, {}, "", {}
DEFER_CB = nil
reaper.GetExtState = function(_, k) return extstate[k] or "" end
reaper.file_exists = function(p) return FILE_EXISTS(p) end
reaper.GetResourcePath = function() return "/no-reaper-install-here" end
reaper.GetOS = function() return "Linux_x86_64" end
reaper.GetSelectedMediaItem = function() return 1 end
reaper.GetActiveTake = function() return 1 end
reaper.GetMediaItemTake_Source = function() return 1 end
reaper.GetMediaSourceFileName = function() return AUDIO end
reaper.MB = function(message) MB_MSGS[#MB_MSGS+1] = message end
reaper.ShowConsoleMsg = function(message) CONSOLE = CONSOLE .. message end
reaper.Undo_BeginBlock = function() end
reaper.Undo_EndBlock = function() end
reaper.InsertMedia = function(path, mode)
  assert(mode == 1, "must import to a new track")
  IMPORTED[#IMPORTED+1] = path
  TRACK_GUIDS[#TRACK_GUIDS+1] = "guid-" .. tostring(#TRACK_GUIDS+1)
  return #IMPORTED
end
reaper.GetNumTracks = function() return #TRACK_GUIDS end
reaper.GetTrack = function(_, index) return { index = index } end
reaper.GetTrackGUID = function(track) return TRACK_GUIDS[track.index + 1] end
reaper.GetSetMediaTrackInfo_String = function() end
reaper.EnumProjects = function() return 1 end
reaper.defer = function(cb) DEFER_CB = cb end
"""

CHILD = """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--fresh-output-dir', action='store_true')
parser.add_argument('--reaper-report-midi', action='store_true')
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('audio')
args = parser.parse_args()
assert args.audio == fixture['audio'], (args.audio, fixture['audio'])
assert args.fresh_output_dir and args.reaper_report_midi
out = args.output_dir
assert out.parent == Path(fixture['root']) and out.name.startswith('tab-agent-'), out
out.mkdir()  # Mirror the CLI's refusal to reuse an existing run directory.
for suffix in fixture['written']:
    (out / f'{Path(args.audio).stem}_{suffix}.mid').write_bytes(b'MThd fresh')
for suffix in fixture['reported']:
    print(f'[tab-agent-midi] {suffix}')
print('[tab-agent] stub ran')
raise SystemExit(fixture['exit_code'])
"""


class _Stream:
    """Replay the real subprocess's output and exit status to Lua."""

    def __init__(self, command: str):
        # The Lua action constructs this command from fixture paths only.
        result = subprocess.run(  # nosec B603
            ["/bin/sh", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=15,
            env={**os.environ, "TAB_AGENT_LITERAL": "wrong-if-expanded"},
        )
        self.lines = iter(result.stdout.splitlines())
        self.result = result.returncode

    def read(self, _fmt):
        return next(self.lines, None)

    def close(self):
        if self.result < 0:
            return None, "signal", -self.result
        return (True if self.result == 0 else None), "exit", self.result


@dataclass(frozen=True)
class _Case:
    label: str
    written: tuple[str, ...]
    expected: tuple[str, ...]
    stale: bool = False
    exit_code: int = 0
    audio_name: str = "song.wav"
    reported: tuple[str, ...] | None = None
    dialog: str | None = None


def _check_case(case: _Case) -> None:
    from lupa import LuaRuntime

    with tempfile.TemporaryDirectory(prefix="reaper-safety-") as scratch:
        install = Path(scratch) / "Tab Agent install"
        out = Path(scratch) / "mix exports"
        install.mkdir()
        out.mkdir()
        audio = install / case.audio_name
        audio.write_bytes(b"RIFF")
        old_files = []
        if case.stale:
            previous = out / "tab-agent-previous"
            previous.mkdir()
            for root in (out, previous):
                for suffix in MIDI_SUFFIXES:
                    target = root / f"{audio.stem}_{suffix}.mid"
                    target.write_bytes(b"MThd stale")
                    old_files.append(target)

        fixture = {
            "audio": str(audio),
            "root": str(out),
            "written": case.written,
            "reported": case.written if case.reported is None else case.reported,
            "exit_code": case.exit_code,
        }
        (install / "main.py").write_text(f"fixture = {fixture!r}\n{CHILD}", encoding="utf-8")

        lua = LuaRuntime(unpack_returned_tuples=True)
        host = lua.globals()
        host.INSTALL, host.OUTDIR, host.AUDIO = str(install), str(out), str(audio)
        host.FILE_EXISTS = lambda path: Path(path).is_file()
        lua.execute(STUB)
        host.io.popen = lambda command, _mode: _Stream(command)
        lua.execute((ROOT / "reaper" / "TabAgent.lua").read_text(encoding="utf-8"))

        for _ in range(100):
            callback = host.DEFER_CB
            if callback is None:
                break
            host.DEFER_CB = None
            callback()
        else:
            raise RuntimeError("deferred callback did not finish")

        console = str(host.CONSOLE)
        assert "[tab-agent] stub ran" in console, console
        sessions = [p for p in out.glob("tab-agent-*") if p.name != "tab-agent-previous"]
        assert len(sessions) == 1 and sessions[0].is_dir(), sessions
        expected = [str(sessions[0] / f"{audio.stem}_{suffix}.mid") for suffix in case.expected]
        imported = list(host.IMPORTED.values())
        assert imported == expected, f"imported={imported}, expected={expected}"
        assert all(Path(p).read_bytes() == b"MThd fresh" for p in imported)
        assert all(path.read_bytes() == b"MThd stale" for path in old_files)
        dialogs = list(host.MB_MSGS.values())
        if case.dialog:
            assert len(dialogs) == 1 and case.dialog in dialogs[0], dialogs
        else:
            assert dialogs == [], dialogs


def main() -> int:
    if os.name != "posix" or not Path("/bin/sh").is_file():
        print("The REAPER subprocess check requires a POSIX shell.", file=sys.stderr)
        return 1

    bass = ("bass",)
    cases = [
        _Case(
            "failed child with partial output imports nothing",
            bass,
            (),
            True,
            1,
            dialog="Transcription failed",
        ),
        _Case("success imports all reported tracks", MIDI_SUFFIXES, MIDI_SUFFIXES),
        _Case(
            "success without new MIDI warns and ignores stale files",
            (),
            (),
            True,
            dialog="No MIDI from this run",
        ),
        _Case("only the new bass is imported despite stale MIDI", bass, bass, True),
        _Case(
            "shell variable syntax reaches the child literally",
            bass,
            bass,
            audio_name="take $TAB_AGENT_LITERAL 雪.wav",
        ),
        _Case(
            "apostrophe in source path reaches the child literally",
            bass,
            bass,
            audio_name="it's a song.wav",
        ),
        _Case(
            "an unreported file in the fresh directory is not imported",
            ("lead_guitar", "bass"),
            bass,
            reported=bass,
        ),
    ]
    failures = 0
    for case in cases:
        try:
            _check_case(case)
        except Exception as exc:
            print(f"  FAIL  {case.label}: {type(exc).__name__}: {exc}")
            failures += 1
        else:
            print(f"  PASS  {case.label}")
    print(f"\n{len(cases) - failures}/{len(cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
