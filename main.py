import argparse
import json
import os
import sys
from datetime import datetime

from agents import EarAgent, SplitterAgent, TabAgent
from suno_postprocessor import SunoNotePostprocessor, process_suno_audio

# Fix Windows console encoding for emojis (skipped under pytest; detach would break capture)
if sys.platform == "win32" and "pytest" not in sys.modules:
    import codecs

    if hasattr(sys.stdout, "detach"):
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Technique detection constants
TECHNIQUE_SLIDE = "slide"
TECHNIQUE_HAMMER = "hammer"
TECHNIQUE_PULL = "pull"
TECHNIQUE_PICK = "pick"


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no python-dotenv dependency). Existing env vars win."""
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError as e:
        # A broken .env must not abort a run, but it must not be invisible
        # either: the user believes the file was applied.
        print(f"⚠️  Could not read {path}: {e}", file=sys.stderr)


_load_dotenv()


def load_user_memory():
    """Load user memory and extract preferences."""
    # Path resolution is shared with init_memory so profiles written by
    # `init_memory.py` / `--profile` are found here (Docker: /app/user_memory,
    # local: ./user_memory).
    import init_memory

    memory_dir = init_memory.get_memory_dir()

    os.makedirs(memory_dir, exist_ok=True)

    memory_file = os.path.join(memory_dir, "user_preferences.json")

    # Default configurations
    config = {
        "bass_tuning": [23, 28, 33, 38, 43],  # B0-E1-A1-D2-G2 (5-string)
        "guitar_tuning": [40, 45, 50, 55, 59, 64],  # E2-A2-D3-G3-B3-E4 (standard)
        "bass_num_strings": 5,
        "guitar_num_strings": 6,
        "num_frets": 24,
        "prefer_low_strings": True,
    }

    # Load from file if it exists
    if os.path.exists(memory_file):
        try:
            with open(memory_file) as f:
                preferences = json.load(f)
                if "config" in preferences:
                    config.update(preferences["config"])
        except (OSError, json.JSONDecodeError) as e:
            # Corrupt preferences are a real, user-fixable problem: say so and
            # continue with defaults rather than silently ignoring saved tuning.
            print(
                f"⚠️  Ignoring unreadable user memory {memory_file}: {e}",
                file=sys.stderr,
            )

    return memory_file, config


# Space between adjacent tablature columns so multi-character cells never merge
# (without it, frets 12 and 15 in neighbouring columns render as "1215").
COLUMN_SEPARATOR = " "


def _format_position(pos) -> str:
    """Format a single tab position (fret + optional technique suffix)."""
    t = pos.get("technique", TECHNIQUE_PICK)
    f = pos["fret"]
    if t == TECHNIQUE_SLIDE:
        return f"{f}s"
    if t == TECHNIQUE_HAMMER:
        return f"{f}h"
    if t == TECHNIQUE_PULL:
        return f"{f}p"
    return str(f)


_PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
RAW_MIX_WARNING = "Stem separation failed; these results come from unseparated audio."


def _string_labels(num_strings: int, tuning: list[int] | None = None) -> list[str]:
    """String labels ordered high to low; tuning is ordered low to high."""
    if tuning is not None:
        return [_PITCH_CLASSES[pitch % 12] for pitch in reversed(tuning)]
    if num_strings == 6:
        return ["E", "B", "G", "D", "A", "E"]
    if num_strings == 5:
        return ["G", "D", "A", "E", "B"]
    if num_strings == 4:
        return ["G", "D", "A", "E"]
    return [chr(65 + i) for i in range(num_strings - 1, -1, -1)]


def render_tab_text(
    tab_data,
    instrument="Guitar",
    num_strings: int | None = None,
    tuning: list[int] | None = None,
    *,
    skipped_notes: list[dict] | None = None,
    separation_mode: str | None = None,
) -> str | None:
    """Render aligned fret and onset columns; exact durations are in JSON/MIDI."""
    if not tab_data or not isinstance(tab_data, list):
        return None

    max_used = max(pos["string"] for pos in tab_data) + 1
    num_strings = max(max_used, num_strings or 0, len(tuning) if tuning is not None else 0)
    if tuning is not None and len(tuning) != num_strings:
        tuning = None
    labels = _string_labels(num_strings, tuning)
    show_time = any("start_time" in pos for pos in tab_data)

    # Group identical onsets only. Near-simultaneous notes (e.g. arpeggios)
    # need separate columns, even if they are less than 50 ms apart.
    groups: dict[float, list] = {}
    for pos in tab_data:
        groups.setdefault(pos.get("start_time", 0.0), []).append(pos)

    rows: list[list[str]] = [[] for _ in labels]
    times: list[str] = []
    for onset, positions in sorted(groups.items()):
        cells: list[list[str]] = [[] for _ in labels]
        for pos in positions:
            cells[num_strings - 1 - pos["string"]].append(_format_position(pos))
        # Keep the renderer compatible with old tab data passed directly to it.
        # Newly generated tabs cannot have two notes on one string at an onset.
        column = ["/".join(cell) or "-" for cell in cells]
        stamp = f"{onset:.3f}" if show_time else ""
        width = max(len(stamp), *(len(cell) for cell in column))
        times.append(stamp.ljust(width))
        for row, cell in zip(rows, column, strict=True):
            row.append(cell.ljust(width))

    out = [f"=== {instrument} Tablature ==="]
    if tuning is not None:
        names = (f"{_PITCH_CLASSES[p % 12]}{p // 12 - 1}" for p in tuning)
        out.append(f"Tuning (low to high): {' '.join(names)}")
    if separation_mode == "raw_fallback":
        out.append(f"Warning: {RAW_MIX_WARNING}")
    if skipped_notes:
        out.append(f"Warning: {len(skipped_notes)} detected note(s) omitted from the tab.")
    out.append("")
    label_width = max(4 if show_time else 0, *(len(label) for label in labels))
    if show_time:
        out.append("t(s)|" + COLUMN_SEPARATOR.join(times).rstrip())
    for label, row in zip(labels, rows, strict=True):
        out.append(f"{label:>{label_width}}|{COLUMN_SEPARATOR.join(row).rstrip()}")
    out.append("")
    if any(
        pos.get("technique") in (TECHNIQUE_SLIDE, TECHNIQUE_HAMMER, TECHNIQUE_PULL)
        for pos in tab_data
    ):
        out.append("Technique legend (supplied marks): s=slide, h=hammer-on, p=pull-off")
    else:
        out.append("Playing techniques not inferred.")
    out.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(out) + "\n"


def export_tab_to_txt(
    tab_data,
    output_path,
    instrument="Guitar",
    num_strings: int | None = None,
    tuning: list[int] | None = None,
    *,
    skipped_notes: list[dict] | None = None,
    separation_mode: str | None = None,
) -> None:
    """Export tablature to human-readable text format."""
    text = render_tab_text(
        tab_data,
        instrument,
        num_strings,
        tuning=tuning,
        skipped_notes=skipped_notes,
        separation_mode=separation_mode,
    )
    if text is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)


def export_tab_to_json(
    tab_data,
    output_path,
    instrument="Guitar",
    num_strings: int | None = None,
    *,
    tuning: list[int] | None = None,
    skipped_notes: list[dict] | None = None,
    separation_mode: str | None = None,
) -> None:
    """Export timed fretboard data and its provenance for programmatic use."""
    data = {
        "instrument": instrument,
        "timestamp": datetime.now().isoformat(),
        "num_strings": len(tuning) if tuning is not None else num_strings,
        "tuning": tuning,
        "technique_basis": "not_inferred",
        "separation_mode": separation_mode,
        "skipped_notes": skipped_notes or [],
        "tablature": tab_data,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def apply_profile(profile_key: str, config: dict, memory_file: str | None = None) -> dict:
    """
    Apply a preset profile (see ``init_memory.py --list``) to a config dict.

    The profile is persisted to user memory so later runs (and the ReaPack
    integration, which reads the same file) reuse it.

    Args:
        profile_key: key from ``init_memory.PROFILES``
        config: configuration dict previously returned by ``load_user_memory``
        memory_file: path to persist to; defaults to the standard memory file

    Returns:
        Updated configuration dict.
    """
    import init_memory

    if profile_key not in init_memory.PROFILES:
        available = ", ".join(init_memory.PROFILES)
        raise SystemExit(f"Unknown profile: {profile_key}\nAvailable profiles: {available}")

    saved_path = init_memory.save_profile(profile_key)
    _, config = load_user_memory()
    config["active_profile"] = profile_key
    config["memory_file"] = saved_path
    return config


# Detection thresholds are probabilities: values outside [0, 1] (or NaN/inf)
# cannot be satisfied by the models, so they silently produce empty or junk
# transcriptions instead of failing loudly.
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0


def validate_thresholds(onset: float, frame: float) -> tuple[float, float]:
    """
    Validate onset/frame thresholds against the model's probability range.

    Returns the pair unchanged when valid; raises ``ValueError`` otherwise.
    Called for both CLI flags and values loaded from user memory, so a corrupt
    or hand-edited ``user_preferences.json`` cannot quietly destroy output.
    """
    import math

    for name, value in (("onset", onset), ("frame", frame)):
        if math.isnan(value) or math.isinf(value):
            msg = f"{name} threshold must be a finite number, got {value!r}"
            raise ValueError(msg)
        # Bounds are inclusive: 0.0 (max recall) and 1.0 (max precision) are
        # legitimate extremes, so only out-of-range or non-finite values fail.
        if not (THRESHOLD_MIN <= value <= THRESHOLD_MAX):
            msg = (
                f"{name} threshold must be between {THRESHOLD_MIN} and "
                f"{THRESHOLD_MAX}, got {value!r}"
            )
            raise ValueError(msg)
    return onset, frame


def resolve_thresholds(args, config: dict) -> tuple[float, float]:
    """
    Resolve onset/frame thresholds.

    Precedence: explicit CLI flag > active profile value > built-in default.
    The resolved pair is range-validated so bad values fail loudly (exit) at
    the start of a run rather than producing an empty transcription.
    """
    onset = args.onset
    if onset is None:
        onset = float(config.get("onset_threshold", 0.5))
    frame = args.frame
    if frame is None:
        frame = float(config.get("frame_threshold", 0.3))
    try:
        return validate_thresholds(onset, frame)
    except ValueError as e:
        msg = f"❌ {e}"
        raise SystemExit(msg) from None


def _normalize_instrument(value: str) -> str:
    """Normalize instrument CLI input case-insensitively ('bass' -> 'Bass')."""
    normalized = value.strip().capitalize()
    if normalized not in ("Guitar", "Bass", "All"):
        raise argparse.ArgumentTypeError(
            f"invalid choice: {value!r} (choose from 'Guitar', 'Bass', 'All')"
        )
    return normalized


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="tab-agent",
        description="Tab Agent — AI-powered guitar/bass tablature transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py song.wav
  python main.py song.wav --instrument bass --onset 0.4 --frame 0.3
  python main.py song.wav --profile rock_drop_d --no-tab
        """,
    )
    parser.add_argument("audio", nargs="?", help="Path to audio file (WAV/MP3/FLAC)")
    parser.add_argument(
        "--instrument",
        "-i",
        type=_normalize_instrument,
        choices=["Guitar", "Bass", "All"],
        default="Guitar",
        help="Instrument type (default: Guitar)",
    )
    # Thresholds default to None so a preset profile can supply them when the
    # user does not pass an explicit flag.
    parser.add_argument(
        "--onset",
        type=float,
        default=None,
        help="Onset detection threshold (default: 0.5, or the active profile's value)",
    )
    parser.add_argument(
        "--frame",
        type=float,
        default=None,
        help="Frame activation threshold (default: 0.3, or the active profile's value)",
    )
    parser.add_argument("--no-midi", action="store_true", help="Skip MIDI export")
    parser.add_argument("--no-tab", action="store_true", help="Skip ASCII tab export")
    parser.add_argument("--no-json", action="store_true", help="Skip JSON export")
    # Internal protocol: REAPER imports only MIDI written by this invocation.
    parser.add_argument("--reaper-report-midi", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "rocm", "vulkan", "mps", "directml"],
        default=os.getenv("DEVICE", "auto"),
        help="Compute device (default: DEVICE env var or auto). CUDA = NVIDIA, "
        "rocm = AMD ROCm, vulkan = AMD Vulkan, mps = Apple Silicon, "
        "directml = Windows DirectML. Falls back to CPU if the requested "
        "backend is unavailable.",
    )
    parser.add_argument(
        "--profile", type=str, default=None, help="Preset profile name (see init_memory.py --list)"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None, help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--fresh-output-dir",
        action="store_true",
        help="Require an explicit, new --output-dir so a run cannot reuse old exports",
    )
    parser.add_argument("--version", action="version", version="Tab Agent 1.1.0")
    return parser


def parse_args(argv: list[str] | None = None):
    """Parse CLI arguments (options documented in ``init_memory.py --list``)."""
    return build_parser().parse_args(argv)


def _progress(msg: str) -> None:
    """Emit a progress line for stdout consumers (e.g. the REAPER console
    streaming wrapper). Always flushed so a streaming reader never blocks."""
    print(f"[tab-agent] {msg}", flush=True)


def _export_cli_tab(
    agent: TabAgent,
    notes: list,
    stem: str,
    label: str,
    args,
    separation_mode: str | None,
) -> None:
    if args.no_tab and args.no_json:
        return
    tab = agent.generate_tab(notes)
    skipped = agent.last_skipped_events
    if skipped:
        reasons = ", ".join(sorted({event["reason"] for event in skipped}))
        print(
            f"⚠️  {label}: {len(skipped)} detected note(s) omitted from the tab ({reasons}).",
            file=sys.stderr,
        )
    if not tab:
        return
    if not args.no_tab:
        export_tab_to_txt(
            tab,
            f"{stem}.tab",
            label,
            tuning=agent.tuning,
            skipped_notes=skipped,
            separation_mode=separation_mode,
        )
    if not args.no_json:
        export_tab_to_json(
            tab,
            f"{stem}.json",
            label,
            tuning=agent.tuning,
            skipped_notes=skipped,
            separation_mode=separation_mode,
        )


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.audio:
        parser.print_help()
        sys.exit(1)

    audio_path = os.path.abspath(args.audio)

    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    if args.output_dir:
        output_dir = args.output_dir
    elif os.path.exists("/app/output"):
        output_dir = "/app/output"
    else:
        output_dir = "./output"

    if args.fresh_output_dir:
        if not args.output_dir:
            parser.error("--fresh-output-dir requires --output-dir")
        os.makedirs(os.path.dirname(os.path.abspath(output_dir)), exist_ok=True)
        # Atomically claim the directory. Never process or report old files if
        # the chosen path already exists, even when another run just created it.
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            parser.error(f"Output directory already exists: {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)

    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Load user memory and configuration
    memory_file, config = load_user_memory()

    # Apply a preset profile when requested (persists to user memory)
    if args.profile:
        config = apply_profile(args.profile, config)
        memory_file = config.get("memory_file", memory_file)
        print(f"🎛️  Profile: {args.profile}")

    # Suno artifact detection and preprocessing
    if "suno_aggressive_mode" in config:
        aggressiveness = 0.8 if config["suno_aggressive_mode"] else 0.2
    else:
        aggressiveness = 0.5

    _progress("Analyzing audio quality (Suno artifact detection)...")
    processed_audio, is_suno, suno_metrics = process_suno_audio(
        audio_path,
        output_path=os.path.join(output_dir, f"{song_name}_processed.wav"),
        aggressiveness=aggressiveness,
    )

    # Thresholds: explicit CLI flag > active profile > Suno-adjusted default > built-in default
    onset_threshold, frame_threshold = resolve_thresholds(args, config)
    if is_suno:
        if args.onset is None and "onset_threshold" not in config:
            onset_threshold = 0.6
        if args.frame is None and "frame_threshold" not in config:
            frame_threshold = 0.4

    # Determine which instrument pipelines to run (must match --help exactly):
    # - `--instrument Guitar` (the default): guitar only
    # - `--instrument Bass`: bass only
    # - `--instrument All`, or no flag at all when the user explicitly wants
    #   both, run both pipelines.
    # Previously an *absent* flag forced run_bass True, so a plain
    # `main.py song.wav` produced bass MIDI/tab despite documenting
    # "default: Guitar". `instrument_explicit` is kept so an explicit
    # "--instrument Guitar" and the implicit default behave identically.
    run_guitar = args.instrument == "Guitar" or args.instrument == "All"
    run_bass = args.instrument == "Bass" or args.instrument == "All"

    # Initialize agents
    splitter = SplitterAgent(
        output_dir=os.path.join(output_dir, "stems"),
        device=args.device,
    )

    # Separate stems (use processed audio if Suno)
    _progress("Separating audio stems (Demucs)...")
    stems = splitter.separate_stems(processed_audio)
    separation_mode = splitter.separation_mode
    if separation_mode == "raw_fallback":
        print(f"⚠️  {RAW_MIX_WARNING}", file=sys.stderr)
    note_count = 0

    # Initialize transcription agent
    ear = EarAgent(device=args.device)
    suno_postprocessor = SunoNotePostprocessor()

    def write_midi(notes: list, suffix: str) -> None:
        if args.no_midi or not notes:
            return
        path = os.path.join(output_dir, f"{song_name}_{suffix}.mid")
        ear.export_midi(notes, path)
        if args.reaper_report_midi:
            print(f"[tab-agent-midi] {suffix}", flush=True)

    guitar_tuning = config.get("guitar_tuning", [40, 45, 50, 55, 59, 64])
    bass_tuning = config.get("bass_tuning", [23, 28, 33, 38, 43])
    num_frets = int(config.get("num_frets", 24))
    guitar_prefer_low = (
        bool(config.get("prefer_low_strings", False)) if config.get("active_profile") else False
    )
    bass_prefer_low = bool(config.get("prefer_low_strings", True))

    if run_guitar:
        _progress("Transcribing guitar stems to MIDI (Basic Pitch)...")
        # Process guitar stems
        guitar_stems = splitter.process_guitars(stems["guitar"])
        guitar_min_pitch = min(guitar_tuning, default=40)
        guitar_max_pitch = max(guitar_tuning, default=64) + num_frets

        # Transcribe lead guitar
        lead_notes_raw = ear.transcribe_stem(
            guitar_stems["lead"],
            target="Lead Guitar",
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            min_pitch=guitar_min_pitch,
            max_pitch=guitar_max_pitch,
        )
        lead_notes = ear.humanize_and_clean(lead_notes_raw, is_bass=False)
        lead_notes = suno_postprocessor.process(lead_notes, is_suno, suno_metrics)
        write_midi(lead_notes, "lead_guitar")

        # Transcribe rhythm guitar (left channel)
        rhythm_l_notes_raw = ear.transcribe_stem(
            guitar_stems["left"],
            target="Rhythm Guitar L",
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            min_pitch=guitar_min_pitch,
            max_pitch=guitar_max_pitch,
        )
        rhythm_l_notes = ear.humanize_and_clean(rhythm_l_notes_raw, is_bass=False)
        rhythm_l_notes = suno_postprocessor.process(rhythm_l_notes, is_suno, suno_metrics)
        write_midi(rhythm_l_notes, "rhythm_L")

        # Transcribe rhythm guitar (right channel)
        rhythm_r_notes_raw = ear.transcribe_stem(
            guitar_stems["right"],
            target="Rhythm Guitar R",
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            min_pitch=guitar_min_pitch,
            max_pitch=guitar_max_pitch,
        )
        rhythm_r_notes = ear.humanize_and_clean(rhythm_r_notes_raw, is_bass=False)
        rhythm_r_notes = suno_postprocessor.process(rhythm_r_notes, is_suno, suno_metrics)
        write_midi(rhythm_r_notes, "rhythm_R")

        # Guitar tablature
        guitar_agent = TabAgent(
            tuning=guitar_tuning,
            num_frets=num_frets,
            prefer_low_strings=guitar_prefer_low,
        )

        guitar_tracks = {
            "lead_guitar": ("Lead Guitar", lead_notes),
            "rhythm_L": ("Rhythm Guitar L", rhythm_l_notes),
            "rhythm_R": ("Rhythm Guitar R", rhythm_r_notes),
        }
        for suffix, (label, notes) in guitar_tracks.items():
            note_count += len(notes)
            _export_cli_tab(
                guitar_agent,
                notes,
                os.path.join(output_dir, f"{song_name}_{suffix}"),
                label,
                args,
                separation_mode,
            )

    if run_bass:
        _progress("Transcribing bass stem to MIDI (Basic Pitch)...")
        # Process bass stem
        bass_clean = splitter.process_bass(stems["bass"])
        bass_min_pitch = min(bass_tuning, default=23)
        bass_max_pitch = max(bass_tuning, default=43) + num_frets

        # Transcribe bass
        bass_notes_raw = ear.transcribe_stem(
            bass_clean,
            target="Bass",
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            min_pitch=bass_min_pitch,
            max_pitch=bass_max_pitch,
        )
        bass_notes = ear.humanize_and_clean(bass_notes_raw, is_bass=True, max_pitch=bass_max_pitch)
        bass_notes = suno_postprocessor.process(bass_notes, is_suno, suno_metrics)
        write_midi(bass_notes, "bass")

        # Bass tablature
        bass_agent = TabAgent(
            tuning=bass_tuning,
            num_frets=num_frets,
            prefer_low_strings=bass_prefer_low,
        )

        note_count += len(bass_notes)
        _export_cli_tab(
            bass_agent,
            bass_notes,
            os.path.join(output_dir, f"{song_name}_bass"),
            f"{len(bass_tuning)}-String Bass",
            args,
            separation_mode,
        )

    # Log session to memory
    if os.path.exists(memory_file):
        try:
            with open(memory_file) as f:
                preferences = json.load(f)

            session_log = {
                "timestamp": datetime.now().isoformat(),
                "song": os.path.basename(audio_path),
                "status": "completed" if note_count else "no_notes",
                "separation_mode": separation_mode,
                "onset_threshold": onset_threshold,
                "frame_threshold": frame_threshold,
                "is_suno": bool(is_suno),
            }
            if config.get("active_profile"):
                session_log["profile"] = config["active_profile"]
            preferences.setdefault("sessions", []).append(session_log)

            # Atomic write: a crash mid-serialisation must not corrupt the
            # preferences file (it also gets re-read by the next run).
            tmp_path = f"{memory_file}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(preferences, f, indent=2)
            os.replace(tmp_path, memory_file)
        except (OSError, TypeError, ValueError) as e:
            # Session logging is best-effort; don't fail the run
            print(f"⚠️  Could not update user memory: {e}")

    if not note_count:
        print(f"⚠️  No notes were transcribed for {song_name}.", file=sys.stderr)
        return 2

    # Summary
    print(f"\n✅ Done: {song_name}")
    print(f"   Instrument : {args.instrument}")
    print(f"   Output dir : {output_dir}")
    print(f"   Formats    : {', '.join(_enabled_formats(args)) or 'none'}")
    print(f"   Thresholds : onset={onset_threshold}, frame={frame_threshold}")
    if config.get("active_profile"):
        print(f"   Profile    : {config['active_profile']}")

    return 0


def _enabled_formats(args) -> list[str]:
    """Human-readable list of the export formats enabled by CLI flags."""
    return [
        name
        for name, disabled in (
            ("MIDI", args.no_midi),
            ("Tab", args.no_tab),
            ("JSON", args.no_json),
        )
        if not disabled
    ]


if __name__ == "__main__":
    sys.exit(main())
