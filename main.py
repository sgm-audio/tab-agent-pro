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
        except (OSError, json.JSONDecodeError):
            # Corrupt or unreadable preferences file; use defaults
            pass
    else:
        pass

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


def _string_labels(num_strings: int) -> list[str]:
    """String labels ordered high→low (string index 0 is the lowest string)."""
    if num_strings == 6:
        return ["E", "B", "G", "D", "A", "E"]  # high E, B, G, D, A, low E
    if num_strings == 5:
        return ["G", "D", "A", "E", "B"]  # high G, D, A, E, low B
    if num_strings == 4:
        return ["G", "D", "A", "E"]
    return [chr(65 + i) for i in range(num_strings - 1, -1, -1)]


def export_tab_to_txt(
    tab_data,
    output_path,
    instrument="Guitar",
    num_strings: int | None = None,
) -> None:
    """Export tablature to human-readable text format.

    Groups simultaneous notes (within 50ms) into the same column
    so chords align vertically across strings.

    Notes that share a string within one 50ms group are all rendered, joined
    with ``/`` (previously the later note silently overwrote the earlier one,
    so the ASCII tab disagreed with the JSON export).

    Args:
        tab_data: positions from ``TabAgent.generate_tab``
        output_path: destination ``.tab`` file
        instrument: label written in the file header
        num_strings: instrument string count; keeps the displayed grid the same
            size as the instrument even when the highest strings are unused
    """
    if not tab_data:
        return

    max_used = max(pos["string"] for pos in tab_data) + 1
    num_strings = max(max_used, num_strings or 0)

    # Group positions by start_time (within 50ms tolerance)
    groups: list[tuple[float, list]] = []
    for pos in tab_data:
        t = pos.get("start_time", 0.0)
        # Find or create group
        if groups and abs(groups[-1][0] - t) < 0.05:
            groups[-1][1].append(pos)
        else:
            groups.append((t, [pos]))

    # Build column grid: one column per time group
    cols: list[list[str]] = []
    for _grp_time, grp_positions in groups:
        cells: list[list[str]] = [[] for _ in range(num_strings)]  # one slot per string
        for pos in grp_positions:
            cells[pos["string"]].append(_format_position(pos))
        # Fill untouched strings with "-"; join multiple notes on one string
        cols.append(["/".join(cell) if cell else "-" for cell in cells])

    # Determine column widths (widest entry in each column)
    col_widths = [max(len(col[s]) for s in range(num_strings)) for col in cols]
    separator = COLUMN_SEPARATOR

    # Build each string's line from the columns (left-aligned so every column
    # occupies the same [entry][padding][separator] shape and note positions
    # are predictable when read by eye or by a script).
    lines = [""] * num_strings
    for col_idx, col in enumerate(cols):
        w = col_widths[col_idx]
        for s in range(num_strings):
            lines[s] += col[s].ljust(w) + separator

    labels = _string_labels(num_strings)

    with open(output_path, "w") as f:
        f.write(f"=== {instrument} Tablature ===\n\n")
        # Display high strings first (reverse order)
        for i in range(num_strings - 1, -1, -1):
            f.write(f"{labels[num_strings - 1 - i]}|{lines[i]}\n")
        f.write("\nLegend: s=slide, h=hammer-on, p=pull-off\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def export_tab_to_json(
    tab_data,
    output_path,
    instrument="Guitar",
    num_strings: int | None = None,
) -> None:
    """Export tablature to JSON format for programmatic use."""
    data = {
        "instrument": instrument,
        "timestamp": datetime.now().isoformat(),
        "num_strings": num_strings,
        "tablature": tab_data,
    }

    with open(output_path, "w") as f:
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


def resolve_thresholds(args, config: dict) -> tuple[float, float]:
    """
    Resolve onset/frame thresholds.

    Precedence: explicit CLI flag > active profile value > built-in default.
    """
    onset = args.onset
    if onset is None:
        onset = float(config.get("onset_threshold", 0.5))
    frame = args.frame
    if frame is None:
        frame = float(config.get("frame_threshold", 0.3))
    return onset, frame


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
        choices=["Guitar", "Bass"],
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
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--profile", type=str, default=None, help="Preset profile name (see init_memory.py --list)"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None, help="Output directory (default: ./output)"
    )
    parser.add_argument("--version", action="version", version="Tab Agent 1.0.0")
    return parser


def parse_args(argv: list[str] | None = None):
    """Parse CLI arguments (options documented in ``init_memory.py --list``)."""
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:

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

    os.makedirs(output_dir, exist_ok=True)

    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Load user memory and configuration
    memory_file, config = load_user_memory()

    # Apply a preset profile when requested (persists to user memory)
    if args.profile:
        config = apply_profile(args.profile, config)
        memory_file = config.get("memory_file", memory_file)
        print(f"🎛️  Profile: {args.profile}")

    # Thresholds: explicit CLI flag > active profile > built-in default
    onset_threshold, frame_threshold = resolve_thresholds(args, config)

    # Suno artifact detection and preprocessing

    processed_audio, is_suno, suno_metrics = process_suno_audio(
        audio_path,
        output_path=os.path.join(output_dir, f"{song_name}_processed.wav"),
    )

    # Initialize agents

    splitter = SplitterAgent(output_dir=os.path.join(output_dir, "stems"))

    # Separate stems (use processed audio if Suno)
    stems = splitter.separate_stems(processed_audio)

    # Process guitar stems
    guitar_stems = splitter.process_guitars(stems["guitar"])

    # Process bass stem
    bass_clean = splitter.process_bass(stems["bass"])

    # Initialize transcription agent

    ear = EarAgent(device=args.device)
    suno_postprocessor = SunoNotePostprocessor()

    # Transcribe lead guitar
    lead_notes_raw = ear.transcribe_stem(
        guitar_stems["lead"],
        target="Lead Guitar",
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )
    lead_notes = ear.humanize_and_clean(lead_notes_raw, is_bass=False)
    # Apply Suno post-processing if needed
    lead_notes = suno_postprocessor.process(lead_notes, is_suno, suno_metrics)
    if not args.no_midi:
        lead_midi_path = os.path.join(output_dir, f"{song_name}_lead_guitar.mid")
        ear.export_midi(lead_notes, lead_midi_path)

    # Transcribe rhythm guitar (left channel)
    rhythm_l_notes_raw = ear.transcribe_stem(
        guitar_stems["left"],
        target="Rhythm Guitar L",
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )
    rhythm_l_notes = ear.humanize_and_clean(rhythm_l_notes_raw, is_bass=False)
    rhythm_l_notes = suno_postprocessor.process(rhythm_l_notes, is_suno, suno_metrics)
    if not args.no_midi:
        rhythm_l_midi_path = os.path.join(output_dir, f"{song_name}_rhythm_L.mid")
        ear.export_midi(rhythm_l_notes, rhythm_l_midi_path)

    # Transcribe rhythm guitar (right channel)
    rhythm_r_notes_raw = ear.transcribe_stem(
        guitar_stems["right"],
        target="Rhythm Guitar R",
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )
    rhythm_r_notes = ear.humanize_and_clean(rhythm_r_notes_raw, is_bass=False)
    rhythm_r_notes = suno_postprocessor.process(rhythm_r_notes, is_suno, suno_metrics)
    if not args.no_midi:
        rhythm_r_midi_path = os.path.join(output_dir, f"{song_name}_rhythm_R.mid")
        ear.export_midi(rhythm_r_notes, rhythm_r_midi_path)

    # Transcribe bass
    bass_notes_raw = ear.transcribe_stem(
        bass_clean,
        target="Bass",
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )
    bass_notes = ear.humanize_and_clean(bass_notes_raw, is_bass=True)
    bass_notes = suno_postprocessor.process(bass_notes, is_suno, suno_metrics)
    if not args.no_midi:
        bass_midi_path = os.path.join(output_dir, f"{song_name}_bass.mid")
        ear.export_midi(bass_notes, bass_midi_path)

    # Generate tablature

    guitar_tuning = config["guitar_tuning"]
    bass_tuning = config["bass_tuning"]
    num_frets = config["num_frets"]

    # Guitar tablature
    guitar_agent = TabAgent(tuning=guitar_tuning, num_frets=num_frets)

    guitar_tracks = {
        "lead_guitar": ("Lead Guitar", lead_notes),
        "rhythm_L": ("Rhythm Guitar L", rhythm_l_notes),
        "rhythm_R": ("Rhythm Guitar R", rhythm_r_notes),
    }
    for suffix, (label, notes) in guitar_tracks.items():
        tab = guitar_agent.generate_tab(notes)
        if not args.no_tab and tab:
            export_tab_to_txt(
                tab,
                os.path.join(output_dir, f"{song_name}_{suffix}.tab"),
                label,
                num_strings=len(guitar_tuning),
            )
        if not args.no_json and tab:
            export_tab_to_json(
                tab,
                os.path.join(output_dir, f"{song_name}_{suffix}.json"),
                label,
                num_strings=len(guitar_tuning),
            )
        if notes and not tab:
            print(f"⚠️  No playable positions for {label} — skipping its tablature.")

    # Bass tablature
    bass_agent = TabAgent(tuning=bass_tuning, num_frets=num_frets)

    bass_tab = bass_agent.generate_tab(bass_notes)
    if not args.no_tab and bass_tab:
        export_tab_to_txt(
            bass_tab,
            os.path.join(output_dir, f"{song_name}_bass.tab"),
            f"{len(bass_tuning)}-String Bass",
            num_strings=len(bass_tuning),
        )
    if not args.no_json and bass_tab:
        export_tab_to_json(
            bass_tab,
            os.path.join(output_dir, f"{song_name}_bass.json"),
            f"{len(bass_tuning)}-String Bass",
            num_strings=len(bass_tuning),
        )
    if bass_notes and not bass_tab:
        print("⚠️  No playable positions for Bass — skipping its tablature.")

    # Log session to memory
    if os.path.exists(memory_file):
        try:
            with open(memory_file) as f:
                preferences = json.load(f)

            session_log = {
                "timestamp": datetime.now().isoformat(),
                "song": os.path.basename(audio_path),
                "status": "completed",
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

    # Summary
    print(f"\n✅ Done: {song_name}")
    print(f"   Instrument : {args.instrument}")
    print(f"   Output dir : {output_dir}")
    print(f"   Formats    : {', '.join(_enabled_formats(args)) or 'none'}")
    print(f"   Thresholds : onset={onset_threshold}, frame={frame_threshold}")
    if config.get("active_profile"):
        print(f"   Profile    : {config['active_profile']}")


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
    main()
