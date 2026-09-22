import argparse
import json
import os
import sys
from datetime import datetime

from agents import EarAgent, SplitterAgent, TabAgent
from suno_postprocessor import SunoNotePostprocessor, process_suno_audio

# Fix Windows console encoding for emojis
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Technique detection constants
TECHNIQUE_SLIDE = "slide"
TECHNIQUE_HAMMER = "hammer"
TECHNIQUE_PULL = "pull"
TECHNIQUE_PICK = "pick"


def load_user_memory():
    """Load user memory and extract preferences."""
    # In Docker: /app/user_memory
    # On Windows/Local: ./user_memory
    if os.path.exists("/app") and not sys.platform.startswith("win"):
        memory_dir = "/app/user_memory"
    else:
        memory_dir = "./user_memory"

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


def export_tab_to_txt(tab_data, output_path, instrument="Guitar") -> None:
    """Export tablature to human-readable text format.

    Groups simultaneous notes (within 50ms) into the same column
    so chords align vertically across strings.
    """
    if not tab_data:
        return

    num_strings = max(pos["string"] for pos in tab_data) + 1

    # Format a single position into a display string
    def _fmt(pos):
        t = pos.get("technique", TECHNIQUE_PICK)
        f = pos["fret"]
        if t == TECHNIQUE_SLIDE:
            return f"{f}s"
        if t == TECHNIQUE_HAMMER:
            return f"{f}h"
        if t == TECHNIQUE_PULL:
            return f"{f}p"
        return str(f)

    # Group positions by start_time (within 50ms tolerance)
    groups = []  # list of (time, [pos, ...])
    for pos in tab_data:
        t = pos.get("start_time", 0.0)
        # Find or create group
        if groups and abs(groups[-1][0] - t) < 0.05:
            groups[-1][1].append(pos)
        else:
            groups.append((t, [pos]))

    # Build column grid: one column per time group
    cols = []
    for _grp_time, grp_positions in groups:
        col = [""] * num_strings  # one slot per string
        for pos in grp_positions:
            col[pos["string"]] = _fmt(pos)
        # Fill empty strings with "-"
        for i in range(num_strings):
            if col[i] == "":
                col[i] = "-"
        cols.append(col)

    # Determine column widths (widest entry in each column)
    col_widths = [max(len(col[s]) for s in range(num_strings)) for col in cols]

    # Build each string's line from the columns
    lines = [""] * num_strings
    for col_idx, col in enumerate(cols):
        w = col_widths[col_idx]
        for s in range(num_strings):
            lines[s] += col[s].center(w)

    # String labels, high→low (reversed from string indices 0=lowest)
    if num_strings == 6:
        labels = ["E", "B", "G", "D", "A", "E"]  # high E, B, G, D, A, low E
    elif num_strings == 5:
        labels = ["G", "D", "A", "E", "B"]  # high G, D, A, E, low B
    elif num_strings == 4:
        labels = ["G", "D", "A", "E"]
    else:
        labels = [chr(65 + i) for i in range(num_strings - 1, -1, -1)]

    with open(output_path, "w") as f:
        f.write(f"=== {instrument} Tablature ===\n\n")
        # Display high strings first (reverse order)
        for i in range(num_strings - 1, -1, -1):
            f.write(f"{labels[num_strings - 1 - i]}|{lines[i]}|\n")
        f.write("\nLegend: s=slide, h=hammer-on, p=pull-off\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def export_tab_to_json(tab_data, output_path, instrument="Guitar") -> None:
    """Export tablature to JSON format for programmatic use."""
    data = {
        "instrument": instrument,
        "timestamp": datetime.now().isoformat(),
        "tablature": tab_data,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tab Agent — AI-powered guitar/bass tablature transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py song.wav
  python main.py song.wav --instrument bass --onset 0.4 --frame 0.3
  python main.py song.wav --no-midi --no-tab
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
    parser.add_argument(
        "--onset", type=float, default=0.5, help="Onset detection threshold (default: 0.5)"
    )
    parser.add_argument(
        "--frame", type=float, default=0.3, help="Frame activation threshold (default: 0.3)"
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
    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if not args.audio:
        parse_args().print_help()
        sys.exit(1)

    audio_path = os.path.abspath(args.audio)

    if not os.path.exists(audio_path):
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

    # Suno artifact detection and preprocessing

    processed_audio, is_suno, suno_metrics = process_suno_audio(
        audio_path,
        output_path=os.path.join(output_dir, f"{song_name}_processed.wav"),
    )

    # Use CLI-provided thresholds (or defaults)
    onset_threshold = args.onset
    frame_threshold = args.frame

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

    # Guitar tablature
    guitar_agent = TabAgent(tuning=config["guitar_tuning"], num_frets=config["num_frets"])

    lead_tab = guitar_agent.generate_tab(lead_notes)
    if not args.no_tab:
        export_tab_to_txt(
            lead_tab,
            os.path.join(output_dir, f"{song_name}_lead_guitar.tab"),
            "Lead Guitar",
        )
    if not args.no_json:
        export_tab_to_json(
            lead_tab,
            os.path.join(output_dir, f"{song_name}_lead_guitar.json"),
            "Lead Guitar",
        )

    rhythm_l_tab = guitar_agent.generate_tab(rhythm_l_notes)
    if not args.no_tab:
        export_tab_to_txt(
            rhythm_l_tab,
            os.path.join(output_dir, f"{song_name}_rhythm_L.tab"),
            "Rhythm Guitar L",
        )
    if not args.no_json:
        export_tab_to_json(
            rhythm_l_tab,
            os.path.join(output_dir, f"{song_name}_rhythm_L.json"),
            "Rhythm Guitar L",
        )

    rhythm_r_tab = guitar_agent.generate_tab(rhythm_r_notes)
    if not args.no_tab:
        export_tab_to_txt(
            rhythm_r_tab,
            os.path.join(output_dir, f"{song_name}_rhythm_R.tab"),
            "Rhythm Guitar R",
        )
    if not args.no_json:
        export_tab_to_json(
            rhythm_r_tab,
            os.path.join(output_dir, f"{song_name}_rhythm_R.json"),
            "Rhythm Guitar R",
        )

    # Bass tablature
    bass_agent = TabAgent(tuning=config["bass_tuning"], num_frets=config["num_frets"])

    bass_tab = bass_agent.generate_tab(bass_notes)
    if not args.no_tab:
        export_tab_to_txt(
            bass_tab, os.path.join(output_dir, f"{song_name}_bass.tab"), "5-String Bass"
        )
    if not args.no_json:
        export_tab_to_json(
            bass_tab,
            os.path.join(output_dir, f"{song_name}_bass.json"),
            "5-String Bass",
        )

    # Log session to memory
    if os.path.exists(memory_file):
        try:
            with open(memory_file) as f:
                preferences = json.load(f)

            session_log = {
                "timestamp": datetime.now().isoformat(),
                "song": os.path.basename(audio_path),
                "status": "completed",
            }
            preferences.setdefault("sessions", []).append(session_log)

            with open(memory_file, "w") as f:
                json.dump(preferences, f, indent=2)
        except OSError:
            # Session logging is best-effort; don't fail the run
            pass

    # Summary
    if not args.no_midi:
        pass
    if not args.no_tab:
        pass
    if not args.no_json:
        pass


if __name__ == "__main__":
    main()
