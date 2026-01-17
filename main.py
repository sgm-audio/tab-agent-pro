import os
import sys
import json
from datetime import datetime
from agents import SplitterAgent, EarAgent, TabAgent
from suno_postprocessor import process_suno_audio, SunoNotePostprocessor

# Fix Windows console encoding for emojis
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

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
            with open(memory_file, 'r') as f:
                preferences = json.load(f)
                if "config" in preferences:
                    config.update(preferences["config"])
            print(f"🧠 Loaded preferences from: {memory_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load preferences: {e}")
            print("   Using default configuration")
    else:
        print("📋 Using default configuration (run init_memory.py to customize)")

    return memory_file, config

def export_tab_to_txt(tab_data, output_path, instrument="Guitar"):
    """Export tablature to human-readable text format."""
    if not tab_data:
        print(f"⚠️  No tab data to export for {instrument}")
        return

    num_strings = max(pos['string'] for pos in tab_data) + 1

    # Create ASCII tab grid
    lines = [[] for _ in range(num_strings)]

    for pos in tab_data:
        string_idx = pos['string']
        fret = pos['fret']
        technique = pos.get('technique', 'pick')

        # Format fret number with technique marker
        if technique == "slide":
            fret_str = f"{fret}s"
        elif technique == "hammer":
            fret_str = f"{fret}h"
        elif technique == "pull":
            fret_str = f"{fret}p"
        else:
            fret_str = str(fret)

        # Add to appropriate string
        for i in range(num_strings):
            if i == string_idx:
                lines[i].append(fret_str.ljust(3))
            else:
                lines[i].append("---")

    # Write to file
    with open(output_path, 'w') as f:
        f.write(f"=== {instrument} Tablature ===\n\n")

        # String labels (reverse order for display)
        string_labels = ["E", "A", "D", "G", "B", "E"] if num_strings == 6 else ["B", "E", "A", "D", "G"]

        for i in range(num_strings - 1, -1, -1):
            f.write(f"{string_labels[i]}|{''.join(lines[i])}|\n")

        f.write(f"\nLegend: s=slide, h=hammer-on, p=pull-off\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"📄 Saved tab: {output_path}")

def export_tab_to_json(tab_data, output_path, instrument="Guitar"):
    """Export tablature to JSON format for programmatic use."""
    data = {
        "instrument": instrument,
        "timestamp": datetime.now().isoformat(),
        "tablature": tab_data
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"📄 Saved JSON: {output_path}")

def main():
    print("=" * 60)
    print("🎸 TAB AGENT - Audio to Tablature Pipeline")
    print("=" * 60)

    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py <audio_file>")
        sys.exit(1)

    song_file = sys.argv[1]

    # Determine paths (Docker vs local)
    if os.path.exists("/app/input"):
        input_dir = "/app/input"
        output_dir = "/app/output"
    else:
        input_dir = "./input"
        output_dir = "./output"
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    audio_path = os.path.join(input_dir, song_file)

    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found: {audio_path}")
        sys.exit(1)

    song_name = os.path.splitext(song_file)[0]
    print(f"\n🎵 Processing: {song_file}")
    print(f"📂 Input: {input_dir}")
    print(f"📂 Output: {output_dir}\n")

    # Load user memory and configuration
    memory_file, config = load_user_memory()

    # Suno artifact detection and preprocessing
    print("\n" + "=" * 60)
    print("STAGE 0: AUDIO QUALITY ANALYSIS")
    print("=" * 60)

    processed_audio, is_suno, suno_metrics = process_suno_audio(
        audio_path,
        output_path=os.path.join(output_dir, f"{song_name}_processed.wav")
    )

    # Adjust transcription parameters based on audio quality
    if is_suno:
        onset_threshold = 0.6  # Higher threshold for noisy AI audio
        frame_threshold = 0.4
    else:
        onset_threshold = 0.5  # Standard for clean audio
        frame_threshold = 0.3

    # Initialize agents
    print("\n" + "=" * 60)
    print("STAGE 1-3: STEM SEPARATION & PROCESSING")
    print("=" * 60)

    splitter = SplitterAgent(output_dir=os.path.join(output_dir, "stems"))

    # Separate stems (use processed audio if Suno)
    stems = splitter.separate_stems(processed_audio)

    # Process guitar stems
    guitar_stems = splitter.process_guitars(stems['guitar'])

    # Process bass stem
    bass_clean = splitter.process_bass(stems['bass'])

    # Initialize transcription agent
    print("\n" + "=" * 60)
    print("STAGE 4: AUDIO TRANSCRIPTION")
    print("=" * 60)

    ear = EarAgent()
    suno_postprocessor = SunoNotePostprocessor()

    # Transcribe lead guitar
    print("\n🎸 Transcribing Lead Guitar...")
    lead_notes_raw = ear.transcribe_stem(
        guitar_stems['lead'],
        target="Lead Guitar",
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold
    )
    lead_notes = ear.humanize_and_clean(lead_notes_raw, is_bass=False)
    # Apply Suno post-processing if needed
    lead_notes = suno_postprocessor.process(lead_notes, is_suno, suno_metrics)
    lead_midi_path = os.path.join(output_dir, f"{song_name}_lead_guitar.mid")
    ear.export_midi(lead_notes, lead_midi_path)

    # Transcribe rhythm guitar (left channel)
    print("\n🎸 Transcribing Rhythm Guitar (L)...")
    rhythm_l_notes_raw = ear.transcribe_stem(
        guitar_stems['left'],
        target="Rhythm Guitar L",
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold
    )
    rhythm_l_notes = ear.humanize_and_clean(rhythm_l_notes_raw, is_bass=False)
    rhythm_l_notes = suno_postprocessor.process(rhythm_l_notes, is_suno, suno_metrics)
    rhythm_l_midi_path = os.path.join(output_dir, f"{song_name}_rhythm_L.mid")
    ear.export_midi(rhythm_l_notes, rhythm_l_midi_path)

    # Transcribe rhythm guitar (right channel)
    print("\n🎸 Transcribing Rhythm Guitar (R)...")
    rhythm_r_notes_raw = ear.transcribe_stem(
        guitar_stems['right'],
        target="Rhythm Guitar R",
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold
    )
    rhythm_r_notes = ear.humanize_and_clean(rhythm_r_notes_raw, is_bass=False)
    rhythm_r_notes = suno_postprocessor.process(rhythm_r_notes, is_suno, suno_metrics)
    rhythm_r_midi_path = os.path.join(output_dir, f"{song_name}_rhythm_R.mid")
    ear.export_midi(rhythm_r_notes, rhythm_r_midi_path)

    # Transcribe bass
    print("\n🎸 Transcribing Bass...")
    bass_notes_raw = ear.transcribe_stem(
        bass_clean,
        target="Bass",
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold
    )
    bass_notes = ear.humanize_and_clean(bass_notes_raw, is_bass=True)
    bass_notes = suno_postprocessor.process(bass_notes, is_suno, suno_metrics)
    bass_midi_path = os.path.join(output_dir, f"{song_name}_bass.mid")
    ear.export_midi(bass_notes, bass_midi_path)

    # Generate tablature
    print("\n" + "=" * 60)
    print("STAGE 5: TABLATURE GENERATION")
    print("=" * 60)

    # Guitar tablature
    guitar_agent = TabAgent(
        tuning=config['guitar_tuning'],
        num_frets=config['num_frets']
    )

    print("\n🎸 Generating Lead Guitar Tab...")
    lead_tab = guitar_agent.generate_tab(lead_notes)
    export_tab_to_txt(lead_tab, os.path.join(output_dir, f"{song_name}_lead_guitar.tab"), "Lead Guitar")
    export_tab_to_json(lead_tab, os.path.join(output_dir, f"{song_name}_lead_guitar.json"), "Lead Guitar")

    print("\n🎸 Generating Rhythm Guitar (L) Tab...")
    rhythm_l_tab = guitar_agent.generate_tab(rhythm_l_notes)
    export_tab_to_txt(rhythm_l_tab, os.path.join(output_dir, f"{song_name}_rhythm_L.tab"), "Rhythm Guitar L")
    export_tab_to_json(rhythm_l_tab, os.path.join(output_dir, f"{song_name}_rhythm_L.json"), "Rhythm Guitar L")

    print("\n🎸 Generating Rhythm Guitar (R) Tab...")
    rhythm_r_tab = guitar_agent.generate_tab(rhythm_r_notes)
    export_tab_to_txt(rhythm_r_tab, os.path.join(output_dir, f"{song_name}_rhythm_R.tab"), "Rhythm Guitar R")
    export_tab_to_json(rhythm_r_tab, os.path.join(output_dir, f"{song_name}_rhythm_R.json"), "Rhythm Guitar R")

    # Bass tablature
    bass_agent = TabAgent(
        tuning=config['bass_tuning'],
        num_frets=config['num_frets']
    )

    print("\n🎸 Generating Bass Tab...")
    bass_tab = bass_agent.generate_tab(bass_notes)
    export_tab_to_txt(bass_tab, os.path.join(output_dir, f"{song_name}_bass.tab"), "5-String Bass")
    export_tab_to_json(bass_tab, os.path.join(output_dir, f"{song_name}_bass.json"), "5-String Bass")

    # Log session to memory
    if os.path.exists(memory_file):
        try:
            with open(memory_file, 'r') as f:
                preferences = json.load(f)

            session_log = {
                "timestamp": datetime.now().isoformat(),
                "song": song_file,
                "status": "completed"
            }
            preferences.setdefault("sessions", []).append(session_log)

            with open(memory_file, 'w') as f:
                json.dump(preferences, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not log session: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n📁 Output Files:")
    print(f"   MIDI Files:")
    print(f"      - {song_name}_lead_guitar.mid")
    print(f"      - {song_name}_rhythm_L.mid")
    print(f"      - {song_name}_rhythm_R.mid")
    print(f"      - {song_name}_bass.mid")
    print(f"\n   Tablature Files:")
    print(f"      - {song_name}_lead_guitar.tab")
    print(f"      - {song_name}_rhythm_L.tab")
    print(f"      - {song_name}_rhythm_R.tab")
    print(f"      - {song_name}_bass.tab")
    print(f"\n   JSON Files:")
    print(f"      - {song_name}_lead_guitar.json")
    print(f"      - {song_name}_rhythm_L.json")
    print(f"      - {song_name}_rhythm_R.json")
    print(f"      - {song_name}_bass.json")
    print(f"\n📂 Location: {output_dir}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
