"""
Tab Agent - Hugging Face Gradio Interface (MVP)
AI-powered guitar/bass tablature transcription using Basic Pitch

This is the web UI for the Tab Agent transcription system.
Optimized for Zero GPU deployment with Basic Pitch model.
"""

import os
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr

# Health checks & structured logging
from monitoring import health

# Zero GPU support for faster processing
try:
    import spaces

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  Running without Zero GPU support")

# Import Tab Agent modules
from agents import EarAgent, SplitterAgent, TabAgent
from main import export_tab_to_json, export_tab_to_txt
from suno_postprocessor import SunoNotePostprocessor, process_suno_audio

# Configuration
TEMP_DIR = tempfile.gettempdir()
OUTPUT_DIR = Path(TEMP_DIR) / "tab_agent_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Default tunings
GUITAR_TUNING = [40, 45, 50, 55, 59, 64]  # E2-A2-D3-G3-B3-E4
BASS_TUNING = [23, 28, 33, 38, 43]  # B0-E1-A1-D2-G2


# Apply Zero GPU decorator if available
if GPU_AVAILABLE:

    @spaces.GPU
    def process_audio(
        audio_file,
        instrument_type="Guitar",
        include_midi=True,
        include_tab=True,
        include_json=True,
        progress=gr.Progress(),
    ):
        """
        Process audio file and generate tablature.

        Args:
            audio_file: Path to uploaded audio file
            instrument_type: "Guitar" or "Bass"
            include_midi: Export MIDI files
            include_tab: Export ASCII tab files
            include_json: Export JSON files
            progress: Gradio progress callback

        Returns:
            Tuple of (status_message, output_files_zip)

        """
        return _process_audio_impl(
            audio_file,
            instrument_type,
            include_midi,
            include_tab,
            include_json,
            progress,
        )

else:

    def process_audio(
        audio_file,
        instrument_type="Guitar",
        include_midi=True,
        include_tab=True,
        include_json=True,
        progress=gr.Progress(),
    ):
        """Process audio file and generate tablature (CPU-only)."""
        return _process_audio_impl(
            audio_file,
            instrument_type,
            include_midi,
            include_tab,
            include_json,
            progress,
        )


def _process_audio_impl(
    audio_file,
    instrument_type,
    include_midi,
    include_tab,
    include_json,
    progress,
):
    """
    Internal implementation of audio processing.
    """
    if audio_file is None:
        return "❌ Please upload an audio file", None

    try:
        # Create unique output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = OUTPUT_DIR / f"session_{timestamp}"
        session_dir.mkdir(exist_ok=True)

        # Track processing time for ETA / README accuracy
        start_time = time.time()

        # Get file info
        audio_path = Path(audio_file)
        song_name = audio_path.stem

        progress(0.1, desc="🎵 Initializing agents...")

        # Stage 0: Suno artifact detection and preprocessing
        progress(0.15, desc="🔍 Analyzing audio quality...")
        processed_audio, is_suno, suno_metrics = process_suno_audio(
            str(audio_path),
            output_path=str(session_dir / f"{song_name}_processed.wav"),
        )

        # Adjust thresholds for AI-generated audio
        if is_suno:
            onset_threshold = 0.6
            frame_threshold = 0.4
        else:
            onset_threshold = 0.5
            frame_threshold = 0.3

        # Initialize agents (auto-detects GPU via Zero GPU)
        splitter = SplitterAgent(output_dir=str(session_dir / "stems"))
        ear = EarAgent(device="auto")  # Auto-detect: GPU if available, else CPU
        suno_postprocessor = SunoNotePostprocessor()

        # Stage 1-3: Stem separation and processing
        progress(0.2, desc="🎵 Separating audio stems (Demucs)...")
        stems = splitter.separate_stems(processed_audio)

        progress(0.3, desc="🎸 Processing guitar stems...")
        if instrument_type == "Guitar":
            guitar_stems = splitter.process_guitars(stems["guitar"])
            processed_stems = {
                "lead": guitar_stems["lead"],
                "rhythm_l": guitar_stems["left"],
                "rhythm_r": guitar_stems["right"],
            }
        else:  # Bass
            bass_clean = splitter.process_bass(stems["bass"])
            processed_stems = {"bass": bass_clean}

        # Stage 4: Transcription
        progress(0.5, desc="🎸 Transcribing to MIDI...")

        results = {}
        for stem_name, stem_path in processed_stems.items():
            progress(
                0.5 + (0.3 / len(processed_stems)),
                desc=f"🎸 Transcribing {stem_name}...",
            )

            # Transcribe
            notes_raw = ear.transcribe_stem(
                stem_path,
                target=instrument_type,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
            )
            notes_clean = ear.humanize_and_clean(notes_raw, is_bass=(instrument_type == "Bass"))
            # Apply Suno post-processing if needed
            notes_clean = suno_postprocessor.process(notes_clean, is_suno, suno_metrics)

            # Export MIDI
            if include_midi:
                midi_path = session_dir / f"{song_name}_{stem_name}.mid"
                ear.export_midi(notes_clean, str(midi_path))

            results[stem_name] = notes_clean

        # Stage 5: Tablature generation
        progress(0.8, desc="📝 Generating tablature...")

        if instrument_type == "Guitar":
            tab_agent = TabAgent(tuning=GUITAR_TUNING, num_frets=24)
        else:
            tab_agent = TabAgent(tuning=BASS_TUNING, num_frets=24)

        for stem_name, notes in results.items():
            tab_data = tab_agent.generate_tab(notes)

            # Export tab files
            if include_tab:
                tab_path = session_dir / f"{song_name}_{stem_name}.tab"
                export_tab_to_txt(
                    tab_data,
                    str(tab_path),
                    instrument=f"{instrument_type} - {stem_name}",
                )

            if include_json:
                json_path = session_dir / f"{song_name}_{stem_name}.json"
                export_tab_to_json(
                    tab_data,
                    str(json_path),
                    instrument=f"{instrument_type} - {stem_name}",
                )

        # Create ZIP archive
        progress(0.9, desc="📦 Creating download package...")
        zip_path = session_dir / f"{song_name}_tablature.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in session_dir.rglob("*"):
                if file.is_file() and file != zip_path:
                    arcname = file.relative_to(session_dir)
                    zipf.write(file, arcname)

        progress(1.0, desc="✅ Complete!")

        # Generate status message
        elapsed = time.time() - start_time
        file_count = len(list(session_dir.glob("*.*"))) - 1  # Exclude zip
        status_msg = f"""
✅ **Transcription Complete!**

- **Song**: {song_name}
- **Instrument**: {instrument_type}
- **Files Generated**: {file_count}
- **Processing Time**: {elapsed:.1f}s
- **Formats**: {
            ", ".join(
                [
                    "MIDI" if include_midi else "",
                    "Tab" if include_tab else "",
                    "JSON" if include_json else "",
                ]
            ).strip(", ")
        }

📥 **Download the ZIP file below to get all outputs!**
        """

        return status_msg, str(zip_path)

    except Exception as e:
        import traceback

        error_msg = (
            f"❌ **Error during processing:**\n\n```\n{e!s}\n\n{traceback.format_exc()}\n```"
        )
        return error_msg, None


# Create Gradio interface
def create_ui():
    """Create Gradio UI interface."""
    with gr.Blocks(title="Tab Agent - AI Tablature Transcription", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🎸 Tab Agent - AI Tablature Transcription (MVP)

AI-powered transcription for guitar and bass using **Basic Pitch** (Spotify's proven model).

## Features
- 🎵 **Multi-stage pipeline**: Demucs stem separation + spatial processing
- 🤖 **Basic Pitch AI**: Production-ready transcription model (Spotify)
- 🎸 **Multi-track support**: Lead guitar, rhythm guitars (L/R), bass
- 📝 **Multiple formats**: MIDI, ASCII tablature, JSON
- 🎯 **Optimal fingering**: Dynamic programming for playable tabs
- ✨ **Technique detection**: Slides, hammer-ons, pull-offs
- ⚡ **Zero GPU**: Faster processing with Hugging Face Zero GPU

---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Audio")

                audio_input = gr.Audio(
                    label="Audio File",
                    type="filepath",
                    sources=["upload"],
                )

                instrument_type = gr.Radio(
                    label="Instrument Type",
                    choices=["Guitar", "Bass"],
                    value="Guitar",
                )

                gr.Markdown("### Export Options")

                with gr.Group():
                    export_midi = gr.Checkbox(label="MIDI Files", value=True)
                    export_tab = gr.Checkbox(label="ASCII Tablature", value=True)
                    export_json = gr.Checkbox(label="JSON Data", value=True)

                transcribe_btn = gr.Button(
                    "🎸 Transcribe to Tablature",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Results")

                status_output = gr.Markdown(
                    value="Upload an audio file and click 'Transcribe' to begin.",
                    label="Status",
                )

                download_output = gr.File(label="Download Results (ZIP)", interactive=False)

        # Examples
        gr.Markdown("### Example Audio Files")
        gr.Examples(
            examples=[
                ["examples/guitar_solo.wav", "Guitar"],
                ["examples/bass_groove.wav", "Bass"],
            ],
            inputs=[audio_input, instrument_type],
        )

        # Information
        with gr.Accordion("ℹ️ How It Works", open=False):
            gr.Markdown("""
### Processing Pipeline

1. **Stem Separation (Demucs)**: Isolates guitar/bass from full mix
2. **Spatial Processing**: Separates lead and rhythm guitars using mid-side technique
3. **AI Transcription (Basic Pitch)**: Converts audio to MIDI notes with proven accuracy
4. **Tablature Generation**: Dynamic programming finds optimal fingering
5. **Technique Detection**: Identifies slides, hammer-ons, pull-offs

### Output Formats

- **MIDI**: Import into DAWs (Reaper, Ableton, Logic, etc.)
- **ASCII Tab**: Human-readable tablature for printing
- **JSON**: Programmatic access for custom applications

### Tips for Best Results

- Use high-quality audio (WAV/FLAC preferred)
- Isolate guitar/bass tracks if possible
- Shorter clips (< 60 seconds) process faster
- Clean recordings work better than live/noisy audio
            """)

        with gr.Accordion("🔗 Links & Resources", open=False):
            gr.Markdown("""
- **GitHub**: [Tab-Agent Repository](https://github.com/scottmills306/tab-agent-pro)
- **ReaPack**: [Install for Reaper](https://github.com/scottmills306/tab-agent-pro#reaper-integration)
- **Documentation**: [Full Guide](https://github.com/scottmills306/tab-agent-pro/blob/main/README.md)
- **Basic Pitch**: [Spotify Research](https://github.com/spotify/basic-pitch)

**License**: MIT | **Python**: 3.10+ | **Model**: Basic Pitch | **Acceleration**: Zero GPU
            """)

        # Connect event handlers
        transcribe_btn.click(
            fn=process_audio,
            inputs=[audio_input, instrument_type, export_midi, export_tab, export_json],
            outputs=[status_output, download_output],
        )

    return demo


def create_app():
    """Create FastAPI app with health endpoints and Gradio UI mounted."""
    from fastapi import FastAPI

    demo = create_ui()
    demo.queue()

    parent_app = FastAPI()

    @parent_app.get("/health")
    async def health_endpoint():
        return health.as_dict()

    @parent_app.get("/health/metrics")
    async def metrics_endpoint():
        from monitoring import default_metrics

        return default_metrics.summary()

    parent_app = gr.mount_gradio_app(parent_app, demo, path="/")
    return parent_app


# Main entry point
if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(create_app(), host=os.getenv("HOST", "127.0.0.1"), port=7860)
