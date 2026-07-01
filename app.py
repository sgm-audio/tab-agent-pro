"""
Tab Agent - Hugging Face Gradio Interface (MVP)
AI-powered guitar/bass tablature transcription using Basic Pitch

This is the web UI for the Tab Agent transcription system.
Optimized for Zero GPU deployment with Basic Pitch model.
"""

import atexit
import os
import shutil
import signal
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

MAX_FILE_SIZE_MB = 50
MAX_DURATION_SEC = 300  # 5 minutes
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}


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


def _validate_audio(audio_path: Path) -> str | None:
    """Validate audio file. Returns error message or None."""
    if not audio_path.exists():
        return "File not found"
    ext = audio_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"Unsupported format: {ext}. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return f"File too large: {size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)"
    try:
        import librosa

        duration = librosa.get_duration(path=str(audio_path))
        if duration > MAX_DURATION_SEC:
            return f"Audio too long: {duration:.0f}s (max {MAX_DURATION_SEC}s)"
    except Exception as e:
        return f"Cannot read audio: {e}"
    return None


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
    cleanup_stale_sessions()

    if audio_file is None:
        return "❌ Please upload an audio file", None

    # Validate
    audio_path = Path(audio_file)
    error = _validate_audio(audio_path)
    if error:
        return f"❌ {error}", None

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
        from monitoring import get_logger

        get_logger("app").error("transcription_failed", exc=e)
        error_msg = f"❌ **Transcription failed:** {e}"
        return error_msg, None


# Create Gradio interface
def create_ui():
    """Create Gradio UI interface."""
    css = """
    .tab-agent-header h1 { font-size: 2.2rem; margin-bottom: 0; }
    .tab-agent-header p { color: #666; margin-top: 0.25rem; }
    .tab-agent-footer { text-align: center; color: #999; font-size: 0.85rem; padding-top: 1rem; border-top: 1px solid #e5e7eb; margin-top: 1.5rem; }
    .status-box { min-height: 120px; }
    """
    with gr.Blocks(
        title="Tab Agent — AI Tablature Transcription",
        theme=gr.themes.Soft(),
        css=css,
    ) as demo:
        gr.HTML("""
            <div class="tab-agent-header">
                <h1>🎸 Tab Agent</h1>
                <p>Upload guitar or bass audio. Get tablature, MIDI, and JSON back.</p>
            </div>
            """)

        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=320):
                gr.Markdown("### 1. Upload Audio")
                audio_input = gr.Audio(
                    label="",
                    type="filepath",
                    sources=["upload"],
                )

                gr.Markdown("### 2. Configure")
                instrument_type = gr.Radio(
                    label="Instrument",
                    choices=["Guitar", "Bass"],
                    value="Guitar",
                )

                with gr.Group():
                    gr.Markdown("**Export formats**")
                    with gr.Row():
                        export_midi = gr.Checkbox(label="MIDI", value=True)
                        export_tab = gr.Checkbox(label="Tablature", value=True)
                        export_json = gr.Checkbox(label="JSON", value=True)

                transcribe_btn = gr.Button(
                    "Transcribe",
                    variant="primary",
                    size="lg",
                )

                gr.Examples(
                    examples=[
                        ["examples/guitar_solo.wav", "Guitar"],
                        ["examples/bass_groove.wav", "Bass"],
                    ],
                    inputs=[audio_input, instrument_type],
                    label="Try these samples",
                )

            with gr.Column(scale=3, min_width=420):
                gr.Markdown("### Results")
                status_output = gr.Markdown(
                    value="Upload audio and click **Transcribe** to begin.",
                    elem_classes="status-box",
                )
                download_output = gr.File(
                    label="Download ZIP",
                    interactive=False,
                    visible=False,
                )

        with gr.Accordion("How it works", open=False):
            gr.Markdown("""
            1. **Stem separation** — Demucs isolates guitar/bass from the mix
            2. **Spatial processing** — Mid-side technique splits lead from rhythm
            3. **AI transcription** — YourMT3+ or Basic Pitch converts audio → MIDI notes
            4. **Tablature generation** — Dynamic programming assigns notes to strings/frets
            5. **Technique detection** — Slides, hammer-ons, pull-offs annotated
            6. **Export** — MIDI, ASCII tab, JSON in a single ZIP
            """)

        with gr.Accordion("Links", open=False):
            gr.Markdown("""
            - [GitHub](https://github.com/scottmills306/tab-agent-pro)
            - [ReaPack](https://github.com/scottmills306/tab-agent-pro#reaper-integration)
            - [Basic Pitch](https://github.com/spotify/basic-pitch)
            """)

        gr.HTML("""
            <div class="tab-agent-footer">
                MIT · <a href="https://github.com/scottmills306/tab-agent-pro">Tab Agent</a> ·
                Built with <a href="https://github.com/spotify/basic-pitch">Basic Pitch</a> ·
                Python 3.10+
            </div>
            """)

        transcribe_btn.click(
            fn=_process_audio_impl,
            inputs=[audio_input, instrument_type, export_midi, export_tab, export_json],
            outputs=[status_output, download_output],
        ).then(
            fn=lambda path: gr.update(visible=path is not None),
            inputs=[download_output],
            outputs=[download_output],
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


def cleanup_temp_dirs():
    """Remove temporary output directories on shutdown."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def handle_signal(sig, frame):
    """Handle shutdown signals gracefully."""
    import sys

    print(f"\nReceived signal {sig}, shutting down...")
    cleanup_temp_dirs()
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)
atexit.register(cleanup_temp_dirs)


def cleanup_stale_sessions(max_age_hours: int = 1):
    """Remove session directories older than max_age_hours."""
    if not OUTPUT_DIR.exists():
        return
    now = datetime.now()
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir():
            try:
                age = now - datetime.fromtimestamp(d.stat().st_mtime)
                if age.total_seconds() > max_age_hours * 3600:
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass


# Main entry point
if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(create_app(), host=os.getenv("HOST", "127.0.0.1"), port=7860)
