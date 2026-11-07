"""
Tab Agent - Hugging Face Gradio Interface
AI-powered guitar/bass tablature transcription using Basic Pitch.

This is the web UI for the Tab Agent transcription system.

Deployment notes:
  * Binds to ``HOST`` (default ``0.0.0.0`` so Docker/Hugging Face Spaces can reach it).
  * Honours ``PORT`` (default 7860, the port Spaces routes to).
  * ``process_audio`` is decorated with ``spaces.GPU`` when the Space actually runs on
    Zero GPU hardware (``SPACES_ZERO_GPU``); otherwise it is a plain CPU function.
"""

import atexit
import contextlib
import os
import shutil
import signal
import sys
import tempfile
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr

from monitoring import get_logger, health


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
        print(f"⚠️  Could not read {path}: {e}", file=sys.stderr)


_load_dotenv()

# Import Tab Agent modules
from agents import EarAgent, SplitterAgent, TabAgent  # noqa: E402
from main import (  # noqa: E402
    export_tab_to_json,
    export_tab_to_txt,
    validate_thresholds,
)
from suno_postprocessor import SunoNotePostprocessor, process_suno_audio  # noqa: E402

# Zero GPU support: the decorator only reserves a GPU on ZeroGPU Spaces. On CPU
# Spaces `spaces.GPU` is a pass-through, so we still import `spaces` when present
# but only report GPU acceleration when the runtime really provides it.
try:
    import spaces

    SPACES_AVAILABLE = True
except ImportError:
    spaces = None  # type: ignore[assignment]
    SPACES_AVAILABLE = False

ZERO_GPU = os.getenv("SPACES_ZERO_GPU", "").lower() in ("1", "true")
GPU_AVAILABLE = SPACES_AVAILABLE and ZERO_GPU

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

# A single transcription holds Demucs + a Basic Pitch model in memory, so the
# queue must be bounded: an unbounded queue turns a busy Space into an OOM kill
# instead of a fast "busy" signal. One worker by default (each request is
# CPU-bound and already uses multiple threads internally).
MAX_CONCURRENT_REQUESTS = max(1, int(os.getenv("MAX_CONCURRENT_REQUESTS", "1") or "1"))
QUEUE_MAX_SIZE = int(os.getenv("QUEUE_MAX_SIZE", "32") or "32")
# Uploads older than this are removed on each request (seconds).
UPLOAD_TTL_SEC = int(os.getenv("UPLOAD_TTL_SEC", "900") or "900")

# PyTorch device for the transcription agent ("auto", "cpu", "cuda", "mps")
DEVICE = os.getenv("DEVICE", "auto")

BASS_TARGETS = {"Bass", "bass"}


def process_audio(
    audio_file,
    instrument_type="Guitar",
    include_midi=True,
    include_tab=True,
    include_json=True,
    progress=gr.Progress(),  # noqa: B008 - Gradio documented pattern
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


if GPU_AVAILABLE:

    @spaces.GPU
    def _gpu_process_audio(*args, **kwargs):
        """Run the pipeline on a Zero GPU slice. Only active on ZeroGPU Spaces."""
        return process_audio(*args, **kwargs)

    _reserve_gpu = True
else:
    _gpu_process_audio = None
    _reserve_gpu = False

# `ui_entry` is what the Gradio button is bound to: the Zero GPU wrapper when the
# runtime provides a GPU, otherwise the plain function. Both accept 5 inputs and
# a defaulted `progress`, so the number of connected inputs never matters.
ui_entry = _gpu_process_audio if _reserve_gpu else process_audio


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
        # Many decoder failures stringify to "": always name the error type so
        # the user gets an actionable message instead of "Cannot read audio: ".
        detail = str(e).strip() or type(e).__name__
        get_logger("app").warning(
            "audio_validation_failed",
            file=audio_path.name,
            error_type=type(e).__name__,
            detail=detail,
        )
        return f"Cannot read audio ({audio_path.suffix or 'no extension'}): {detail}"
    return None


def _load_saved_config() -> dict:
    """Read saved user_preferences.json if present without creating directories."""
    import json

    import init_memory

    prefs_path = os.path.join(init_memory.get_memory_dir(), "user_preferences.json")
    if not os.path.exists(prefs_path):
        return {}
    try:
        with open(prefs_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("config"), dict):
            return dict(data["config"])
    except (OSError, ValueError) as e:
        # Saved preferences are optional; report the problem and use defaults.
        print(
            f"⚠️  Ignoring unreadable saved preferences {prefs_path}: {e}",
            file=sys.stderr,
        )
    return {}


def _process_audio_impl(
    audio_file,
    instrument_type="Guitar",
    include_midi=True,
    include_tab=True,
    include_json=True,
    progress=gr.Progress(),  # noqa: B008 - Gradio documented pattern
) -> tuple[str, str | None]:
    """Internal implementation of audio processing."""
    cleanup_stale_sessions()

    if audio_file is None:
        return "❌ Please upload an audio file", None

    # Validate
    audio_path = Path(audio_file)
    error = _validate_audio(audio_path)
    if error:
        return f"❌ {error}", None

    try:
        # Create a unique output directory. A second-resolution timestamp plus
        # exist_ok=True let two concurrent requests share one directory, so one
        # user's ZIP could contain another user's files and outputs were
        # silently overwritten. mkdtemp guarantees a fresh, atomic directory,
        # which also covers main's same-second retry loop (#8) more simply.
        cleanup_stale_uploads()
        session_dir = Path(tempfile.mkdtemp(prefix="session_", dir=str(OUTPUT_DIR)))
        (session_dir / "stems").mkdir(exist_ok=True)

        # Track processing time for ETA / README accuracy
        start_time = time.time()

        # Get file info
        audio_path = Path(audio_file)
        song_name = audio_path.stem

        progress(0.1, desc="🎵 Initializing agents...")

        user_config = _load_saved_config()

        # Stage 0: Suno artifact detection and preprocessing
        progress(0.15, desc="🔍 Analyzing audio quality...")
        suno_kwargs: dict = {
            "output_path": str(session_dir / f"{song_name}_processed.wav"),
        }
        if "suno_aggressive_mode" in user_config:
            suno_kwargs["aggressiveness"] = 0.8 if user_config["suno_aggressive_mode"] else 0.2
        processed_audio, is_suno, suno_metrics = process_suno_audio(
            str(audio_path),
            **suno_kwargs,
        )
        health.set_component(
            "suno_detector",
            (
                f"ai_audio_detected (hf_ratio={suno_metrics.get('hf_ratio', 0):.4f})"
                if is_suno
                else "clean"
            ),
        )

        # Adjust thresholds for AI-generated audio (saved profile takes precedence)
        default_onset = 0.6 if is_suno else 0.5
        default_frame = 0.4 if is_suno else 0.3
        try:
            onset_threshold, frame_threshold = validate_thresholds(
                float(user_config.get("onset_threshold", default_onset)),
                float(user_config.get("frame_threshold", default_frame)),
            )
        except ValueError as e:
            # A corrupt user_preferences.json must not silently degrade output.
            return f"❌ Invalid saved configuration: {e}", None

        # Initialize agents
        runtime_device = os.getenv("DEVICE", DEVICE)
        splitter = SplitterAgent(
            output_dir=str(session_dir / "stems"),
            device=runtime_device,
        )
        ear = EarAgent(device=runtime_device)
        suno_postprocessor = SunoNotePostprocessor()

        # Stage 1-3: Stem separation and processing
        progress(0.2, desc="🎵 Separating audio stems (Demucs)...")
        stems = splitter.separate_stems(processed_audio)

        progress(0.3, desc="🎸 Processing guitar stems...")
        is_bass = instrument_type in BASS_TARGETS
        if not is_bass:
            guitar_stems = splitter.process_guitars(stems["guitar"])
            processed_stems = {
                "lead": guitar_stems["lead"],
                "rhythm_l": guitar_stems["left"],
                "rhythm_r": guitar_stems["right"],
            }
        else:  # Bass
            bass_clean = splitter.process_bass(stems["bass"])
            processed_stems = {"bass": bass_clean}

        tuning = (
            user_config.get("bass_tuning", BASS_TUNING)
            if is_bass
            else user_config.get("guitar_tuning", GUITAR_TUNING)
        )
        min_pitch = min(tuning) if tuning else None

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
                min_pitch=min_pitch,
            )
            notes_clean = ear.humanize_and_clean(notes_raw, is_bass=is_bass)
            # Apply Suno post-processing if needed
            notes_clean = suno_postprocessor.process(notes_clean, is_suno, suno_metrics)

            # Export MIDI
            if include_midi and notes_clean:
                midi_path = session_dir / f"{song_name}_{stem_name}.mid"
                ear.export_midi(notes_clean, str(midi_path))

            results[stem_name] = notes_clean

        # Stage 5: Tablature generation
        progress(0.8, desc="📝 Generating tablature...")

        num_frets = int(user_config.get("num_frets", 24))
        technique_sensitivity = float(user_config.get("technique_sensitivity", 0.7))
        prefer_low_strings = bool(user_config.get("prefer_low_strings", is_bass))
        tab_agent = TabAgent(
            tuning=tuning,
            num_frets=num_frets,
            prefer_low_strings=prefer_low_strings,
        )

        # Tracks that produced no usable tablature (e.g. all notes unplayable)
        skipped: list[str] = []
        counted_files: set[Path] = set()
        written = 0

        for stem_name, notes in results.items():
            tab_data = tab_agent.generate_tab(
                notes,
                technique_sensitivity=technique_sensitivity,
            )

            if notes and not tab_data:
                skipped.append(stem_name)

            # Export tab files
            if include_tab and tab_data:
                tab_path = session_dir / f"{song_name}_{stem_name}.tab"
                export_tab_to_txt(
                    tab_data,
                    str(tab_path),
                    instrument=f"{instrument_type} - {stem_name}",
                    num_strings=len(tuning),
                )
                counted_files.add(tab_path)
                written += 1

            if include_json and tab_data:
                json_path = session_dir / f"{song_name}_{stem_name}.json"
                export_tab_to_json(
                    tab_data,
                    str(json_path),
                    instrument=f"{instrument_type} - {stem_name}",
                    num_strings=len(tuning),
                )
                counted_files.add(json_path)
                written += 1

        # Create ZIP archive
        progress(0.9, desc="📦 Creating download package...")
        zip_path = session_dir / f"{song_name}_tablature.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in session_dir.rglob("*"):
                if file.is_file() and file != zip_path:
                    arcname = file.relative_to(session_dir)
                    zipf.write(file, arcname)
                    if file not in counted_files:
                        counted_files.add(file)
                        written += 1

        progress(1.0, desc="✅ Complete!")

        if not results or not any(results.values()):
            return (
                "⚠️ **No notes were transcribed.** The audio may be too quiet, "
                "or a transcription backend may be missing "
                "(check `/health` → `components`).",
                str(zip_path),
            )

        # Generate status message
        elapsed = time.time() - start_time
        enabled_formats = [
            name
            for name, enabled in (
                ("MIDI", include_midi),
                ("Tab", include_tab),
                ("JSON", include_json),
            )
            if enabled
        ]
        status_msg = f"""
✅ **Transcription Complete!**

- **Song**: {song_name}
- **Instrument**: {instrument_type}
- **Files Generated**: {written}
- **Processing Time**: {elapsed:.1f}s
- **Formats**: {", ".join(enabled_formats) or "none"}
"""
        if skipped:
            status_msg += (
                f"\n⚠️ No tablature for: {', '.join(skipped)} — notes fell outside the "
                "instrument's range/tuning.\n"
            )

        status_msg += "\n📥 **Download the ZIP file below to get all outputs!**\n"

        health.record_request(success=True, details=f"transcribe: {song_name}")
        return status_msg, str(zip_path)

    except Exception as e:
        from monitoring import get_logger

        # Full detail (type + message) goes to the structured log; the web user
        # gets a correlation-friendly message without internal paths or
        # library internals.
        error_ref = uuid.uuid4().hex[:8]
        get_logger("app").error(
            "transcription_failed",
            exc=e,
            error_ref=error_ref,
            file=Path(audio_file).name,
        )
        health.record_request(success=False, details=f"transcribe: {Path(audio_file).name}")
        error_msg = (
            "❌ **Transcription failed.**\n\n"
            f"Error reference: `{error_ref}` (quote this in a bug report). "
            "The server log has the full traceback. Check `/health` → "
            "`components` to confirm a transcription backend is available."
        )
        return error_msg, None


# Create Gradio interface
UI_CSS = """
    .tab-agent-header h1 { font-size: 2.2rem; margin-bottom: 0; }
    .tab-agent-header p { color: #666; margin-top: 0.25rem; }
    .tab-agent-footer { text-align: center; color: #999; font-size: 0.85rem; padding-top: 1rem; border-top: 1px solid #e5e7eb; margin-top: 1.5rem; }
    .status-box { min-height: 120px; }
    """


def _gradio_major() -> int:
    with contextlib.suppress(Exception):
        return int(gr.__version__.split(".")[0])
    return 4


def _ui_style_kwargs() -> dict:
    """
    Theme/CSS live in the Blocks constructor on Gradio < 6 and moved to
    mount_gradio_app()/launch() on Gradio 6+.
    """
    if _gradio_major() >= 6:
        return {}
    return {"theme": gr.themes.Soft(), "css": UI_CSS}


def create_ui():
    """Create Gradio UI interface."""
    with gr.Blocks(
        title="Tab Agent — AI Tablature Transcription",
        **_ui_style_kwargs(),
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
            1. **Stem separation** — Demucs isolates the guitar/bass/other stems from the mix
            2. **Spatial processing** — Mid-side technique splits lead from rhythm
            3. **AI transcription** — Basic Pitch converts audio → MIDI notes
            4. **Tablature generation** — Dynamic programming assigns notes to strings/frets
            5. **Technique detection** — Slides, hammer-ons, pull-offs annotated
            6. **Export** — MIDI, ASCII tab, JSON in a single ZIP
            """)

        with gr.Accordion("Links", open=False):
            gr.Markdown("""
            - [GitHub](https://github.com/sgm-audio/tab-agent-pro)
            - [ReaPack](https://github.com/sgm-audio/tab-agent-pro#reaper-integration)
            - [Basic Pitch](https://github.com/spotify/basic-pitch)
            """)

        gr.HTML("""
            <div class="tab-agent-footer">
                MIT · <a href="https://github.com/sgm-audio/tab-agent-pro">Tab Agent</a> ·
                Built with <a href="https://github.com/spotify/basic-pitch">Basic Pitch</a> ·
                Python 3.10+
            </div>
            """)

        transcribe_btn.click(
            fn=ui_entry,
            inputs=[audio_input, instrument_type, export_midi, export_tab, export_json],
            outputs=[status_output, download_output],
        ).then(
            fn=lambda path: gr.update(visible=path is not None),
            inputs=[download_output],
            outputs=[download_output],
        )

    return demo


def create_app() -> gr.Blocks:
    """Create FastAPI app with health endpoints and Gradio UI mounted."""
    from fastapi import FastAPI

    demo = create_ui()
    # Bounded queue: refuse work beyond MAX_CONCURRENT_REQUESTS + QUEUE_MAX_SIZE
    # rather than accepting unlimited jobs and exhausting memory.
    demo.queue(
        default_concurrency_limit=MAX_CONCURRENT_REQUESTS,
        max_size=QUEUE_MAX_SIZE,
    )

    parent_app = FastAPI()

    @parent_app.get("/health")
    async def health_endpoint():
        """Liveness + component detail. Always 200 while the process serves."""
        return health.as_dict()

    @parent_app.get("/ready")
    async def ready_endpoint():
        """
        Readiness probe: 200 only when a usable transcription backend exists.

        Liveness (/health) answers "is the process alive"; readiness answers
        "can it actually do its job". Conflating them is what let an image with
        no working backend pass its container healthcheck.
        """
        from fastapi.responses import JSONResponse

        payload = {
            "ready": health.is_ready(),
            "status": health.effective_status(),
            "components": health.component_health(),
        }
        return JSONResponse(status_code=200 if payload["ready"] else 503, content=payload)

    @parent_app.get("/health/metrics")
    async def metrics_endpoint():
        from monitoring import default_metrics

        return default_metrics.summary()

    mount_kwargs: dict = {}
    if _gradio_major() >= 6:
        mount_kwargs = {"theme": gr.themes.Soft(), "css": UI_CSS}

    return gr.mount_gradio_app(parent_app, demo, path="/", **mount_kwargs)


def cleanup_temp_dirs() -> None:
    """Remove temporary output directories on shutdown."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def handle_signal(sig, frame) -> None:
    """Handle shutdown signals gracefully."""
    import sys

    cleanup_temp_dirs()
    sys.exit(0)


atexit.register(cleanup_temp_dirs)


def cleanup_stale_sessions(max_age_hours: float = 1) -> None:
    """Remove session directories older than max_age_hours."""
    if not OUTPUT_DIR.exists():
        return
    now = datetime.now()
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir():
            with contextlib.suppress(OSError):
                age = now - datetime.fromtimestamp(d.stat().st_mtime)
                if age.total_seconds() > max_age_hours * 3600:
                    shutil.rmtree(d, ignore_errors=True)


def _now_timestamp() -> float:
    """Wall-clock seconds, isolated so datetime patching cannot affect pruning."""
    import datetime as _dt

    return _dt.datetime.now().timestamp()


def cleanup_stale_uploads(max_age_sec: int = UPLOAD_TTL_SEC) -> int:
    """
    Delete uploaded files Gradio has finished with but left on disk.

    Gradio writes each upload to a temp directory and never removes it, so a
    long-lived Space accumulates every audio file ever submitted. Only files
    inside the Gradio temp root are touched, and only once they are older than
    ``max_age_sec`` so in-flight requests are not disturbed. Returns the number
    of entries removed.
    """
    removed = 0
    root = Path(tempfile.gettempdir()) / "gradio"
    if not root.is_dir():
        return 0
    # Use a real clock so tests (or callers) that patch app.datetime cannot
    # turn pruning into a no-op or delete live uploads.
    cutoff = _now_timestamp() - max_age_sec
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0
    for entry in entries:
        with contextlib.suppress(OSError):
            if entry.stat().st_mtime >= cutoff:
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
            removed += 1
    if removed:
        get_logger("app").info("uploads_pruned", count=removed, ttl_s=max_age_sec)
    return removed


def serve() -> None:
    """
    Run the FastAPI + Gradio app with uvicorn.

    Binds HOST (default 0.0.0.0 so Docker/Hugging Face Spaces can reach it) and
    PORT (default 7860, the port Spaces routes to).
    """
    import threading

    import uvicorn

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

    # Binding 0.0.0.0 is required for container/Hugging Face Spaces routing;
    # set HOST=127.0.0.1 for local-only use.
    uvicorn.run(
        create_app(),
        host=os.getenv("HOST", "0.0.0.0"),  # nosec B104
        port=int(os.getenv("PORT", "7860")),
    )


# Main entry point
if __name__ == "__main__":
    serve()
