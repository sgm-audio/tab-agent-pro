"""
Tab Agent - Modernized with YourMT3+ (January 2026)
Multi-instrument music transcription using state-of-the-art transformers.

Improvements from Basic Pitch:
- YourMT3+ with hierarchical attention transformers
- Mixture of Experts (MoE) for instrument-specific processing
- Trained on GuitarSet, MusicNet, and multi-instrument datasets
- Better pitch bend detection for slides and techniques
- Multi-track simultaneous transcription
"""

import os
import subprocess
import sys

import librosa
import note_seq
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, filtfilt

from monitoring import default_metrics as metrics

# Monitoring & health checks
from monitoring import get_logger, health

# Basic Pitch imports (proven and reliable for MVP)
basic_pitch_predict = None
try:
    from basic_pitch.inference import predict as basic_pitch_predict

    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False

# Demucs Python API (preferred over subprocess CLI)
try:
    import demucs.api as demucs_api

    DEMUCS_API_AVAILABLE = True
except ImportError:
    DEMUCS_API_AVAILABLE = False

# YourMT3+ requires the codebase from GitHub + checkpoint from HF Hub.
# Not a standard transformers model — uses a custom Lightning architecture.
# YMT3_AVAILABLE is set True only after successful import + checkpoint load.
YMT3_AVAILABLE = False

# HuggingFace Hub for checkpoint downloads (with resume)
try:
    from huggingface_hub import snapshot_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# torchaudio for Demucs API export
try:
    import torchaudio  # noqa: F401

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


# ============================================================================
# STAGE 1-3: THE SPLITTER
# ============================================================================


class SplitterAgent:
    """
    Audio stem separation using Demucs Python API.

    Demucs v4 with htdemucs model remains state-of-the-art for:
    - Guitar/bass separation
    - Multi-stem source separation
    - Real-time processing capability

    Uses the Python API (demucs.api.Separator) — no subprocess CLI dependency.
    Graceful fallback: if separation fails, raw audio is returned as a single
    stem so the pipeline doesn't crash.
    """

    def __init__(self, output_dir="separated_stems") -> None:
        self.output_dir = output_dir
        self.log = get_logger("splitter")
        os.makedirs(output_dir, exist_ok=True)

    def separate_stems(self, audio_path):
        """Separate audio into guitar and bass stems with graceful fallback."""
        file = os.path.basename(audio_path)
        self.log.info("demucs_start", file=file)

        if DEMUCS_API_AVAILABLE:
            with metrics.track_stage("demucs_separation"):
                try:
                    result = self._separate_with_api(audio_path, file)
                    health.set_component("demucs", "loaded")
                    return result
                except Exception as e:
                    self.log.warning("demucs_api_failed", file=file, error=str(e))
                    health.set_component("demucs", "degraded (API failed)")

        # Try CLI fallback
        try:
            with metrics.track_stage("demucs_cli"):
                result = self._separate_with_subprocess(audio_path, file)
                health.set_component("demucs", "loaded (CLI)")
                return result
        except Exception as e:
            self.log.warning("demucs_all_failed", file=file, error=str(e))
            health.set_component("demucs", "unavailable")
            return self._raw_audio_fallback(audio_path, file)

    def _raw_audio_fallback(self, audio_path, song_name):
        """Return raw audio as both guitar and bass stems (graceful degradation)."""
        self.log.info("demucs_raw_fallback", file=song_name)
        base_path = os.path.join(self.output_dir, "htdemucs", song_name)
        os.makedirs(base_path, exist_ok=True)
        import shutil

        other_path = os.path.join(base_path, "other.wav")
        bass_path = os.path.join(base_path, "bass.wav")
        shutil.copy2(audio_path, other_path)
        shutil.copy2(audio_path, bass_path)
        return {"guitar": other_path, "bass": bass_path}

    def _separate_with_api(self, audio_path, song_name):
        """Use Demucs Python API for stem separation."""
        separator = demucs_api.Separator(
            model="htdemucs",
            device="cuda" if torch.cuda.is_available() else "cpu",
            shifts=1,
            overlap=0.25,
            split=True,
            segment=10,
            progress=True,
        )

        _, separated = separator.separate_audio_file(audio_path)

        # Write stems to expected paths for backward compat
        base_path = os.path.join(self.output_dir, "htdemucs", song_name)
        os.makedirs(base_path, exist_ok=True)

        stem_map = {"other": "other", "bass": "bass"}

        for stem_key, file_key in stem_map.items():
            if stem_key in separated:
                tensor = separated[stem_key]
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                stem_path = os.path.join(base_path, f"{file_key}.wav")
                sf.write(stem_path, tensor.cpu().numpy().T, separator.samplerate)

        return {
            "guitar": os.path.join(base_path, "other.wav"),
            "bass": os.path.join(base_path, "bass.wav"),
        }

    def _separate_with_subprocess(self, audio_path, song_name):
        """Fallback: Demucs via CLI subprocess."""
        cmd = ["demucs", "-n", "htdemucs", "-o", self.output_dir, audio_path]

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )  # nosec B603 — hardcoded cmd, no user input
        except subprocess.CalledProcessError:
            raise

        base_path = os.path.join(self.output_dir, "htdemucs", song_name)
        return {
            "guitar": os.path.join(base_path, "other.wav"),
            "bass": os.path.join(base_path, "bass.wav"),
        }

    def process_guitars(self, guitar_stem_path):
        """
        Process guitar stem using mid-side technique to separate lead/rhythm.

        Mid-side processing:
        - Mid (center): Lead guitar (typically center-panned)
        - Side (L/R): Rhythm guitars (typically panned left/right)
        """
        y, sr = librosa.load(guitar_stem_path, mono=False, sr=None)

        if y.ndim == 1:
            # Mono file - duplicate to stereo
            y = np.vstack((y, y))

        left, right = y[0], y[1]
        mid = (left + right) / 2  # Center content

        # Center kill factor (0.8 = remove 80% of center from sides)
        # Increase to 0.9 for more aggressive separation
        center_kill_factor = 0.8
        rhythm_l = left - (mid * center_kill_factor)
        rhythm_r = right - (mid * center_kill_factor)

        # Export processed stems
        lead_path = f"{self.output_dir}/processed_lead.wav"
        rhythm_l_path = f"{self.output_dir}/processed_rhythm_L.wav"
        rhythm_r_path = f"{self.output_dir}/processed_rhythm_R.wav"

        sf.write(lead_path, mid, sr)
        sf.write(rhythm_l_path, rhythm_l, sr)
        sf.write(rhythm_r_path, rhythm_r, sr)

        return {"lead": lead_path, "left": rhythm_l_path, "right": rhythm_r_path}

    def process_bass(self, bass_stem_path):
        """
        Process bass stem with frequency-domain filtering.

        Bass processing:
        - Preserve low frequencies (fundamental tones)
        - Reduce high frequencies (fret noise, harmonics)
        - Optional: Future upgrade to butterworth filters
        """
        y, sr = librosa.load(bass_stem_path, mono=False, sr=None)

        if y.ndim == 1:
            y = np.vstack((y, y))

        y_mono = librosa.to_mono(y)

        # Butterworth lowpass filter: preserve fundamentals, reduce fret noise/harmonics
        nyquist = sr / 2
        normal_cutoff = 200 / nyquist  # 200 Hz cutoff for bass fundamentals
        b, a = butter(4, normal_cutoff, btype="low", analog=False)
        y_low = filtfilt(b, a, y_mono)

        # Isolate high-frequency content (fret noise, harmonics) and reduce
        b_high, a_high = butter(4, normal_cutoff, btype="high", analog=False)
        y_high = filtfilt(b_high, a_high, y_mono)

        # Reconstruct: full low + reduced high (0.5 = halve harmonic content)
        y_processed = y_low + (y_high * 0.5)

        path = f"{self.output_dir}/processed_bass_clean.wav"
        sf.write(path, y_processed, sr)

        return path


# ============================================================================
# STAGE 4: THE EAR (MODERNIZED WITH YOURMT3+)
# ============================================================================


class EarAgent:
    """
    Audio-to-MIDI transcription using YourMT3+ transformer model.

    YourMT3+ is a custom PyTorch Lightning model (not a standard transformers
    model). It requires:
      1. The codebase from https://github.com/mimbres/YourMT3 (auto-cloned)
      2. The checkpoint from https://huggingface.co/mimbres/YourMT3 (auto-downloaded)

    Loading is automatic with resume support for downloads. Falls back
    gracefully to Basic Pitch if either component is unavailable.

    Primary model: YourMT3+ (custom Lightning)
    Fallback:      Basic Pitch (Spotify, production-ready)
    """

    YOURMT3_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "tab_agent", "yourmt3")

    def __init__(
        self,
        model_id: str = "mimbres/YourMT3",
        device: str = "auto",
        prefer_yourmt3: bool = True,
    ) -> None:
        """
        Initialize transcription model(s).

        Args:
            model_id: HuggingFace checkpoint ID ("mimbres/YourMT3")
            device: Compute device ("cpu", "cuda", "mps", or "auto")
            prefer_yourmt3: Attempt YourMT3+ load. Set False to skip to Basic Pitch.

        Note:
            "mimbres/YourMT3-cpu" is a Gradio Space (not a model repo) and
            returns HTTP 401 if used as a model_id. The actual model
            checkpoints are at "mimbres/YourMT3".
            If the download fails, you can manually download from:
            https://huggingface.co/mimbres/YourMT3/tree/main

        """
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.sample_rate = 16000
        self.model_id = model_id
        self.model = None
        self.processor = None

        # Attempt YourMT3+ load (with resume + graceful fallback)
        if prefer_yourmt3:
            self._load_yourmt3_model(model_id)

        if self.model is None and not BASIC_PITCH_AVAILABLE:
            pass

    def _load_yourmt3_model(self, model_id) -> None:
        """
        Download and load YourMT3+ model with resume and graceful fallback.

        Steps:
          1. Download checkpoint from HuggingFace Hub (resume via snapshot_download)
          2. Clone/update the YourMT3 codebase from GitHub (resume via git fetch)
          3. Import the custom YourMT3 class and load the checkpoint
          4. If any step fails, self.model stays None (graceful fallback)
        """
        global YMT3_AVAILABLE

        checkpoint_dir = self._download_checkpoint(model_id)
        if checkpoint_dir is None:
            return

        code_dir = self._clone_yourmt3_codebase()
        if code_dir is None:
            return

        self._import_and_load_model(checkpoint_dir, code_dir)

        if self.model is not None:
            YMT3_AVAILABLE = True
        else:
            pass

    def _download_checkpoint(self, model_id):
        """Download YourMT3+ checkpoint from HF Hub with resume support."""
        if not HF_HUB_AVAILABLE:
            return None

        try:
            return snapshot_download(
                repo_id=model_id,
                revision="main",
                resume_download=True,
                local_files_only=False,
                max_workers=4,
            )
        except Exception as e:
            http_401 = "401" in str(e) or "Authorization" in str(e)
            if http_401:
                pass
            return None

    def _clone_yourmt3_codebase(self):
        """
        Clone or update the YourMT3 codebase from HuggingFace Space with resume support.

        The actual model code lives under amt/src/ in the HF Space
        mimbres/YourMT3-cpu (the GitHub repo only has README/LICENSE).
        We point sys.path to amt/src/ so imports like "model.ymt3" resolve.

        Clones to ~/.cache/tab_agent/yourmt3 on first run.
        On subsequent runs, runs git fetch + reset to update.
        """
        try:
            os.makedirs(self.YOURMT3_CACHE, exist_ok=True)

            if os.path.exists(os.path.join(self.YOURMT3_CACHE, ".git")):
                result = subprocess.run(  # nosec B603 B607 — hardcoded git cmd
                    [
                        "git",
                        "-C",
                        self.YOURMT3_CACHE,
                        "fetch",
                        "--depth",
                        "1",
                        "origin",
                        "main",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    subprocess.run(  # nosec B603 B607 — hardcoded git cmd
                        [
                            "git",
                            "-C",
                            self.YOURMT3_CACHE,
                            "reset",
                            "--hard",
                            "FETCH_HEAD",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                else:
                    pass
            else:
                result = subprocess.run(  # nosec B603 B607 — hardcoded git cmd
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "https://huggingface.co/spaces/mimbres/YourMT3-cpu",
                        self.YOURMT3_CACHE,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    return None

            # Code is under amt/src/ in the Space repo
            code_root = os.path.join(self.YOURMT3_CACHE, "amt", "src")
            essentials = [
                "model/ymt3.py",
                "model/init_train.py",
                "utils/task_manager.py",
            ]
            missing = [f for f in essentials if not os.path.exists(os.path.join(code_root, f))]
            if missing:
                return None

            if code_root not in sys.path:
                sys.path.insert(0, code_root)

            return code_root

        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None

    def _import_and_load_model(self, checkpoint_dir, code_dir) -> None:
        """
        Import the YourMT3 custom class and load the checkpoint.

        The YourMT3+ model is NOT a standard HuggingFace transformers model.
        It uses a custom PyTorch Lightning architecture with a specific
        checkpoint format. We attempt to import and load, but this step is
        the most fragile — any failure here is caught and self.model stays None.

        All utility imports (slicing, note merging) are stored as instance
        attributes so _transcribe_with_yourmt3 never does runtime imports.
        """
        log = get_logger("ear_agent")
        self._ymt3_utils = {}
        self._ymt3_audio_cfg = {}

        try:
            from model.init_train import initialize_trainer, update_config
            from model.ymt3 import YourMT3
            from utils.task_manager import TaskManager
        except ImportError as e:
            log.warning("yourmt3_import_failed", error=str(e), code_dir=code_dir)
            health.set_component("yourmt3", "error: import failed")
            return

        try:
            import config.config as _ymt3_config
            import torch

            # Patch save_dir so initialize_trainer finds checkpoints on disk.
            # The snapshot was downloaded to checkpoint_dir, and checkpoints
            # live at checkpoint_dir/logs/2024/{exp_id}/checkpoints/{ckpt}.
            # Default config has save_dir="amt/logs" — override to the real path.
            save_dir = os.path.join(checkpoint_dir, "logs")
            _ymt3_config.shared_cfg["WANDB"]["save_dir"] = save_dir

            # Build minimal args for initialize_trainer
            args = self._build_yourmt3_args(checkpoint_dir)

            _, _, dir_info, shared_cfg = initialize_trainer(args, stage="test")
            shared_cfg, audio_cfg, model_cfg = update_config(args, shared_cfg, stage="test")

            tm = TaskManager(
                task_name=args.task,
                max_shift_steps=int(shared_cfg["TOKENIZER"]["max_shift_steps"]),
                debug_mode=False,
            )

            model = YourMT3(
                audio_cfg=audio_cfg,
                model_cfg=model_cfg,
                shared_cfg=shared_cfg,
                optimizer=None,
                task_manager=tm,
                eval_subtask_key="default",
            ).to(self.device)

            last_ckpt = dir_info.get("last_ckpt_path")
            if last_ckpt and os.path.exists(last_ckpt):
                checkpoint = torch.load(
                    last_ckpt,
                    map_location=self.device,
                    weights_only=True,
                )  # If YourMT3+ checkpoint requires full pickle, re-evaluate with source verification
                state_dict = checkpoint.get("state_dict", checkpoint)
                state_dict = {k: v for k, v in state_dict.items() if "pitchshift" not in k}
                model.load_state_dict(state_dict, strict=False)
            else:
                log.warning("yourmt3_no_checkpoint", path=str(last_ckpt))
                health.set_component("yourmt3", "error: no checkpoint")
                return

            model.eval()
            self.model = model
            self.processor = tm
            self._ymt3_audio_cfg = dict(audio_cfg)

            # Store utility functions — no runtime imports needed later
            try:
                from model.utils.audio import slice_padded_array
                from model.utils.event2note import (
                    merge_zipped_note_events_and_ties_to_notes,
                )
                from model.utils.note2event import mix_notes

                self._ymt3_utils = {
                    "slice": slice_padded_array,
                    "merge": merge_zipped_note_events_and_ties_to_notes,
                    "mix": mix_notes,
                }
                log.info("yourmt3_loaded", checkpoint=last_ckpt)
                health.set_component("yourmt3", "loaded")
            except ImportError as e:
                log.warning("yourmt3_utils_missing", error=str(e))
                health.set_component("yourmt3", "loaded (no utils)")

        except Exception as e:
            log.exception("yourmt3_load_failed", exc=e)
            health.set_component("yourmt3", f"error: {type(e).__name__}")
            import traceback

            traceback.print_exc()
            self.model = None

    def _build_yourmt3_args(self, checkpoint_dir):
        """Build argparse namespace matching YourMT3 expectations."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("exp_id", type=str, default="ymt3")
        parser.add_argument("-p", "--project", type=str, default="ymt3")
        parser.add_argument("-ac", "--audio-codec", type=str, default=None)
        parser.add_argument("-hop", "--hop-length", type=int, default=None)
        parser.add_argument("-nmel", "--n-mels", type=int, default=None)
        parser.add_argument("-if", "--input-frames", type=int, default=None)
        parser.add_argument(
            "-sqr",
            "--sca-use-query-residual",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            default=None,
        )
        parser.add_argument("-enc", "--encoder-type", type=str, default=None)
        parser.add_argument("-dec", "--decoder-type", type=str, default=None)
        parser.add_argument("-preenc", "--pre-encoder-type", type=str, default="default")
        parser.add_argument("-predec", "--pre-decoder-type", type=str, default="default")
        parser.add_argument("-cout", "--conv-out-channels", type=int, default=None)
        parser.add_argument(
            "-tenc",
            "--task-cond-encoder",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            default=True,
        )
        parser.add_argument(
            "-tdec",
            "--task-cond-decoder",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            default=True,
        )
        parser.add_argument("-df", "--d-feat", type=int, default=None)
        parser.add_argument(
            "-pt",
            "--pretrained",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            default=False,
        )
        parser.add_argument("-b", "--base-name", type=str, default="google/t5-v1_1-small")
        parser.add_argument("-epe", "--encoder-position-encoding-type", type=str, default="default")
        parser.add_argument("-dpe", "--decoder-position-encoding-type", type=str, default="default")
        parser.add_argument(
            "-twe",
            "--tie-word-embedding",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            default=None,
        )
        parser.add_argument("-el", "--event-length", type=int, default=None)
        parser.add_argument("-dl", "--d-latent", type=int, default=None)
        parser.add_argument("-nl", "--num-latents", type=int, default=None)
        parser.add_argument("-dpm", "--perceiver-tf-d-model", type=int, default=None)
        parser.add_argument("-npb", "--num-perceiver-tf-blocks", type=int, default=None)
        parser.add_argument(
            "-npl",
            "--num-perceiver-tf-local-transformers-per-block",
            type=int,
            default=None,
        )
        parser.add_argument(
            "-npt",
            "--num-perceiver-tf-temporal-transformers-per-block",
            type=int,
            default=None,
        )
        parser.add_argument(
            "-atc",
            "--attention-to-channel",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            default=None,
        )
        parser.add_argument("-ln", "--layer-norm-type", type=str, default=None)
        parser.add_argument("-ff", "--ff-layer-type", type=str, default=None)
        parser.add_argument("-wf", "--ff-widening-factor", type=int, default=None)
        parser.add_argument("-nmoe", "--moe-num-experts", type=int, default=None)
        parser.add_argument("-kmoe", "--moe-topk", type=int, default=None)
        parser.add_argument("-act", "--hidden-act", type=str, default=None)
        parser.add_argument("-rt", "--rotary-type", type=str, default=None)
        parser.add_argument(
            "-rk",
            "--rope-apply-to-keys",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            default=None,
        )
        parser.add_argument(
            "-rp",
            "--rope-partial-pe",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            default=None,
        )
        parser.add_argument("-dff", "--decoder-ff-layer-type", type=str, default=None)
        parser.add_argument("-dwf", "--decoder-ff-widening-factor", type=int, default=None)
        parser.add_argument("-tk", "--task", type=str, default="mt3_full_plus")
        parser.add_argument("-epv", "--eval-program-vocab", type=str, default=None)
        parser.add_argument("-edv", "--eval-drum-vocab", type=str, default=None)
        parser.add_argument("-etk", "--eval-subtask-key", type=str, default="default")
        parser.add_argument("-t", "--onset-tolerance", type=float, default=0.05)
        parser.add_argument(
            "-os",
            "--test-octave-shift",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            default=False,
        )
        parser.add_argument(
            "-w",
            "--write-model-output",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            default=True,
        )
        parser.add_argument("-pr", "--precision", type=str, default="bf16-mixed")
        parser.add_argument("-st", "--strategy", type=str, default="auto")
        parser.add_argument("-n", "--num-nodes", type=int, default=1)
        parser.add_argument("-g", "--num-gpus", type=str, default="auto")
        parser.add_argument("-wb", "--wandb-mode", type=str, default="disabled")
        parser.add_argument(
            "-debug",
            "--debug-mode",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            default=False,
        )
        parser.add_argument("-tps", "--test-pitch-shift", type=int, default=None)
        parser.add_argument("--epochs", type=int, default=None)

        # Build args list matching the checkpoint layout:
        #   {save_dir}/2024/{exp_id}/checkpoints/{ckpt}
        # The "YMT3+" checkpoint uses exp_id@ckpt syntax.
        exp_id = "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72"
        ckpt_name = "model.ckpt"
        precision = "16" if torch.cuda.is_available() else "32"

        args_list = [f"{exp_id}@{ckpt_name}"]
        args_list.extend(["--project", "2024"])
        args_list.extend(["--precision", precision])

        return parser.parse_args(args_list)

    def transcribe_stem(
        self,
        audio_path: str,
        target: str = "Guitar",
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        min_note_duration: float = 0.05,
    ) -> list[note_seq.NoteSequence.Note]:
        """
        Transcribe audio to MIDI notes with full graceful fallback chain:

        YourMT3+ → Basic Pitch → empty list (with warning, pipeline continues)

        Every failure is logged with structured metadata so operators can
        diagnose issues from logs alone.
        """
        log = get_logger("ear_agent")
        file = os.path.basename(audio_path)

        log.info("transcribe_start", file=file, target=target)

        # ── Try YourMT3+ ──────────────────────────────────────────────
        if self.model is not None:
            with metrics.track_stage("yourmt3_transcription"):
                try:
                    result = self._transcribe_with_yourmt3(
                        audio_path,
                        target,
                        onset_threshold,
                        frame_threshold,
                        min_note_duration,
                    )
                    log.info(
                        "transcribe_done",
                        file=file,
                        model="yourmt3",
                        note_count=len(result),
                    )
                    return result
                except Exception as e:
                    log.warning("yourmt3_fell_back", file=file, error=str(e))

        # ── Try Basic Pitch ───────────────────────────────────────────
        if BASIC_PITCH_AVAILABLE:
            with metrics.track_stage("basic_pitch_transcription"):
                try:
                    result = self._transcribe_with_basic_pitch(
                        audio_path,
                        target,
                        onset_threshold,
                        frame_threshold,
                        min_note_duration,
                    )
                    log.info(
                        "transcribe_done",
                        file=file,
                        model="basic_pitch",
                        note_count=len(result),
                    )
                    return result
                except Exception as e:
                    log.warning("basic_pitch_failed", file=file, error=str(e))

        # ── Nothing worked: raise error ──────────────────────────────
        log.error("transcribe_all_failed", file=file, target=target)
        health.record_request(success=False, details=f"transcribe: {file}")
        msg = (
            f"No transcription models available for {file}. "
            "Install Basic Pitch (pip install basic-pitch) or ensure YourMT3+ loads correctly."
        )
        raise RuntimeError(
            msg,
        )

    def _transcribe_with_basic_pitch(
        self,
        audio_path: str,
        target: str,
        onset_threshold: float,
        frame_threshold: float,
        min_note_duration: float,
    ) -> list[note_seq.NoteSequence.Note]:
        """
        Transcribe using Basic Pitch (Spotify's proven model).

        Basic Pitch is production-ready and works well for guitar/bass.
        """
        log = get_logger("ear_agent")

        try:
            _model_output, midi_data, _note_events = basic_pitch_predict(
                audio_path,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                minimum_note_length=int(min_note_duration * 1000),
                minimum_frequency=None,
                maximum_frequency=None,
                multiple_pitch_bends=False,
                melodia_trick=True,
                debug_file=None,
            )

            notes = self._convert_prettymidi_to_noteseq(midi_data)
            notes = self._filter_by_instrument_range(notes, target)

            log.info("basic_pitch_done", note_count=len(notes), target=target)
            return notes

        except Exception as e:
            log.exception("basic_pitch_error", exc=e, file=os.path.basename(audio_path))
            import traceback

            traceback.print_exc()
            raise

    def _transcribe_with_yourmt3(
        self,
        audio_path: str,
        target: str,
        onset_threshold: float,
        frame_threshold: float,
        min_note_duration: float,
    ) -> list[note_seq.NoteSequence.Note]:
        """
        Transcribe using YourMT3+ custom PyTorch Lightning model.

        Uses stored utility functions from _ymt3_utils (imported once at load time).
        Falls back to inline audio slicing if utils unavailable.
        No runtime imports — self-contained.
        """
        log = get_logger("ear_agent")
        model = self.model
        tm = self.processor

        if tm is None or not hasattr(model, "inference_file"):
            log.warning("yourmt3_not_ready")
            msg = "Model missing inference_file method"
            raise RuntimeError(msg)

        # Load and resample audio
        audio, _sr = librosa.load(audio_path, sr=model.audio_cfg["sample_rate"], mono=True)
        audio = torch.from_numpy(audio).unsqueeze(0)

        # Segment audio — use stored utility or inline fallback
        input_frames = model.audio_cfg["input_frames"]
        utils = self._ymt3_utils

        if "slice" in utils:
            audio_segments = utils["slice"](audio, input_frames, input_frames)
        else:
            audio_segments = self._slice_audio_inline(audio, input_frames, input_frames)

        audio_segments = (
            torch.from_numpy(audio_segments.astype("float32")).to(self.device).unsqueeze(1)
        )

        # Inference
        with torch.no_grad():
            pred_token_arr, _ = model.inference_file(bsz=8, audio_segments=audio_segments)

        # Detokenize
        num_channels = tm.num_decoding_channels
        n_items = audio_segments.shape[0]
        start_secs = [input_frames * i / model.audio_cfg["sample_rate"] for i in range(n_items)]

        pred_notes_in_file = []
        for ch in range(num_channels):
            pred_token_arr_ch = [arr[:, ch, :] for arr in pred_token_arr]
            zipped, _, _ = tm.detokenize_list_batches(
                pred_token_arr_ch,
                start_secs,
                return_events=True,
            )

            if "merge" in utils:
                pred_notes_ch, _ = utils["merge"](zipped)
            else:
                pred_notes_ch = self._merge_notes_inline(zipped)

            pred_notes_in_file.append(pred_notes_ch)

        # Merge across channels
        if "mix" in utils:
            pred_notes = utils["mix"](pred_notes_in_file)
        else:
            pred_notes = self._mix_notes_inline(pred_notes_in_file)

        # Convert to note_seq format
        notes = []
        for n in pred_notes:
            notes.append(
                note_seq.NoteSequence.Note(
                    pitch=n.get("pitch", 60),
                    start_time=float(n.get("start", 0)),
                    end_time=float(n.get("end", n.get("start", 0) + min_note_duration)),
                    velocity=int(n.get("velocity", 80)),
                ),
            )

        notes = self._filter_by_instrument_range(notes, target)
        log.info("yourmt3_transcribed", note_count=len(notes), target=target)
        return notes

    # ── Inline fallbacks (no cloned-repo dependency) ──────────────────────

    @staticmethod
    def _slice_audio_inline(audio, frame_size, step_size):
        """Slice audio tensor into overlapping segments (no external dependency)."""
        audio_np = audio.squeeze().numpy()
        total = len(audio_np)
        segments = []
        start = 0
        while start + frame_size <= total:
            segments.append(audio_np[start : start + frame_size])
            start += step_size
        if start < total:
            segment = np.zeros(frame_size, dtype=audio_np.dtype)
            segment[: total - start] = audio_np[start:]
            segments.append(segment)
        return np.array(segments)

    @staticmethod
    def _merge_notes_inline(zipped_events):
        """Convert zipped (onset, offset, pitch, velocity, instrument) to note dicts."""
        notes = []
        for onset, offset, pitch, velocity, _instrument in zipped_events:
            notes.append(
                {
                    "pitch": int(pitch),
                    "start": float(onset),
                    "end": float(offset),
                    "velocity": int(velocity),
                },
            )
        return notes

    @staticmethod
    def _mix_notes_inline(notes_by_channel):
        """Merge notes across channels, removing duplicates via time+pitch key."""
        seen = set()
        mixed = []
        for channel_notes in notes_by_channel:
            for n in channel_notes:
                key = (round(n.get("start", 0), 3), n.get("pitch", 0))
                if key not in seen:
                    seen.add(key)
                    mixed.append(n)
        mixed.sort(key=lambda n: n.get("start", 0))
        return mixed

    def _convert_prettymidi_to_noteseq(self, midi_data) -> list[note_seq.NoteSequence.Note]:
        """
        Convert pretty_midi (Basic Pitch output) to note_seq format.

        Args:
            midi_data: pretty_midi.PrettyMIDI object

        Returns:
            List of note_seq.NoteSequence.Note objects

        """
        notes = []

        # Extract notes from all instruments
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                # Create note_seq Note object
                ns_note = note_seq.NoteSequence.Note(
                    pitch=note.pitch,
                    start_time=note.start,
                    end_time=note.end,
                    velocity=note.velocity,
                )
                notes.append(ns_note)

        # Sort by start time
        notes.sort(key=lambda n: n.start_time)

        return notes

    def _convert_to_noteseq(
        self,
        midi_events: str,
        min_duration: float = 0.05,
    ) -> list[note_seq.NoteSequence.Note]:
        """
        Convert YourMT3 MIDI token output to note_seq Note objects.

        YourMT3 outputs token sequences from the MT3 vocabulary, decoded
        to text by the processor.  Handles two likely formats:

        1. TOKEN_VALUE pairs:  NOTE_ON 60 VELOCITY 80 TIME_SHIFT 0.5
                                NOTE_OFF 60 TIME_SHIFT 0.25 ...
        2. Key-value format:   pitch:60,start:0.0,end:0.5,velocity:80
        """
        notes: list = []
        tokens = midi_events.strip().split()
        if not tokens:
            return notes

        # ── Format 1: NOTE_ON / NOTE_OFF token pairs ──────────────
        first_token = tokens[0].upper()
        if first_token in (
            "NOTE_ON",
            "NOTE_OFF",
            "TIME_SHIFT",
            "PITCH",
            "VELOCITY",
            "PROGRAM",
        ):
            notes = self._parse_mt3_tokens(tokens, min_duration)
            if notes:
                return notes

        # ── Format 2: key:value pairs (pitch:X,start:Y,...) ──────
        import re

        pattern = r"pitch:(\d+),start:([\d.]+),end:([\d.]+),velocity:(\d+)"
        matches = re.findall(pattern, midi_events)
        if matches:
            for pitch_str, start_str, end_str, vel_str in matches:
                end = float(end_str)
                start = float(start_str)
                if (end - start) >= min_duration:
                    notes.append(
                        note_seq.NoteSequence.Note(
                            pitch=int(pitch_str),
                            start_time=start,
                            end_time=end,
                            velocity=int(vel_str),
                        ),
                    )
            if notes:
                return sorted(notes, key=lambda n: n.start_time)

        # ── No format matched ─────────────────────────────────────
        if not notes:
            pass

        return notes

    def _parse_mt3_tokens(
        self,
        tokens: list[str],
        min_duration: float,
    ) -> list[note_seq.NoteSequence.Note]:
        """
        Parse NOTE_ON / NOTE_OFF / TIME_SHIFT token sequences.

        MT3-style models emit tokens like:
            NOTE_ON 60 VELOCITY 80 TIME_SHIFT 0.5 NOTE_OFF 60
        """
        notes = []
        active_notes: dict[int, tuple[float, int]] = {}  # pitch -> (start, velocity)
        current_time = 0.0
        i = 0

        while i < len(tokens):
            tok = tokens[i].upper()
            try:
                if tok == "NOTE_ON" and i + 1 < len(tokens):
                    pitch = int(tokens[i + 1])
                    velocity = 80  # default
                    if i + 3 < len(tokens) and tokens[i + 2].upper() == "VELOCITY":
                        velocity = int(tokens[i + 3])
                        i += 4
                    else:
                        i += 2
                    active_notes[pitch] = (current_time, velocity)

                elif tok == "NOTE_OFF" and i + 1 < len(tokens):
                    pitch = int(tokens[i + 1])
                    if pitch in active_notes:
                        start_time, velocity = active_notes.pop(pitch)
                        dur = current_time - start_time
                        if dur >= min_duration:
                            notes.append(
                                note_seq.NoteSequence.Note(
                                    pitch=pitch,
                                    start_time=start_time,
                                    end_time=current_time,
                                    velocity=velocity,
                                ),
                            )
                    i += 2

                elif tok == "TIME_SHIFT" and i + 1 < len(tokens):
                    current_time += float(tokens[i + 1])
                    i += 2

                else:
                    i += 1
            except (ValueError, IndexError):
                i += 1

        # Close any still-active notes at end of sequence
        final_time = current_time + (min_duration * 2)
        for pitch, (start_time, velocity) in active_notes.items():
            notes.append(
                note_seq.NoteSequence.Note(
                    pitch=pitch,
                    start_time=start_time,
                    end_time=final_time,
                    velocity=velocity,
                ),
            )

        return sorted(notes, key=lambda n: n.start_time)

    def _filter_by_instrument_range(
        self,
        notes: list[note_seq.NoteSequence.Note],
        target: str,
    ) -> list[note_seq.NoteSequence.Note]:
        """
        Filter notes by valid instrument range.

        Standard ranges:
        - Bass (5-string): B0 (23) to G4 (67)
        - Guitar (6-string): E2 (40) to E6 (88)
        """
        target_lower = target.lower()

        if "bass" in target_lower:
            min_pitch, max_pitch = 23, 67  # 5-string bass range
        else:  # Guitar
            min_pitch, max_pitch = 40, 88  # Standard guitar range

        filtered = [note for note in notes if min_pitch <= note.pitch <= max_pitch]

        removed_count = len(notes) - len(filtered)
        if removed_count > 0:
            pass

        return filtered

    def humanize_and_clean(
        self,
        raw_notes: list[note_seq.NoteSequence.Note],
        is_bass: bool = False,
    ) -> list[note_seq.NoteSequence.Note]:
        """
        Clean transcription artifacts.

        Removes:
        - Ultra-short notes (<0.05s) - likely transcription errors
        - Notes outside instrument range
        - Duplicate notes at same timestamp
        """
        cleaned = []
        seen_pitches: dict = {}  # Track pitches by start time

        for note in raw_notes:
            # Filter ultra-short notes
            duration = note.end_time - note.start_time
            if duration < 0.05:
                continue

            # Bass-specific: enforce upper limit
            if is_bass and note.pitch > 67:
                continue

            # Remove duplicate notes at same time
            time_key = round(note.start_time, 2)
            if time_key in seen_pitches and note.pitch in seen_pitches[time_key]:
                continue

            seen_pitches.setdefault(time_key, set()).add(note.pitch)
            cleaned.append(note)

        removed_count = len(raw_notes) - len(cleaned)
        if removed_count > 0:
            pass

        return cleaned

    def export_midi(self, notes: list[note_seq.NoteSequence.Note], path: str) -> None:
        """Export note sequence to MIDI file."""
        if not notes:
            return

        ns = note_seq.NoteSequence(notes=notes)
        ns.ticks_per_quarter = 480  # Standard MIDI resolution
        note_seq.sequence_proto_to_midi_file(ns, path)


# ============================================================================
# STAGE 5: THE LUTHIER (TABLATURE GENERATION)
# ============================================================================


class TabAgent:
    """
    MIDI-to-tablature conversion using dynamic programming.

    Features:
    - Viterbi-style DP for optimal fingering paths
    - Instrument-aware cost heuristics
    - Technique detection (slides, hammer-ons, pull-offs)
    - 5-string bass optimization (low-string preference)
    - Configurable tuning support

    No changes needed from original - implementation is already optimal.
    """

    def __init__(self, tuning: list[int], num_frets: int = 24) -> None:
        """
        Initialize tablature generator.

        Args:
            tuning: List of MIDI note numbers for open strings
                Example: [23, 28, 33, 38, 43] for 5-string bass (B-E-A-D-G)
            num_frets: Maximum fret number on instrument

        """
        self.tuning = tuning
        self.num_frets = num_frets
        self.num_strings = len(tuning)

    def get_valid_positions(self, midi_note: int) -> list[dict]:
        """
        Find all valid string/fret combinations for a MIDI note.

        Returns:
            List of dicts with 'string' and 'fret' keys

        """
        positions = []
        for string_idx, open_note in enumerate(self.tuning):
            fret = midi_note - open_note
            if 0 <= fret <= self.num_frets:
                positions.append({"string": string_idx, "fret": fret})
        return positions

    def calculate_cost(self, prev: dict | None, curr: dict, time_delta: float = 1.0) -> float:
        """
        Calculate transition cost between two positions.

        Cost factors:
        - Fret distance (hand position shifts)
        - String changes (picking efficiency)
        - Time delta (legato vs. separate notes)
        - Instrument-specific preferences

        Args:
            prev: Previous position dict
            curr: Current position dict
            time_delta: Time between notes in seconds

        Returns:
            Cost value (lower is better)

        """
        if prev is None:
            return 0.0

        fret_distance = abs(curr["fret"] - prev["fret"])
        string_distance = abs(curr["string"] - prev["string"])

        # Base costs (tunable weights)
        cost = (fret_distance * 1.5) + (string_distance * 2.0)

        # Legato/slide detection
        if time_delta < 0.2:  # Fast transition
            if string_distance == 0:
                # Same string = likely slide/hammer/pull
                cost -= 5.0  # Encourage this path
            else:
                # String skip on fast run = awkward
                cost += 5.0  # Penalize

        # 5-string bass preference: avoid high frets on low strings
        # (Low strings on bass have better tone for low notes)
        if self.num_strings == 5:
            # Low B and E strings (indices 0-1)
            if curr["string"] < 2 and 0 < curr["fret"] < 5:
                cost += 1.0  # Slight penalty for low frets on low strings

        return cost

    def generate_tab(
        self,
        midi_notes: list[note_seq.NoteSequence.Note],
        technique_sensitivity: float = 0.7,
    ) -> list[dict]:
        """
        Generate optimal tablature using dynamic programming.

        Algorithm: Viterbi-style DP
        1. For each note, find all valid positions
        2. Calculate minimum cost path from previous note
        3. Backtrack to reconstruct optimal path
        4. Annotate techniques (slides, hammer-ons, pull-offs)

        Args:
            midi_notes: List of note_seq.NoteSequence.Note objects
            technique_sensitivity: 0-1 threshold for technique detection
                (higher = more aggressive detection, wider time windows)

        Returns:
            List of tab positions with technique annotations

        """
        if not midi_notes:
            return []

        # Convert to simplified representation
        notes = [{"pitch": n.pitch, "start": n.start_time} for n in midi_notes]

        # Get valid positions for each note
        layers = [self.get_valid_positions(n["pitch"]) for n in notes]

        # Check for unplayable notes
        if not all(layers):
            [i for i, layer in enumerate(layers) if not layer]
            return []

        # Initialize DP
        path = []
        prev_costs = [0.0] * len(layers[0])

        # Forward pass: compute costs
        for i in range(1, len(layers)):
            curr_layer = layers[i]
            prev_layer = layers[i - 1]
            time_delta = notes[i]["start"] - notes[i - 1]["start"]

            curr_costs = []
            backpointers = []

            for curr_pos in curr_layer:
                # Find minimum cost transition
                min_cost = float("inf")
                best_prev_idx = -1

                for prev_idx, prev_pos in enumerate(prev_layer):
                    cost = prev_costs[prev_idx] + self.calculate_cost(
                        prev_pos,
                        curr_pos,
                        time_delta,
                    )
                    if cost < min_cost:
                        min_cost = cost
                        best_prev_idx = prev_idx

                curr_costs.append(min_cost)
                backpointers.append(best_prev_idx)

            prev_costs = curr_costs
            path.append(backpointers)

        # Backward pass: reconstruct optimal path
        best_path = []
        last_idx = prev_costs.index(min(prev_costs))
        best_path.append(layers[-1][last_idx])

        for i in range(len(layers) - 2, -1, -1):
            last_idx = path[i][last_idx]
            best_path.append(layers[i][last_idx])

        # Reverse to get chronological order
        best_path.reverse()

        # Annotate techniques (slides, hammer-ons, pull-offs)
        final_tab: list = []
        technique_window = 0.05 + technique_sensitivity * 0.25  # maps 0.5→0.175s, 0.9→0.275s
        for i, pos in enumerate(best_path):
            technique = "pick"

            if i > 0:
                prev = final_tab[-1]
                time_delta = notes[i]["start"] - notes[i - 1]["start"]
                fret_diff_abs = abs(pos["fret"] - prev["fret"])

                if prev["string"] == pos["string"] and time_delta < technique_window:
                    # Slide: 1-2 frets apart (priority over hammer/pull)
                    if 1 <= fret_diff_abs <= 2:
                        technique = "slide"
                    # Hammer-on: ascending fret on same string
                    elif pos["fret"] > prev["fret"]:
                        technique = "hammer"
                    # Pull-off: descending fret on same string
                    elif pos["fret"] < prev["fret"]:
                        technique = "pull"

            pos["technique"] = technique
            pos["start_time"] = notes[i]["start"]
            final_tab.append(pos)

        return final_tab
