"""
Tab Agent — transcription pipeline.

Multi-instrument music transcription using Spotify's Basic Pitch (ONNX) for
audio-to-MIDI and Demucs for stem separation. Everything runs locally on the
end user's machine — there is no cloud inference.

Pipeline
--------
1. SplitterAgent — Demucs stem separation (vocals / drums / bass / other).
2. EarAgent      — Basic Pitch audio-to-MIDI transcription.
3. TabAgent      — Dynamic-programming MIDI-to-tablature conversion.

The transcription stack runs on CPU out of the box. GPU acceleration is
available through :mod:`device` (CUDA, ROCm, Vulkan, MPS, DirectML).
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess  # nosec B404 — fixed command list, shell=False, used by Demucs CLI fallback
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

from device import DeviceConfig, install_onnx_provider_override, resolve_device
from monitoring import default_metrics as metrics
from monitoring import get_logger, health

# Basic Pitch imports (proven and reliable transcription backend)
basic_pitch_predict = None
BASIC_PITCH_MODEL: Path | None = None
try:
    from basic_pitch import ONNX_PRESENT, FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import predict as _bp_predict

    basic_pitch_predict = _bp_predict
    BASIC_PITCH_MODEL = build_icassp_2022_model_path(FilenameSuffix.onnx)
    BASIC_PITCH_AVAILABLE = ONNX_PRESENT
except ImportError:
    BASIC_PITCH_AVAILABLE = False

# Demucs Python API (preferred over subprocess CLI)
try:
    import demucs.api as demucs_api

    DEMUCS_API_AVAILABLE = True
except ImportError:
    DEMUCS_API_AVAILABLE = False

# Publish backend availability at import time so /health and /ready are
# meaningful before the first request.
health.set_component("basic_pitch", "available" if BASIC_PITCH_AVAILABLE else "unavailable")
health.set_component("demucs", "available" if DEMUCS_API_AVAILABLE else "unavailable")
health.set_status("up")


def _apply_cache_dir() -> str | None:
    """Honour the documented CACHE_DIR env var for model/weight caches.

    Demucs and Basic Pitch download model weights on first use; both consult
    environment variables for their cache location. When ``CACHE_DIR`` is set
    (via .env or the real environment) we point those caches at it so the
    documented variable actually controls where weights are stored. Returns the
    resolved cache directory, or ``None`` when unset.
    """
    cache_dir = os.environ.get("CACHE_DIR")
    if not cache_dir:
        return None
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    # HuggingFace hub (Basic Pitch ONNX model) and XDG-aware tools.
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
    # Demucs stores downloaded checkpoints under a per-user dir.
    os.environ.setdefault("DEMUCS_DIR", cache_dir)
    return cache_dir


# ============================================================================
# STAGE 1-3: THE SPLITTER (Demucs stem separation)
# ============================================================================


class SplitterAgent:
    """
    Audio stem separation using Demucs.

    Demucs v4 (htdemucs) splits a mix into vocals / drums / bass / other.
    ``other`` is treated as the guitar stem; ``bass`` as the bass stem.

    Uses the Demucs Python API (``demucs.api.Separator``) — no subprocess CLI
    dependency. Graceful fallback: if separation fails entirely, raw audio is
    returned as a single stem so the pipeline doesn't crash.
    """

    def __init__(self, output_dir="separated_stems", device: str | None = None) -> None:
        self.output_dir = output_dir
        config: DeviceConfig = resolve_device(device)
        self.device: str = config.torch_device
        self.backend: str = config.backend
        self.label: str = config.label
        self.log = get_logger("splitter")
        self.separation_mode: str | None = None
        os.makedirs(output_dir, exist_ok=True)
        _apply_cache_dir()
        self.log.info(
            "splitter_init",
            device=self.device,
            backend=self.backend,
        )

    def separate_stems(self, audio_path):
        """Separate audio into guitar and bass stems with graceful fallback."""
        file = os.path.basename(audio_path)
        self.separation_mode = None
        self.log.info("demucs_start", file=file, device=self.device)

        if DEMUCS_API_AVAILABLE:
            try:
                with metrics.track_stage("demucs_separation"):
                    result = self._separate_with_api(audio_path, file)
                    health.set_component("demucs", "loaded")
                    self.separation_mode = "demucs"
                    return result
            except Exception as e:
                self.log.warning("demucs_api_failed", file=file, error=str(e))
                health.set_component("demucs", "degraded (API failed)")

        # Try CLI fallback
        try:
            with metrics.track_stage("demucs_cli"):
                result = self._separate_with_subprocess(audio_path, file)
                health.set_component("demucs", "loaded (CLI)")
                self.separation_mode = "demucs"
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
        other_path = os.path.join(base_path, "other.wav")
        bass_path = os.path.join(base_path, "bass.wav")
        shutil.copy2(audio_path, other_path)
        shutil.copy2(audio_path, bass_path)
        self.separation_mode = "raw_fallback"
        return {"guitar": other_path, "bass": bass_path}

    def _separate_with_api(self, audio_path, song_name):
        """Use Demucs Python API for stem separation."""
        separator = demucs_api.Separator(
            model="htdemucs",
            device=self.device,
            shifts=1,
            overlap=0.25,
            split=True,
            # Use the model's trained segment length (HTDemucs caps it below 10 s).
            progress=True,
        )

        _, separated = separator.separate_audio_file(audio_path)

        missing = {"other", "bass"} - separated.keys()
        if missing:
            raise ValueError(f"Demucs API did not return required stems: {sorted(missing)}")

        base_path = tempfile.mkdtemp(prefix="demucs-api-", dir=self.output_dir)
        try:
            for stem_key in ("other", "bass"):
                samples = separated[stem_key].cpu().numpy()
                if samples.ndim == 1:
                    samples = samples[np.newaxis, :]
                # Check before writing: PCM encoding can hide non-finite samples.
                if (
                    samples.ndim != 2
                    or samples.shape[0] not in (1, 2)
                    or samples.shape[1] == 0
                    or not np.isfinite(samples).all()
                ):
                    raise ValueError(f"Invalid Demucs API stem: {stem_key}")
                sf.write(
                    os.path.join(base_path, f"{stem_key}.wav"), samples.T, separator.samplerate
                )
            return self._checked_stems(base_path)
        except Exception:
            shutil.rmtree(base_path, ignore_errors=True)
            raise

    def _separate_with_subprocess(self, audio_path, song_name):
        """Fallback: Demucs via CLI subprocess."""
        # An exit code alone does not prove stems exist. Read only files
        # created in this attempt; a previous run may have the same basename.
        cli_dir = tempfile.mkdtemp(prefix="demucs-cli-", dir=self.output_dir)
        cmd = ["demucs", "-n", "htdemucs", "-d", self.device, "-o", cli_dir, audio_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)  # nosec B603
            stem_dir = os.path.splitext(song_name)[0]
            return self._checked_stems(os.path.join(cli_dir, "htdemucs", stem_dir))
        except Exception:
            shutil.rmtree(cli_dir, ignore_errors=True)
            raise

    @staticmethod
    def _checked_stems(base_path: str) -> dict[str, str]:
        stems = {
            "guitar": os.path.join(base_path, "other.wav"),
            "bass": os.path.join(base_path, "bass.wav"),
        }
        expected = None
        for path in stems.values():
            try:
                with sf.SoundFile(path) as stem:
                    shape = (stem.frames, stem.samplerate, stem.channels)
                    if stem.frames <= 0 or stem.samplerate <= 0 or stem.channels not in (1, 2):
                        raise ValueError(f"Invalid Demucs stem format: {path}")
                    if expected is not None and shape != expected:
                        raise ValueError("Demucs stems have inconsistent length or format")
                    expected = shape
                    frames = 0
                    for block in stem.blocks(blocksize=65536, dtype="float32"):
                        if not np.isfinite(block).all():
                            raise ValueError(f"Non-finite samples in Demucs stem: {path}")
                        frames += len(block)
                    if frames != stem.frames:
                        raise ValueError(f"Incomplete Demucs stem: {path}")
            except (OSError, RuntimeError) as exc:
                raise ValueError(f"Demucs stem is missing or unreadable: {path}") from exc
        return stems

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
        - Preserve low frequencies (fundamental tones below 200 Hz)
        - Reduce high frequencies (fret noise, harmonics above 200 Hz)
        - Uses 4th-order Butterworth low-pass and high-pass filters
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
# STAGE 4: THE EAR (Basic Pitch audio-to-MIDI transcription)
# ============================================================================


class EarAgent:
    """
    Audio-to-MIDI transcription using Spotify's Basic Pitch (ONNX).

    Basic Pitch is the sole transcription backend: lightweight, CPU-friendly,
    and GPU-acceleratable through :mod:`device` when the matching ONNX
    Runtime package is installed. It never calls out to a model hub.

    Primary model: Basic Pitch (Spotify, ONNX)
    """

    def __init__(self, device: str = "auto") -> None:
        """
        Initialize the transcription backend.

        Args:
            device: Compute device preference ("auto", "cpu", "cuda", "rocm",
                "vulkan", "mps", "directml"). See :func:`device.resolve_device`.
        """
        config: DeviceConfig = resolve_device(device)
        self.device: str = config.torch_device
        self.backend: str = config.backend
        self.label: str = config.label
        self.sample_rate = 16000
        self.log = get_logger("ear_agent")
        _apply_cache_dir()

        health.set_component(
            "basic_pitch",
            "available" if BASIC_PITCH_AVAILABLE else "unavailable",
        )

        # Upgrade Basic Pitch's hardcoded CPU-only ONNX provider to the
        # requested GPU provider when one is available.
        install_onnx_provider_override(config)

        if not BASIC_PITCH_AVAILABLE:
            self.log.warning(
                "no_transcription_backend",
                detail="Basic Pitch is not installed; transcribe_stem() will raise.",
            )

    def transcribe_stem(
        self,
        audio_path: str,
        target: str = "Guitar",
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        min_note_duration: float = 0.05,
        min_pitch: int | None = None,
        max_pitch: int | None = None,
    ) -> list:
        """
        Transcribe audio to MIDI notes.

        Falls back gracefully: Basic Pitch is tried, and if it is not installed
        a :class:`RuntimeError` is raised with a clear message.
        """
        log = self.log
        file = os.path.basename(audio_path)

        log.info("transcribe_start", file=file, target=target, device=self.device)

        # ── Try Basic Pitch ───────────────────────────────────────────
        failure: Exception | None = None
        if BASIC_PITCH_AVAILABLE:
            try:
                with metrics.track_stage("basic_pitch_transcription"):
                    result = self._transcribe_with_basic_pitch(
                        audio_path,
                        target,
                        onset_threshold,
                        frame_threshold,
                        min_note_duration,
                        min_pitch=min_pitch,
                        max_pitch=max_pitch,
                    )
                log.info(
                    "transcribe_done",
                    file=file,
                    model="basic_pitch",
                    device=self.device,
                    note_count=len(result),
                )
                return result
            except Exception as e:
                failure = e
                log.warning("basic_pitch_failed", file=file, error=str(e))

        log.error("transcribe_all_failed", file=file, target=target)
        health.record_request(success=False, details=f"transcribe: {file}")
        if isinstance(failure, ModuleNotFoundError):
            msg = f"Basic Pitch dependency missing: {failure.name}. Check the runtime installation."
        elif failure is not None:
            msg = (
                f"Basic Pitch could not transcribe {file}. Check the audio, ONNX model and runtime."
            )
        else:
            msg = (
                f"No transcription models available for {file}. "
                "Install Basic Pitch with ONNX Runtime and its dependencies (mir_eval, resampy)."
            )
        raise RuntimeError(msg) from failure

    def _transcribe_with_basic_pitch(
        self,
        audio_path: str,
        target: str,
        onset_threshold: float,
        frame_threshold: float,
        min_note_duration: float,
        min_pitch: int | None = None,
        max_pitch: int | None = None,
    ) -> list:
        """
        Transcribe using Basic Pitch (Spotify's proven ONNX model).
        """
        log = self.log

        if not BASIC_PITCH_AVAILABLE or basic_pitch_predict is None:
            raise RuntimeError("basic_pitch is not installed")

        try:
            _model_output, midi_data, _note_events = basic_pitch_predict(
                audio_path,
                model_or_model_path=BASIC_PITCH_MODEL,
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
            notes = self._filter_by_instrument_range(
                notes, target, min_pitch=min_pitch, max_pitch=max_pitch
            )

            health.set_component("basic_pitch", "loaded")
            log.info("basic_pitch_done", note_count=len(notes), target=target)
            return notes

        except ModuleNotFoundError as e:
            # e.g. mir_eval/resampy missing because basic-pitch was installed
            # with --no-deps. Report the missing module plainly.
            health.set_component("basic_pitch", f"error: missing {e.name}")
            log.error(
                "basic_pitch_dependency_missing",
                exc=e,
                file=os.path.basename(audio_path),
                detail="install basic-pitch with its dependencies (mir_eval, resampy)",
            )
            raise
        except Exception as e:
            health.set_component("basic_pitch", f"error: {type(e).__name__}")
            log.error("basic_pitch_error", exc=e, file=os.path.basename(audio_path))
            raise

    def _convert_prettymidi_to_noteseq(self, midi_data) -> list:
        """Convert pretty_midi (Basic Pitch output) to note_seq Note objects."""
        import note_seq

        notes = []

        for instrument in midi_data.instruments:
            for note in instrument.notes:
                ns_note = note_seq.NoteSequence.Note(
                    pitch=note.pitch,
                    start_time=note.start,
                    end_time=note.end,
                    velocity=note.velocity,
                )
                notes.append(ns_note)

        notes.sort(key=lambda n: n.start_time)
        return notes

    def _filter_by_instrument_range(
        self,
        notes: list,
        target: str,
        min_pitch: int | None = None,
        max_pitch: int | None = None,
    ) -> list:
        """
        Filter notes by valid instrument range.

        Standard ranges:
        - Bass (5-string): B0 (23) to G4 (67)
        - Guitar (6-string): E2 (40) to E6 (88)

        Callers can supply both bounds from the actual tuning and fret count.
        """
        target_lower = target.lower()

        if "bass" in target_lower:
            default_min, default_max = 23, 67  # 5-string bass range
        else:  # Guitar
            default_min, default_max = 40, 88  # Standard guitar range

        low_bound = default_min if min_pitch is None else min_pitch
        high_bound = default_max if max_pitch is None else max_pitch
        filtered = [note for note in notes if low_bound <= note.pitch <= high_bound]

        removed_count = len(notes) - len(filtered)
        if removed_count > 0:
            get_logger("ear_agent").info(
                "notes_filtered_out_of_range",
                target=target,
                removed=removed_count,
                kept=len(filtered),
            )

        return filtered

    def humanize_and_clean(
        self,
        raw_notes: list,
        is_bass: bool = False,
        max_pitch: int | None = None,
    ) -> list:
        """
        Clean transcription artifacts.

        Removes:
        - Invalid or ultra-short notes (<0.05s)
        - Notes above the requested range (default for bass: above G4)
        - Duplicate notes at the same timestamp
        """
        cleaned = []
        upper_bound = max_pitch if max_pitch is not None else (67 if is_bass else None)
        seen_pitches: dict = {}  # Track pitches by start time

        for note in raw_notes:
            # Invalid timing must not reach MIDI or tab export.
            start, end = note.start_time, note.end_time
            if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end - start < 0.05:
                continue

            if upper_bound is not None and note.pitch > upper_bound:
                continue

            # Remove duplicate notes at same time
            time_key = round(note.start_time, 2)
            if time_key in seen_pitches and note.pitch in seen_pitches[time_key]:
                continue

            seen_pitches.setdefault(time_key, set()).add(note.pitch)
            cleaned.append(note)

        removed_count = len(raw_notes) - len(cleaned)
        if removed_count > 0:
            get_logger("ear_agent").info(
                "notes_cleaned",
                removed=removed_count,
                kept=len(cleaned),
                is_bass=is_bass,
            )

        return cleaned

    def export_midi(self, notes: list, path: str) -> None:
        """Export note sequence to MIDI file."""
        if not notes:
            return

        import note_seq

        ns = note_seq.NoteSequence(notes=notes)
        ns.ticks_per_quarter = 480  # Standard MIDI resolution
        note_seq.sequence_proto_to_midi_file(ns, path)


# ============================================================================
# STAGE 5: THE LUTHIER (Tablature generation)
# ============================================================================

# Keep a high cost for a hypothetical same-string, same-onset transition.
# Actual collisions are rejected by the solver's duration-aware occupancy rule.
SAME_STRING_CHORD_PENALTY = 1000.0


@dataclass(frozen=True, order=True, slots=True)
class _TabNote:
    start: float
    pitch: int
    end: float


@dataclass(slots=True)
class _PlacedNote:
    index: int
    position: dict[str, int]
    previous: _PlacedNote | None


@dataclass(slots=True)
class _TabPath:
    occupied: tuple[int, ...]  # active note index per string; -1 means free
    tail: _PlacedNote | None
    omitted: int = 0
    cost: float = 0.0

    def rank(self) -> tuple[int, float]:
        return self.omitted, self.cost


class TabAgent:
    """Time-aware fingering search with hard string and fret-span constraints.

    A string can carry one note at a time. Overlapping fretted notes must fit
    within a conservative hand span (open strings don't occupy a finger). The
    search prioritizes keeping notes over minimizing movement and bounds its
    polyphonic frontier; it does not promise an optimal fingering or model
    every physical constraint. Omitted notes are reported to callers.
    """

    MAX_FRET_SPAN = 4
    SEARCH_WIDTH = 128

    def __init__(
        self,
        tuning: list[int],
        num_frets: int = 24,
        prefer_low_strings: bool = False,
    ) -> None:
        """
        Initialize tablature generator.

        Args:
            tuning: List of MIDI note numbers for open strings
                Example: [23, 28, 33, 38, 43] for 5-string bass (B-E-A-D-G)
            num_frets: Maximum fret number on instrument
            prefer_low_strings: Penalize high-fret positions on upper strings
        """
        self.tuning = tuning
        self.num_frets = num_frets
        self.num_strings = len(tuning)
        self.prefer_low_strings = prefer_low_strings
        self.last_skipped_notes: list[int] = []
        self.last_skipped_events: list[dict] = []

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
        - Time between onsets
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

        # Prefer economical moves in fast monophonic passages.
        if 0 <= time_delta < 0.2:  # Fast transition
            if string_distance == 0:
                cost -= 5.0
            else:
                # String skip on fast run = awkward
                cost += 5.0  # Penalize

        # Nearby onsets may be successive releases and attacks, not a chord.
        if time_delta == 0 and string_distance == 0:
            cost += SAME_STRING_CHORD_PENALTY

        # 5-string bass heuristic: slight penalty for low fretted notes (1-4)
        # on the lowest strings (B0/E1, indices 0-1)
        if self.num_strings == 5 and curr["string"] < 2 and 0 < curr["fret"] < 5:
            # Low B and E strings (indices 0-1)
            cost += 1.0  # Slight penalty for low frets on low strings

        # When prefer_low_strings is enabled, discourage high frets on upper strings
        if self.prefer_low_strings and curr["string"] >= 2 and curr["fret"] >= 10:
            cost += 1.0

        return cost

    @staticmethod
    def _polyphonic_notes(notes: list[_TabNote]) -> list[bool]:
        """Whether each event overlaps any other event (even one later omitted)."""
        latest_end = -math.inf
        flags = []
        for i, note in enumerate(notes):
            before = note.start < latest_end
            after = i + 1 < len(notes) and notes[i + 1].start < note.end
            flags.append(before or after)
            latest_end = max(latest_end, note.end)
        return flags

    @staticmethod
    def _keep_path(paths: dict[tuple, _TabPath], candidate: _TabPath) -> None:
        """Merge paths with the same held strings and last note/position."""
        tail = candidate.tail
        key = (
            (candidate.occupied, tail.index, tail.position["string"])
            if tail
            else (candidate.occupied, -1, -1)
        )
        current = paths.get(key)
        if current is None or candidate.rank() < current.rank():
            paths[key] = candidate

    def _assign_positions(
        self, notes: list[_TabNote], positions: list[list[dict]], polyphonic: list[bool]
    ) -> dict[int, dict[str, int]]:
        paths = [_TabPath((-1,) * self.num_strings, None)]
        for i, (note, choices) in enumerate(zip(notes, positions, strict=True)):
            next_paths: dict[tuple, _TabPath] = {}
            for path in paths:
                occupied = tuple(
                    k if k >= 0 and notes[k].end > note.start else -1 for k in path.occupied
                )
                # A skip is useful when retaining a note can block another one.
                # Disjoint notes collapse to the original per-position DP.
                if polyphonic[i] or not choices:
                    self._keep_path(
                        next_paths, _TabPath(occupied, path.tail, path.omitted + 1, path.cost)
                    )
                frets = [
                    notes[k].pitch - self.tuning[string]
                    for string, k in enumerate(occupied)
                    if k >= 0 and notes[k].pitch > self.tuning[string]
                ]
                low, high = (
                    (max(frets) - self.MAX_FRET_SPAN, min(frets) + self.MAX_FRET_SPAN)
                    if frets
                    else (0, self.num_frets)
                )
                prev = path.tail
                for position in choices:
                    string, fret = position["string"], position["fret"]
                    if occupied[string] >= 0 or (fret and not low <= fret <= high):
                        continue
                    cost = self.calculate_cost(
                        prev.position if prev else None,
                        position,
                        note.start - notes[prev.index].start if prev else 0.0,
                    )
                    self._keep_path(
                        next_paths,
                        _TabPath(
                            occupied[:string] + (i,) + occupied[string + 1 :],
                            _PlacedNote(i, position, prev),
                            path.omitted,
                            path.cost + cost,
                        ),
                    )
            paths = sorted(next_paths.values(), key=_TabPath.rank)[: self.SEARCH_WIDTH]

        placed = {}
        node = min(paths, key=_TabPath.rank).tail
        while node is not None:
            placed[node.index] = node.position
            node = node.previous
        return placed

    def generate_tab(self, midi_notes: list, technique_sensitivity: float = 0.7) -> list[dict]:
        """Assign valid events and report any omissions with their reasons.

        Notes must have finite, nonnegative onsets and positive durations. The
        intervals are half-open, so a note can begin exactly at another's release.
        The legacy ``technique_sensitivity`` argument is ignored: ``pick`` is
        an unmarked note, not evidence of a picked attack.
        """
        self.last_skipped_notes = []
        self.last_skipped_events = []
        notes = []
        for note in midi_notes:
            start = note.start_time
            end: float | None = getattr(note, "end_time", None)
            if end is not None and not math.isfinite(end):
                end = None
            finite_start = math.isfinite(start)
            if not finite_start or start < 0 or end is None or end <= start:
                self.last_skipped_notes.append(note.pitch)
                self.last_skipped_events.append(
                    {
                        "pitch": note.pitch,
                        "start_time": start if finite_start else None,
                        "end_time": end,
                        "reason": "invalid_timing",
                    }
                )
                continue
            notes.append(_TabNote(start, note.pitch, end))
        notes.sort()

        positions = [self.get_valid_positions(n.pitch) for n in notes]
        assigned = self._assign_positions(notes, positions, self._polyphonic_notes(notes))
        tab: list[dict] = []
        for i, note in enumerate(notes):
            if i not in assigned:
                self.last_skipped_notes.append(note.pitch)
                self.last_skipped_events.append(
                    {
                        "pitch": note.pitch,
                        "start_time": note.start,
                        "end_time": note.end,
                        "reason": "fingering_conflict" if positions[i] else "out_of_range",
                    }
                )
                continue
            tab.append(
                {
                    **assigned[i],
                    "start_time": note.start,
                    "end_time": note.end,
                    "technique": "pick",
                }
            )

        if self.last_skipped_notes:
            get_logger("tab_agent").warning(
                "unplayable_notes_skipped",
                pitches=sorted(set(self.last_skipped_notes)),
                skipped=len(self.last_skipped_notes),
                kept=len(tab),
            )
        return tab
