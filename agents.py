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

import os
import subprocess  # nosec B404 — fixed command list, shell=False, used by Demucs CLI fallback
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, filtfilt

from device import DeviceConfig, install_onnx_provider_override, resolve_device
from monitoring import default_metrics as metrics
from monitoring import get_logger, health

# Basic Pitch imports (proven and reliable transcription backend)
basic_pitch_predict = None
try:
    from basic_pitch.inference import predict as _bp_predict

    basic_pitch_predict = _bp_predict
    BASIC_PITCH_AVAILABLE = True
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
health.set_component(
    "basic_pitch", "available" if BASIC_PITCH_AVAILABLE else "unavailable"
)
health.set_component(
    "demucs", "available" if DEMUCS_API_AVAILABLE else "unavailable"
)
health.set_status("up")


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
        os.makedirs(output_dir, exist_ok=True)
        self.log.info(
            "splitter_init",
            device=self.device,
            backend=self.backend,
        )

    def separate_stems(self, audio_path):
        """Separate audio into guitar and bass stems with graceful fallback."""
        file = os.path.basename(audio_path)
        self.log.info("demucs_start", file=file, device=self.device)

        if DEMUCS_API_AVAILABLE:
            try:
                with metrics.track_stage("demucs_separation"):
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
            device=self.device,
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
            )  # nosec B603
        except subprocess.CalledProcessError:
            raise

        base_path = os.path.join(self.output_dir, "htdemucs", song_name)
        stem_dir = os.path.splitext(song_name)[0]
        alt_base = os.path.join(self.output_dir, "htdemucs", stem_dir)
        if not os.path.exists(base_path) and os.path.exists(alt_base):
            base_path = alt_base
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
        min_pitch: Optional[int] = None,
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
                log.warning("basic_pitch_failed", file=file, error=str(e))

        # ── Nothing worked: raise error ──────────────────────────────
        log.error("transcribe_all_failed", file=file, target=target)
        health.record_request(success=False, details=f"transcribe: {file}")
        msg = (
            f"No transcription models available for {file}. "
            "Install Basic Pitch (pip install basic-pitch) with its runtime "
            "dependencies (mir_eval, resampy)."
        )
        raise RuntimeError(msg)

    def _transcribe_with_basic_pitch(
        self,
        audio_path: str,
        target: str,
        onset_threshold: float,
        frame_threshold: float,
        min_note_duration: float,
        min_pitch: Optional[int] = None,
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
            notes = self._filter_by_instrument_range(notes, target, min_pitch=min_pitch)

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
        min_pitch: Optional[int] = None,
    ) -> list:
        """
        Filter notes by valid instrument range.

        Standard ranges:
        - Bass (5-string): B0 (23) to G4 (67)
        - Guitar (6-string): E2 (40) to E6 (88), or lower when drop-tuned
          (e.g. Drop D passes min_pitch=38)
        """
        import note_seq

        target_lower = target.lower()

        if "bass" in target_lower:
            default_min, max_pitch = 23, 67  # 5-string bass range
        else:  # Guitar
            default_min, max_pitch = 40, 88  # Standard guitar range

        low_bound = default_min if min_pitch is None else min_pitch
        filtered = [note for note in notes if low_bound <= note.pitch <= max_pitch]

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
    ) -> list:
        """
        Clean transcription artifacts.

        Removes:
        - Ultra-short notes (<0.05s) - likely transcription errors
        - Notes outside instrument range (bass > G4)
        - Duplicate notes at the same timestamp
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


class TabAgent:
    """
    MIDI-to-tablature conversion using dynamic programming.

    Features:
    - Viterbi-style DP for optimal fingering paths
    - Instrument-aware cost heuristics
    - Technique detection (slides, hammer-ons, pull-offs)
    - 5-string bass optimization (low-string preference)
    - Configurable tuning support

    Notes outside the instrument's range are skipped (and reported in
    ``last_skipped_notes``) instead of discarding the whole track.
    """

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
        if 0 <= time_delta < 0.2:  # Fast transition
            if string_distance == 0:
                # Same string = likely slide/hammer/pull
                cost -= 5.0  # Encourage this path
            else:
                # String skip on fast run = awkward
                cost += 5.0  # Penalize

        # 5-string bass heuristic: slight penalty for low fretted notes (1-4)
        # on the lowest strings (B0/E1, indices 0-1)
        if self.num_strings == 5 and curr["string"] < 2 and 0 < curr["fret"] < 5:
            # Low B and E strings (indices 0-1)
            cost += 1.0  # Slight penalty for low frets on low strings

        # When prefer_low_strings is enabled, discourage high frets on upper strings
        if self.prefer_low_strings and curr["string"] >= 2 and curr["fret"] >= 10:
            cost += 1.0

        return cost

    def generate_tab(
        self,
        midi_notes: list,
        technique_sensitivity: float = 0.7,
    ) -> list[dict]:
        """
        Generate optimal tablature using dynamic programming.

        Algorithm: Viterbi-style DP
        1. For each note, find all valid positions
        2. Calculate minimum cost path from previous note
        3. Backtrack to reconstruct optimal path
        4. Annotate techniques (slides, hammer-ons, pull-offs)

        Notes with no valid string/fret position are skipped and listed in
        ``self.last_skipped_notes``; the remaining notes still produce tablature.
        Returns an empty list only when no note is playable.

        Args:
            midi_notes: List of note_seq.NoteSequence.Note objects
            technique_sensitivity: 0-1 threshold for technique detection
                (higher = more aggressive detection, wider time windows)

        Returns:
            List of tab positions with technique annotations

        """
        # Reflect this call even when there is nothing to do
        self.last_skipped_notes = []

        if not midi_notes:
            return []

        # Convert to simplified representation
        notes = [{"pitch": n.pitch, "start": n.start_time} for n in midi_notes]

        # Get valid positions for each note, skipping notes that cannot be played
        # on this instrument (previously one such note discarded the whole track).
        positions_by_note = [(note, self.get_valid_positions(note["pitch"])) for note in notes]
        playable = [(note, positions) for note, positions in positions_by_note if positions]

        self.last_skipped_notes = [
            int(note["pitch"]) for note, positions in positions_by_note if not positions
        ]
        if self.last_skipped_notes:
            get_logger("tab_agent").warning(
                "unplayable_notes_skipped",
                pitches=sorted(set(self.last_skipped_notes)),
                skipped=len(self.last_skipped_notes),
                kept=len(playable),
            )

        if not playable:
            return []

        notes = [note for note, _positions in playable]
        layers = [positions for _note, positions in playable]

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
            pos = dict(pos)
            technique = "pick"

            if i > 0:
                prev = final_tab[-1]
                time_delta = notes[i]["start"] - notes[i - 1]["start"]
                fret_diff_abs = abs(pos["fret"] - prev["fret"])

                if prev["string"] == pos["string"] and 0 <= time_delta < technique_window:
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
