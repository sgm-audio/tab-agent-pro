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

import itertools
import math
import os
import shutil
import subprocess  # nosec B404 — fixed command list, shell=False, used by Demucs CLI fallback
import tempfile

import librosa
import numpy as np
import soundfile as sf
import torch  # noqa: F401 — retained for callers that patch agents.torch
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
health.set_component("basic_pitch", "available" if BASIC_PITCH_AVAILABLE else "unavailable")
health.set_component("demucs", "available" if DEMUCS_API_AVAILABLE else "unavailable")
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
        self.separation_mode = "unknown"
        os.makedirs(output_dir, exist_ok=True)
        self.log.info(
            "splitter_init",
            device=self.device,
            backend=self.backend,
        )

    def separate_stems(self, audio_path):
        """Separate audio into guitar and bass stems with graceful fallback."""
        file = os.path.basename(audio_path)
        self.separation_mode = "unknown"
        self.log.info("demucs_start", file=file, device=self.device)

        if DEMUCS_API_AVAILABLE:
            try:
                with metrics.track_stage("demucs_separation"):
                    result = self._separate_with_api(audio_path, file)
                    self.separation_mode = "api"
                    health.set_component("demucs", "loaded")
                    return result
            except Exception as e:
                self.log.warning("demucs_api_failed", file=file, error=str(e))
                health.set_component("demucs", "degraded (API failed)")

        # Try CLI fallback
        try:
            with metrics.track_stage("demucs_cli"):
                result = self._separate_with_subprocess(audio_path, file)
                self.separation_mode = "cli"
                health.set_component("demucs", "loaded (CLI)")
                return result
        except Exception as e:
            self.log.warning("demucs_all_failed", file=file, error=str(e))
            health.set_component("demucs", "unavailable")
            result = self._raw_audio_fallback(audio_path, file)
            self.separation_mode = "passthrough"
            return result

    def _raw_audio_fallback(self, audio_path, song_name):
        """Return raw audio as both guitar and bass stems (graceful degradation)."""
        self.log.info("demucs_raw_fallback", file=song_name)
        base_path = os.path.join(self.output_dir, "htdemucs", song_name)
        os.makedirs(base_path, exist_ok=True)

        other_path = os.path.join(base_path, "other.wav")
        bass_path = os.path.join(base_path, "bass.wav")
        shutil.copy2(audio_path, other_path)
        shutil.copy2(audio_path, bass_path)
        return {"guitar": other_path, "bass": bass_path}

    @staticmethod
    def _validate_stems(paths: dict[str, str]) -> None:
        """Reject missing, corrupt, empty, inconsistent or non-finite audio."""
        expected = None
        for path in paths.values():
            try:
                info = sf.info(path)
                if info.frames <= 0 or info.samplerate <= 0 or info.channels not in (1, 2):
                    raise ValueError(f"Invalid stem format: {path}")
                shape = (info.frames, info.samplerate, info.channels)
                if expected is not None and shape != expected:
                    raise ValueError("Demucs stems have inconsistent length or format")
                expected = shape
                for block in sf.blocks(path, blocksize=65536, dtype="float32"):
                    if not np.isfinite(block).all():
                        raise ValueError(f"Non-finite samples in stem: {path}")
            except (OSError, RuntimeError) as exc:
                raise ValueError(f"Demucs stem is missing or unreadable: {path}") from exc

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

        missing = {"other", "bass"} - set(separated)
        if missing:
            raise ValueError(f"Demucs API missing required stems: {sorted(missing)}")

        # Use a fresh staging directory so older files cannot fill missing stems.
        base_path = tempfile.mkdtemp(prefix="demucs-api-", dir=self.output_dir)
        paths = {
            "guitar": os.path.join(base_path, "other.wav"),
            "bass": os.path.join(base_path, "bass.wav"),
        }
        try:
            for key, target in (("other", "guitar"), ("bass", "bass")):
                tensor = separated[key]
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                data = tensor.cpu().numpy()
                if (
                    data.ndim != 2
                    or data.shape[0] not in (1, 2)
                    or data.shape[1] == 0
                    or not np.isfinite(data).all()
                ):
                    raise ValueError(f"Invalid Demucs API stem: {key}")
                sf.write(paths[target], data.T, separator.samplerate)
            self._validate_stems(paths)
            return paths
        except Exception:
            shutil.rmtree(base_path, ignore_errors=True)
            raise

    def _separate_with_subprocess(self, audio_path, song_name):
        """Fallback: Demucs via CLI subprocess."""
        staging = tempfile.mkdtemp(prefix="demucs-cli-", dir=self.output_dir)
        cmd = ["demucs", "-n", "htdemucs", "-o", staging, audio_path]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )  # nosec B603

            # Demucs strips the source extension when naming its output folder.
            base_path = os.path.join(staging, "htdemucs", os.path.splitext(song_name)[0])
            paths = {
                "guitar": os.path.join(base_path, "other.wav"),
                "bass": os.path.join(base_path, "bass.wav"),
            }
            self._validate_stems(paths)
            return paths
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

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
        min_pitch: int | None = None,
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
        min_pitch: int | None = None,
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
        min_pitch: int | None = None,
    ) -> list:
        """
        Filter notes by valid instrument range.

        Standard ranges:
        - Bass (5-string): B0 (23) to G4 (67)
        - Guitar (6-string): E2 (40) to E6 (88), or lower when drop-tuned
          (e.g. Drop D passes min_pitch=38)
        """

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


TabState = tuple[tuple[int, int] | None, tuple[tuple[int, int, int], ...]]


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
        self.last_unplayable_chords: list[dict] = []

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

    def _record_tab_conflict(self, notes: list[dict], group: list[int], reason: str) -> list[dict]:
        """Record a conflict and refuse to emit an incomplete or unplayable tab."""
        start = notes[group[0]]["start"]
        overlapping = [n["pitch"] for n in notes[: group[0]] if n["end"] > start]
        conflict = {
            "start_time": start,
            "pitches": overlapping + [notes[i]["pitch"] for i in group],
            "reason": reason,
        }
        self.last_unplayable_chords.append(conflict)
        get_logger("tab_agent").warning("unplayable_tab", **conflict)
        return []

    def generate_tab(
        self,
        midi_notes: list,
        technique_sensitivity: float = 0.7,
    ) -> list[dict]:
        """Assign notes while retaining string occupancy for their full duration.

        Simultaneous or near-simultaneous onsets (less than 50 ms apart) use
        distinct strings and are rendered in one chord column. Active fretted
        notes must fit within a conservative four-fret span; open strings do
        not count toward that span. Impossible conflicts return no tab and are
        recorded in ``last_unplayable_chords``. Search limits also fail closed.

        ``technique_sensitivity`` remains accepted for API compatibility, but
        note events alone cannot establish slide/hammer/pull technique. The
        default ``pick`` value therefore means no technique evidence, not a
        confirmed picked attack. Individually out-of-range notes retain the
        legacy skip-and-report behavior in ``last_skipped_notes``.
        """
        self.last_skipped_notes = []
        self.last_unplayable_chords = []
        if not midi_notes:
            return []

        notes: list[dict] = [
            {"pitch": int(n.pitch), "start": float(n.start_time), "end": float(n.end_time)}
            for n in midi_notes
        ]
        for note in notes:
            if (
                not math.isfinite(note["start"])
                or not math.isfinite(note["end"])
                or note["start"] < 0
                or note["end"] <= note["start"]
            ):
                self.last_unplayable_chords.append(
                    {
                        "start_time": note["start"],
                        "pitches": [note["pitch"]],
                        "reason": "invalid note timing",
                    }
                )
                get_logger("tab_agent").warning(
                    "invalid_note_timing", **self.last_unplayable_chords[-1]
                )
                return []

        positions_by_note = [(note, self.get_valid_positions(note["pitch"])) for note in notes]
        self.last_skipped_notes = [
            note["pitch"] for note, positions in positions_by_note if not positions
        ]
        playable = [(note, positions) for note, positions in positions_by_note if positions]
        if self.last_skipped_notes:
            get_logger("tab_agent").warning(
                "unplayable_notes_skipped",
                pitches=sorted(set(self.last_skipped_notes)),
                skipped=len(self.last_skipped_notes),
                kept=len(playable),
            )
        if not playable:
            return []

        playable.sort(key=lambda item: item[0]["start"])
        notes = [note for note, _ in playable]
        layers = [positions for _, positions in playable]
        groups: list[list[int]] = []
        for i, note in enumerate(notes):
            if groups and note["start"] - notes[groups[-1][0]]["start"] < 0.05:
                groups[-1].append(i)
            else:
                groups.append([i])

        # A state keeps all still-sounding (note index, string, fret) assignments
        # and the last position, which is needed by the transition cost. States
        # with identical future-relevant data can be safely merged.
        states: dict[TabState, tuple[float, int]] = {(None, ()): (0.0, 0)}
        nodes: list[tuple[int, tuple[tuple[int, int, int], ...]]] = [(-1, ())]
        max_states, max_transitions = 10000, 250000
        for group in groups:
            if len(group) > self.num_strings:
                return self._record_tab_conflict(notes, group, "too many simultaneous notes")
            options = []
            for positions in itertools.product(*(layers[i] for i in group)):
                if len({p["string"] for p in positions}) != len(positions):
                    continue
                frets = [p["fret"] for p in positions if p["fret"] > 0]
                if frets and max(frets) - min(frets) > 4:
                    continue
                options.append(positions)
            if not options:
                return self._record_tab_conflict(notes, group, "no playable chord assignment")

            next_states: dict[TabState, tuple[float, int]] = {}
            transitions = 0
            start = notes[group[0]]["start"]
            end_of_group = notes[group[-1]]["start"]
            for (last, active), (cost, parent) in states.items():
                active = tuple(item for item in active if notes[item[0]]["end"] > start)
                for positions in options:
                    transitions += 1
                    if transitions > max_transitions:
                        return self._record_tab_conflict(
                            notes, group, "assignment search limit exceeded"
                        )
                    valid = True
                    for j, index in enumerate(group):
                        now = notes[index]["start"]
                        sounding = [(s, f) for idx, s, f in active if notes[idx]["end"] > now]
                        sounding.extend(
                            (positions[k]["string"], positions[k]["fret"])
                            for k in range(j)
                            if notes[group[k]]["end"] > now
                        )
                        pos = positions[j]
                        if any(s == pos["string"] for s, _ in sounding):
                            valid = False
                            break
                        frets = [f for _, f in sounding if f > 0]
                        if pos["fret"] > 0:
                            frets.append(pos["fret"])
                        if frets and max(frets) - min(frets) > 4:
                            valid = False
                            break
                    if not valid:
                        continue
                    held = tuple(item for item in active if notes[item[0]]["end"] > end_of_group)
                    held += tuple(
                        (index, pos["string"], pos["fret"])
                        for index, pos in zip(group, positions, strict=True)
                        if notes[index]["end"] > end_of_group
                    )
                    last_position = positions[-1]
                    key: TabState = ((last_position["string"], last_position["fret"]), held)
                    elapsed = start - notes[group[0] - 1]["start"] if group[0] else 0.0
                    previous = {"string": last[0], "fret": last[1]} if last else None
                    new_cost = cost + self.calculate_cost(previous, positions[0], elapsed)
                    if key not in next_states or new_cost < next_states[key][0]:
                        assignment = tuple(
                            (index, pos["string"], pos["fret"])
                            for index, pos in zip(group, positions, strict=True)
                        )
                        nodes.append((parent, assignment))
                        next_states[key] = (new_cost, len(nodes) - 1)
            if not next_states:
                return self._record_tab_conflict(
                    notes, group, "overlapping notes cannot be assigned"
                )
            if len(next_states) > max_states:
                return self._record_tab_conflict(notes, group, "assignment search limit exceeded")
            states = next_states

        _, node = min(states.values(), key=lambda value: value[0])
        chosen: list[tuple[int, int, int]] = []
        while node:
            node, assignment = nodes[node]
            chosen.extend(assignment)
        chosen.sort(key=lambda item: item[0])
        return [
            {
                "string": string,
                "fret": fret,
                "technique": "pick",
                "start_time": notes[index]["start"],
            }
            for index, string, fret in chosen
        ]
