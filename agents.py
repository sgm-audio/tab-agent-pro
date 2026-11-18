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

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

from device import DeviceConfig, install_onnx_provider_override, resolve_device
from monitoring import default_metrics as metrics
from monitoring import get_logger, health
from tab_validation import TabValidationReport, validate_tab

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


class TabAgent:
    """
    MIDI-to-tablature conversion, fingered one *column* at a time.

    A column is every note that starts within ``GROUP_TOLERANCE_S`` of the
    column's first note. Fingering a column means choosing exactly one
    string/fret position per pitch, with a hard constraint that no two notes in
    the column share a string — so a chord is fingered as a chord, not as an
    arpeggio.

    Features:
    - Viterbi DP over columns (one transition per column, not per note)
    - Physical constraints: one note per string, bounded hand span
    - Instrument-aware cost heuristics (open strings, low frets, position shifts)
    - Technique detection (slides, hammer-ons, pull-offs) between columns
    - Configurable tuning support

    Notes that cannot be fingered are skipped and reported in
    ``last_skipped_notes`` (pitches) and ``last_skipped_columns`` (per-column
    detail), instead of discarding the whole track.
    """

    # Notes starting within this window may belong to the same column (chord).
    # Basic Pitch's analysis hop is 256 samples / 22050 Hz ~= 11.6 ms, so notes
    # struck together share an onset frame while two onsets one frame apart are
    # two events. A cluster is only kept together if it can actually be fingered
    # as one (see _split_unfingerable), so this can never merge a played run.
    GROUP_TOLERANCE_S = 0.012
    # Widest gap, in frets, between fretted notes of one column. Anything wider
    # is not reachable and the offending notes are dropped, not printed.
    MAX_HAND_SPAN = 5
    # Columns further apart than this are not transitions: the hand moves
    # during the gap, so shift/legato costs must not apply.
    MAX_TRANSITION_S = 1.5
    # Cap on candidate fingerings per column, to keep wide chords tractable.
    MAX_CANDIDATES_PER_COLUMN = 24
    # Cost of moving the hand one fret between consecutive columns.
    POSITION_SHIFT_COST = 0.75
    # Extra cost per fret beyond a comfortable stretch.
    HAND_SPAN_COST = 4.0
    # A same-string fret change of at least this many frets is called a slide.
    # Below it, hammer-on/pull-off is the far more likely reading: a 1-2 fret
    # "slide" is indistinguishable from (and much rarer than) legato.
    SLIDE_MIN_FRETS = 3

    def __init__(
        self,
        tuning: list[int],
        num_frets: int = 24,
        prefer_low_strings: bool = False,
    ) -> None:
        """
        Initialize tablature generator.

        Args:
            tuning: List of MIDI note numbers for open strings, lowest first
                Example: [23, 28, 33, 38, 43] for 5-string bass (B-E-A-D-G)
            num_frets: Maximum fret number on instrument
            prefer_low_strings: Prefer low-position fingerings instead of
                reaching far up the neck
        """
        self.tuning = list(tuning)
        self.num_frets = num_frets
        self.num_strings = len(tuning)
        self.prefer_low_strings = prefer_low_strings
        self.last_skipped_notes: list[int] = []
        # Per-column detail for callers that report what was dropped:
        # [{"time": float, "dropped": [pitch, ...], "reason": str}, ...]
        self.last_skipped_columns: list[dict] = []
        # Populated by generate_tab(); see tab_validation.TabValidationReport.
        self.last_validation: TabValidationReport | None = None

    # ── positions and costs ──────────────────────────────────────────────

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

    def playability_reason(self, midi_note: int) -> str:
        """Explain why a pitch has no position on this instrument."""
        if self.num_strings and midi_note < self.tuning[0]:
            return f"pitch {midi_note} is below the lowest open string ({self.tuning[0]})"
        if self.num_strings and midi_note > self.tuning[-1] + self.num_frets:
            return (
                f"pitch {midi_note} is above the highest fret of the highest "
                f"string ({self.tuning[-1] + self.num_frets})"
            )
        return f"pitch {midi_note} has no string/fret position"

    def _low_string_penalty(self, position: dict) -> float:
        """
        Instrument-specific penalties for a single position.

        Shared by :meth:`_position_cost` (absolute cost) and
        :meth:`calculate_cost` (transition cost) so the rules live in one place.
        Each caller applies it exactly once, so no path double-counts it.
        """
        penalty = 0.0
        # The 1-4 fret stretch low on a 5-string bass is awkward.
        if self.num_strings == 5 and position["string"] < 2 and 0 < position["fret"] < 5:
            penalty += 1.0
        if self.prefer_low_strings and position["string"] >= 2 and position["fret"] >= 10:
            penalty += 1.0
        return penalty

    def _position_cost(self, position: dict) -> float:
        """
        Cost of *being* at a position, independent of what came before.

        Absolute fret matters: the 24th fret of the low string is technically
        reachable, but nobody plays there when the same note is available open
        on a higher string. Open strings and low frets are cheap.
        """
        fret = position["fret"]
        cost = self._low_string_penalty(position)
        if fret == 0:
            return cost
        cost += fret * 0.35
        if fret > 12:
            cost += (fret - 12) * 1.2
        return cost

    def calculate_cost(self, prev: dict | None, curr: dict, time_delta: float = 1.0) -> float:
        """
        Calculate transition cost between two positions.

        Cost factors:
        - Fret distance (hand position shifts)
        - String changes (picking efficiency)
        - Time delta (legato vs. separate notes)
        - Instrument-specific preferences

        This is the cost of playing ``curr`` *after* ``prev``. It deliberately
        excludes any absolute-position cost (that lives in
        :meth:`_position_cost`), so the same position has zero transition cost.

        A non-positive ``time_delta`` means the two positions are simultaneous
        (a fingering change inside one column), not a musical transition, so it
        gets neither the legato discount nor the string-skip penalty.

        Args:
            prev: Previous position dict
            curr: Current position dict
            time_delta: Time between notes in seconds

        Returns:
            Cost value (lower is better)
        """
        cost = self._low_string_penalty(curr)
        if prev is None:
            return cost

        fret_distance = abs(curr["fret"] - prev["fret"])
        string_distance = abs(curr["string"] - prev["string"])

        # Base costs (tunable weights)
        cost += (fret_distance * 1.5) + (string_distance * 2.0)

        if 0 < time_delta < 0.2:  # Fast transition, actually sequential
            if string_distance == 0:
                # Same string = likely slide/hammer/pull
                cost -= 5.0  # Encourage this path
            else:
                # String skip on fast run = awkward
                cost += 5.0  # Penalize

        return cost

    @staticmethod
    def _barre_of(fingering: list[dict]) -> int:
        """Hand centre of a fingering: the middle of its fretted notes."""
        fretted = [p["fret"] for p in fingering if p["fret"] > 0]
        if not fretted:
            return 0
        return (min(fretted) + max(fretted)) // 2

    def _hand_span(self, fingering: list[dict]) -> int:
        """Gap between the lowest and highest fretted note (open strings exempt)."""
        fretted = [p["fret"] for p in fingering if p["fret"] > 0]
        if len(fretted) < 2:
            return 0
        return max(fretted) - min(fretted)

    def _fingering_cost(self, fingering: list[dict]) -> float:
        """Cost of a complete column fingering, including its hand span."""
        barre = self._barre_of(fingering)
        cost = 0.0
        for position in fingering:
            # Reaching away from the hand centre is uncomfortable.
            cost += self._position_cost(position) + abs(position["fret"] - barre) * 0.25
        over = max(0, self._hand_span(fingering) - self.MAX_HAND_SPAN)
        cost += over * self.HAND_SPAN_COST
        return cost

    def _column_cost(self, prev: dict | None, fingering: list[dict], time_delta: float) -> float:
        """
        Cost of moving from the previous column's fingering to this one.

        Single-note columns use the pairwise :meth:`calculate_cost`, which is
        what melodies are made of. Wider columns are costed by hand shift,
        because note-to-note distances are not meaningful when the notes are
        struck together.
        """
        if prev is None:
            return 0.0
        barre = self._barre_of(fingering)
        if len(prev) == 1 and len(fingering) == 1:
            # A melody is note-to-note. calculate_cost owns the low-string
            # penalty here, so strip it and let _fingering_cost apply it once.
            return self.calculate_cost(
                prev[0], fingering[0], time_delta
            ) - self._low_string_penalty(fingering[0])
        cost = 0.0
        if time_delta <= self.MAX_TRANSITION_S:
            cost += abs(barre - self._barre_of([prev[0]])) * self.POSITION_SHIFT_COST
        return cost

    # ── grouping ─────────────────────────────────────────────────────────

    def group_notes(self, midi_notes: list) -> list[dict]:
        """
        Group notes into columns of simultaneous onsets.

        A note joins the open column when it starts within
        ``GROUP_TOLERANCE_S`` of the column's *first* note. The comparison is
        always against the column start, never against the previous note, so a
        run of fast sequential notes cannot chain itself into one column.

        A cluster is only kept as one column if it is actually fingered as one:
        notes that would need the same string are split into their own columns
        in pitch order. Without this, a fast line played on one string would be
        reported as a chord (or silently lose notes) instead of as a run — the
        onset detector cannot tell "two notes at once" from "two notes 15 ms
        apart", but the instrument can.

        Returns:
            List of {"time": float, "notes": [...]} sorted by time.
        """
        columns: list[dict] = []
        for note in sorted(midi_notes, key=lambda n: float(n.start_time)):
            start = float(note.start_time)
            if columns and start - columns[-1]["time"] < self.GROUP_TOLERANCE_S:
                columns[-1]["notes"].append(note)
            else:
                columns.append({"time": start, "notes": [note]})

        sequential: list[dict] = []
        for column in columns:
            sequential.extend(self._split_unfingerable(column))
        return sequential

    def _split_unfingerable(self, column: dict) -> list[dict]:
        """
        Turn a cluster into columns that can each be fingered.

        Keeps the widest subset of the cluster that has a string-disjoint
        fingering and emits the rest as later columns, preserving pitch order.
        """
        # A pitch struck twice in one column is one note: a string cannot sound
        # it twice, and Basic Pitch/note_seq both emit duplicates for a doubled
        # take. Collapse them before deciding whether this is a chord.
        notes: list = []
        seen_pitches: set[int] = set()
        for note in sorted(column["notes"], key=lambda n: (int(n.pitch), float(n.start_time))):
            pitch = int(note.pitch)
            if pitch in seen_pitches:
                continue
            seen_pitches.add(pitch)
            notes.append(note)
        if len(notes) < 2:
            return [{"time": column["time"], "notes": notes}]
        if self._greedy_feasible([int(n.pitch) for n in notes]):
            return [{"time": column["time"], "notes": notes}]

        # Not fingerable as a chord: tile the cluster into the widest playable
        # subsets, consuming notes in *time* order. Time order matters: a fast
        # chromatic run carries its short notes only through their own columns.
        ordered = sorted(column["notes"], key=lambda n: (float(n.start_time), int(n.pitch)))
        groups: list[list] = []
        remaining = list(ordered)
        while remaining:
            subset: list = []
            used: set[int] = set()
            rest: list = []
            for note in remaining:
                pitch = int(note.pitch)
                free = [p for p in self.get_valid_positions(pitch) if p["string"] not in used]
                if free and self._greedy_feasible([int(n.pitch) for n in subset] + [pitch]):
                    used.add(min(free, key=lambda p: (p["fret"], p["string"]))["string"])
                    subset.append(note)
                else:
                    rest.append(note)
            if not subset:  # pragma: no cover - every pitch resolves to a string
                subset, rest = [remaining[0]], remaining[1:]
            groups.append(subset)
            remaining = rest
        return [{"time": column["time"], "notes": g} for g in groups if g]

    # ── column fingering ─────────────────────────────────────────────────

    def _greedy_feasible(self, pitches: list[int]) -> bool:
        """
        Cheap test: can these pitches take distinct strings?

        Used to decide whether an onset cluster really is one chord before it is
        kept as one column. Pitches most constrained first, which is the usual
        scheduling heuristic; an exact search is only needed at fingering time.
        Pitches with no position at all are ignored here because
        :meth:`finger_column` reports them as dropped.
        """
        options = []
        for pitch in pitches:
            positions = self.get_valid_positions(int(pitch))
            if positions:
                options.append(sorted(positions, key=lambda p: (p["fret"], p["string"])))
        options.sort(key=len)
        used: set[int] = set()
        for candidate_positions in options:
            free = [p for p in candidate_positions if p["string"] not in used]
            if not free:
                return False
            used.add(min(free, key=lambda p: (p["fret"], p["string"]))["string"])
        return True

    def _assignments(self, options_by_pitch: list[list[dict]]) -> list[list[dict]]:
        """
        All string-disjoint assignments of one position per pitch.

        Never returns two notes on the same string: a string can only sound one
        note at a time, so any such assignment would be unplayable.
        """
        results: list[list[dict]] = []

        def walk(idx: int, used: set[int], current: list[dict]) -> None:
            if len(results) >= self.MAX_CANDIDATES_PER_COLUMN:
                return
            if idx == len(options_by_pitch):
                results.append(sorted(current, key=lambda p: p["string"]))
                return
            for position in options_by_pitch[idx]:
                if position["string"] in used:
                    continue
                used.add(position["string"])
                current.append(position)
                walk(idx + 1, used, current)
                current.pop()
                used.discard(position["string"])

        walk(0, set(), [])
        return results

    def finger_column(self, pitches: list[int]) -> tuple[list[list[dict]], list[int], str]:
        """
        Choose string/fret positions for one column, ranked best-first.

        Returns:
            (candidates, dropped_pitches, reason). Candidates are fingerings
            ordered by comfort. When the column cannot be fingered at all,
            ``candidates`` is empty and every pitch is reported as dropped.
        """
        # One string cannot sound the same pitch twice in one column. A pitch
        # that appears twice on the same onset is one note — Basic Pitch and
        # note_seq both emit that for a doubled take — so collapse it.
        unique: list[int] = []
        for pitch in pitches:
            if pitch not in unique:
                unique.append(pitch)

        options_by_pitch: list[list[dict]] = []
        dropped: list[int] = []
        reasons: list[str] = []
        for pitch in unique:
            options = self.get_valid_positions(pitch)
            if not options:
                dropped.append(pitch)
                reasons.append(self.playability_reason(pitch))
            else:
                # ``pitch`` rides along so the validator can re-derive
                # tuning[string] + fret == pitch for every row.
                options_by_pitch.append(
                    [dict(p, pitch=pitch) for p in sorted(options, key=lambda p: p["fret"])]
                )

        if not options_by_pitch:
            return [], dropped, "; ".join(reasons)

        candidates = self._assignments(options_by_pitch)
        if not candidates:
            # Every string is contested: drop the most constrained pitches and
            # keep whatever still fits, rather than losing the whole column.
            kept_options = list(options_by_pitch)
            while kept_options and not candidates:
                victim = min(range(len(kept_options)), key=lambda i: len(kept_options[i]))
                dropped.append(kept_options[victim][0]["pitch"])
                reasons.append("chord needs more strings than the instrument has")
                kept_options.pop(victim)
                candidates = self._assignments(kept_options)

        if not candidates:
            return [], dropped, "; ".join(reasons)

        # A column may be assignable yet still need a stretch no hand can make:
        # keep only the narrowest fingerings, and if even those are too wide drop
        # the notes that cannot be reached, keeping the rest of the chord.
        candidates = self._min_span_fingerings(candidates)
        playable = [c for c in candidates if self._hand_span(c) <= self.MAX_HAND_SPAN]
        if not playable:
            best, span_dropped = self._fit_hand_span(candidates[0])
            if span_dropped:
                dropped.extend(span_dropped)
                reasons.append(
                    f"dropped {len(span_dropped)} note(s) beyond a "
                    f"{self.MAX_HAND_SPAN}-fret hand span"
                )
            playable = [best]

        playable.sort(key=self._fingering_cost)
        return playable, dropped, "; ".join(reasons)

    def _min_span_fingerings(self, candidates: list[list[dict]]) -> list[list[dict]]:
        """
        The narrowest fingerings of a column.

        Some pitches sit on only one string, so a chord can be *assignable*
        (one note per string) yet still unplayable: pitch 76 on a 24-fret
        guitar only exists at fret 12 on the high E, so a voicing that also
        needs fret 4 must put fret 12 in the same column. Filtering by span
        here is what stops that from becoming an unplayable tab.
        """
        if not candidates:
            return []
        spans = [self._hand_span(c) for c in candidates]
        narrowest = min(spans)
        return [c for c, span in zip(candidates, spans, strict=True) if span == narrowest]

    def _fit_hand_span(self, fingering: list[dict]) -> tuple[list[dict], list[int]]:
        """
        Keep as many notes as fit the hand, drop the ones that cannot be reached.

        Every fretted note is tried as the anchor (the fret the hand sits at)
        and the anchor that keeps the most notes wins, lowest fret on a tie.
        Guitarists anchor on a specific fret rather than always on the lowest,
        so anchoring on the lowest note alone would drop playable voicings.
        """
        fretted = sorted((p for p in fingering if p["fret"] > 0), key=lambda p: p["fret"])
        opens = [p for p in fingering if p["fret"] == 0]
        if len(fretted) < 2:
            return sorted(opens + fretted, key=lambda p: p["string"]), []

        # Slide a window exactly MAX_HAND_SPAN frets wide over the fretted notes
        # and keep the window covering the most of them. Measuring distance from
        # an anchor instead would happily keep a 12-fret spread (each end within
        # 5 of the middle), which is the bug this replaces.
        best_kept: list[dict] = []
        for anchor_note in fretted:
            low = anchor_note["fret"]
            high = low + self.MAX_HAND_SPAN
            kept = [p for p in fretted if low <= p["fret"] <= high]
            if len(kept) > len(best_kept):
                best_kept = kept

        dropped = [p["pitch"] for p in fretted if p not in best_kept]
        return sorted(opens + best_kept, key=lambda p: p["string"]), dropped

    # ── technique detection ──────────────────────────────────────────────

    def _technique_window(self, technique_sensitivity: float) -> float:
        """Time window inside which a same-string move may be legato."""
        return 0.05 + technique_sensitivity * 0.25  # 0.5 -> 0.175 s, 0.9 -> 0.275 s

    def _detect_techniques(
        self,
        prev_column: dict,
        column: dict,
        prev_fingering: list[dict],
        fingering: list[dict],
        technique_sensitivity: float,
    ) -> dict[int, str]:
        """
        Label each note of ``column`` as pick / slide / hammer / pull.

        A technique requires all of:

        - the note has its own onset, strictly after the previous column
          (a chord is a single attack, so simultaneous notes are never legato);
        - it is on the same string as a note of the previous column;
        - the fret changed;
        - the previous note is still ringing when this one starts, so it can
          supply the energy. That is what makes a hammer-on a hammer-on, and it
          rules out two separately picked notes on one string.
        - slide: 3+ frets (a 1-2 fret move is far more likely a hammer/pull);
          otherwise ascending -> hammer, descending -> pull.
        """
        techniques: dict[int, str] = {}
        time_delta = column["time"] - prev_column["time"]
        if time_delta <= 0 or time_delta >= self._technique_window(technique_sensitivity):
            return techniques

        prev_by_string = {p["string"]: p for p in prev_fingering}
        prev_end_by_pitch: dict[int, float] = {}
        for note in prev_column["notes"]:
            pitch = int(note.pitch)
            prev_end_by_pitch[pitch] = max(
                prev_end_by_pitch.get(pitch, float("-inf")), float(note.end_time)
            )

        for position in fingering:
            source = prev_by_string.get(position["string"])
            if source is None:
                continue
            fret_diff = position["fret"] - source["fret"]
            if fret_diff == 0:
                continue
            still_ringing = prev_end_by_pitch.get(source["pitch"], float("-inf")) > column["time"]
            if not still_ringing:
                continue
            if abs(fret_diff) >= self.SLIDE_MIN_FRETS:
                techniques[position["pitch"]] = "slide"
            elif fret_diff > 0:
                techniques[position["pitch"]] = "hammer"
            else:
                techniques[position["pitch"]] = "pull"
        return techniques

    # ── public entry point ───────────────────────────────────────────────

    def generate_tab(
        self,
        midi_notes: list,
        technique_sensitivity: float = 0.7,
    ) -> list[dict]:
        """
        Generate tablature, fingering one onset column at a time.

        Algorithm: Viterbi over columns.
        1. Group notes into columns of simultaneous onsets.
        2. For each column, choose one string/fret per note (no shared strings,
           bounded hand span).
        3. Keep the cheapest path of fingerings across columns.
        4. Annotate techniques (slides, hammer-ons, pull-offs) between columns.

        Notes that cannot be fingered are skipped, recorded in
        ``last_skipped_notes`` / ``last_skipped_columns``, and the remaining
        notes still produce tablature. Returns an empty list only when nothing
        is playable.

        Args:
            midi_notes: List of note_seq.NoteSequence.Note objects
            technique_sensitivity: 0-1 threshold for technique detection
                (higher = more aggressive detection, wider time windows)

        Returns:
            List of tab positions, ordered by column then string. Each entry has
            ``string``, ``fret``, ``pitch``, ``technique``, ``start_time`` and
            ``column``.
        """
        self.last_skipped_notes = []
        self.last_skipped_columns = []

        if not midi_notes:
            return []

        # 1-2. Group into columns and finger each one independently, so an
        # unplayable chord can never destroy the rest of the track.
        columns = self.group_notes(midi_notes)
        fingered: list[dict] = []
        for column in columns:
            candidates, dropped, reason = self.finger_column(
                [int(note.pitch) for note in column["notes"]]
            )
            if dropped:
                self.last_skipped_notes.extend(dropped)
                self.last_skipped_columns.append(
                    {"time": column["time"], "dropped": sorted(dropped), "reason": reason}
                )
                get_logger("tab_agent").info(
                    "column_partially_skipped",
                    time=round(column["time"], 3),
                    dropped=sorted(dropped),
                    kept=len(candidates[0]) if candidates else 0,
                    reason=reason,
                )
            if candidates:
                fingered.append({"column": column, "candidates": candidates})

        if not fingered:
            if self.last_skipped_notes:
                get_logger("tab_agent").warning(
                    "unplayable_notes_skipped",
                    pitches=sorted(set(self.last_skipped_notes)),
                    skipped=len(self.last_skipped_notes),
                )
            return []

        # 3. Viterbi: state = (column index, candidate fingering index).
        costs: list[list[float]] = []
        back: list[list[int]] = []
        for idx, entry in enumerate(fingered):
            prev_entry = fingered[idx - 1] if idx > 0 else None
            time_delta = (
                entry["column"]["time"] - prev_entry["column"]["time"] if prev_entry else 0.0
            )
            column_costs: list[float] = []
            column_back: list[int] = []
            for candidate in entry["candidates"]:
                base = self._fingering_cost(candidate)
                if prev_entry is None:
                    column_costs.append(base)
                    column_back.append(-1)
                    continue
                best_cost, best_prev = float("inf"), 0
                for prev_idx, prev_candidate in enumerate(prev_entry["candidates"]):
                    candidate_cost = (
                        costs[-1][prev_idx]
                        + base
                        + self._column_cost(prev_candidate, candidate, time_delta)
                    )
                    if candidate_cost < best_cost:
                        best_cost, best_prev = candidate_cost, prev_idx
                column_costs.append(best_cost)
                column_back.append(best_prev)
            costs.append(column_costs)
            back.append(column_back)

        # Backtrack.
        best = costs[-1].index(min(costs[-1]))
        path: list[list[dict]] = []
        for idx in range(len(fingered) - 1, -1, -1):
            path.append(fingered[idx]["candidates"][best])
            best = back[idx][best] if back[idx][best] >= 0 else 0
        path.reverse()

        # 4. Annotate techniques, then flatten.
        final_tab: list[dict] = []
        for idx, fingering in enumerate(path):
            techniques: dict[int, str] = {}
            if idx > 0:
                techniques = self._detect_techniques(
                    fingered[idx - 1]["column"],
                    fingered[idx]["column"],
                    path[idx - 1],
                    fingering,
                    technique_sensitivity,
                )
            for position in fingering:
                row = dict(position)
                row["technique"] = techniques.get(position["pitch"], "pick")
                row["start_time"] = fingered[idx]["column"]["time"]
                row["column"] = idx
                final_tab.append(row)

        # Enforce the invariants this fingering model depends on instead of
        # trusting them: one note per string per column, every string/fret
        # matching its pitch, bounded hand span. A violation is a bug here, not
        # a bad input, so it raises rather than printing an unplayable tab.
        self.last_validation = validate_tab(
            final_tab,
            self.tuning,
            self.num_frets,
            max_hand_span=self.MAX_HAND_SPAN,
        )
        return final_tab
