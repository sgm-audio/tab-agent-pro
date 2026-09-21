"""
Suno Artifact Post-Processor
Applies heuristics to improve transcription quality on AI-generated audio.

Detection thresholds are tuned for modern Suno v3/v4 and Udio output.
Configurable aggressiveness lets users trade recall for precision.

Aggressiveness levels:
  0.0 = Conservative (fewer false positives, may miss some AI audio)
  0.5 = Balanced (default, good for mixed-use pipelines)
  1.0 = Aggressive (maximum detection, higher false-positive risk)

The harmonic peak filter prevents pure tones (test signals, sine sweeps)
from being falsely flagged regardless of aggressiveness.
"""

import librosa
import note_seq
import numpy as np
import soundfile as sf


class SunoArtifactDetector:
    """
    Detects AI-generated audio artifacts (Suno/Udio signatures).

    Based on spectral analysis of Suno v3/v4 and Udio generations.
    Configurable aggressiveness scales all detection thresholds.

    Args:
        aggressiveness: 0.0-1.0 detection sensitivity
            0.0 = conservative, 0.5 = balanced (default), 1.0 = aggressive

    """

    def __init__(self, aggressiveness: float = 0.5) -> None:
        self.sample_rate = 22050
        self.aggressiveness = float(np.clip(aggressiveness, 0.0, 1.0))
        self._hf_ratio_threshold = 0.35 - (self.aggressiveness * 0.15)  # 0.20-0.35
        self._flatness_threshold = 0.008 + ((1.0 - self.aggressiveness) * 0.004)  # 0.008-0.012
        self._combo_flatness = 0.015 + ((1.0 - self.aggressiveness) * 0.005)  # 0.015-0.020
        self._combo_hf = 0.25 - (self.aggressiveness * 0.10)  # 0.15-0.25

    def analyze(self, audio_path: str) -> tuple[bool, dict]:
        """
        Analyze audio for AI generation artifacts.

        Returns:
            (is_ai_generated, metrics_dict)

        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True, duration=30)

        # Compute spectrogram
        spec = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        metrics = {}

        # 1. High-frequency "metallic shimmer" (Suno signature)
        # AI models often have unnatural energy at 8-16kHz
        hf_mask = freqs > 8000
        hf_energy = np.mean(spec[hf_mask])
        total_energy = np.mean(spec)
        metrics["hf_ratio"] = hf_energy / (total_energy + 1e-10)

        # 2. Spectral flatness (naturalness measure)
        # Lower flatness = more AI-like (less natural variation)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        metrics["spectral_flatness"] = np.mean(spectral_flatness)

        # 3. Temporal consistency
        # AI audio often has unnatural temporal consistency
        rms = librosa.feature.rms(y=y)[0]
        metrics["rms_variance"] = np.var(rms)

        # 4. Zero-crossing rate
        # AI audio sometimes has unusual zero-crossing patterns
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        metrics["zcr_mean"] = np.mean(zcr)

        # 5. Harmonic peak density — distinguishes pure tones from complex audio
        #   - Pure sine waves: 1-2 prominent peaks → NOT AI audio
        #   - Real instruments / AI audio: many peaks across the spectrum
        #   Counteracts the spectral_flatness false-positive on pure tones.
        from scipy.signal import find_peaks

        mean_spectrum = np.mean(spec, axis=1)  # Average over time
        prominence = np.max(mean_spectrum) * 0.02
        peaks, _ = find_peaks(mean_spectrum, prominence=prominence)
        peak_count = len(peaks)
        metrics["harmonic_peak_count"] = int(peak_count)

        is_pure_tone = peak_count <= 3  # 1-3 prominent peaks = test tone / sine
        if is_pure_tone:
            metrics["spectral_flatness"] = 0.5  # Override so flatness rule won't trigger

        # Decision heuristics (tuned on Suno v3/v4 + Udio, configurable aggressiveness)
        is_suno = (
            metrics["hf_ratio"] > self._hf_ratio_threshold
            or (not is_pure_tone and metrics["spectral_flatness"] < self._flatness_threshold)
            or (
                metrics["hf_ratio"] > self._combo_hf
                and metrics["spectral_flatness"] < self._combo_flatness
            )
        )

        if is_suno:
            pass
        else:
            pass

        return is_suno, metrics


class SunoAudioPreprocessor:
    """Pre-process audio to reduce Suno artifacts before transcription."""

    def __init__(self) -> None:
        self.sample_rate = 22050

    def process(self, audio_path: str, output_path: str) -> str:
        """
        Apply preprocessing to reduce AI artifacts.

        Args:
            audio_path: Input audio file
            output_path: Output processed audio file

        Returns:
            Path to processed audio

        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # 1. High-pass filter (remove ultra-low rumble common in AI audio)
        y = self._highpass_filter(y, sr, cutoff=40)

        # 2. Reduce high-frequency metallic shimmer
        y = self._reduce_hf_artifacts(y, sr)

        # 3. Spectral gating (reduce background AI noise floor)
        y = self._spectral_gate(y, sr)

        # Save processed audio
        sf.write(output_path, y, sr)

        return output_path

    def _highpass_filter(self, y: np.ndarray, sr: float, cutoff: int = 40) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency rumble."""
        from scipy.signal import butter, filtfilt

        nyquist = sr / 2
        normal_cutoff = cutoff / nyquist
        b, a = butter(4, normal_cutoff, btype="high", analog=False)
        return filtfilt(b, a, y)

    def _reduce_hf_artifacts(self, y: np.ndarray, sr: float) -> np.ndarray:
        """
        Reduce high-frequency metallic artifacts (8-16kHz).

        Strategy: Apply gentle low-pass filter or reduce gain in problem bands.
        """
        # Use STFT to target specific frequency bands
        stft_matrix = librosa.stft(y)
        mag, phase = librosa.magphase(stft_matrix)

        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=sr)

        # Reduce gain in 8-16kHz range (Suno artifact zone)
        hf_start = np.argmax(freqs >= 8000)
        hf_end = np.argmax(freqs >= 16000)

        # Apply reduction (0.3 = reduce to 30% of original)
        mag[hf_start:hf_end, :] *= 0.3

        # Reconstruct
        processed = mag * phase
        return librosa.istft(processed)

    def _spectral_gate(self, y: np.ndarray, sr: float, threshold_db: float = -40) -> np.ndarray:
        """
        Apply spectral gating to reduce noise floor.

        Removes frequency components below threshold.
        """
        stft_matrix = librosa.stft(y)
        mag, phase = librosa.magphase(stft_matrix)

        # Convert to dB
        mag_db = librosa.amplitude_to_db(mag, ref=np.max)

        # Create mask (1 = keep, 0 = remove)
        mask = (mag_db > threshold_db).astype(float)

        # Apply mask
        mag_gated = mag * mask

        # Reconstruct
        gated = mag_gated * phase
        return librosa.istft(gated)


class SunoNotePostprocessor:
    """Post-process transcribed notes to fix common Suno errors."""

    def process(
        self,
        notes: list[note_seq.NoteSequence.Note],
        is_suno: bool,
        metrics: dict,
    ) -> list[note_seq.NoteSequence.Note]:
        """
        Apply post-processing to fix Suno-specific transcription errors.

        Args:
            notes: Transcribed notes
            is_suno: Whether audio is AI-generated
            metrics: Detection metrics from SunoArtifactDetector

        Returns:
            Cleaned notes

        """
        if not is_suno:
            return notes  # No processing needed for clean audio

        notes = self._remove_octave_errors(notes)
        notes = self._remove_spurious_high_notes(notes)
        return self._smooth_timing(notes)

    def _remove_octave_errors(
        self,
        notes: list[note_seq.NoteSequence.Note],
    ) -> list[note_seq.NoteSequence.Note]:
        """
        Remove octave doubling errors (common in Suno transcriptions).

        Suno's metallic shimmer often causes false harmonics.
        """
        cleaned = []
        i = 0

        while i < len(notes):
            note = notes[i]

            # Check if next note is octave above/below at same time
            if i + 1 < len(notes):
                next_note = notes[i + 1]
                time_diff = abs(next_note.start_time - note.start_time)
                pitch_diff = abs(next_note.pitch - note.pitch)

                # If notes are simultaneous and exactly 12 semitones apart
                if time_diff < 0.05 and pitch_diff == 12:
                    # Keep the lower note (usually correct)
                    cleaned.append(note if note.pitch < next_note.pitch else next_note)
                    i += 2  # Skip both
                    continue

            cleaned.append(note)
            i += 1

        return cleaned

    def _remove_spurious_high_notes(
        self,
        notes: list[note_seq.NoteSequence.Note],
        threshold_pitch: int = 84,  # High E (12th fret, high E string)
    ) -> list[note_seq.NoteSequence.Note]:
        """
        Remove spurious ultra-high notes caused by HF artifacts.

        Suno often transcribes metallic shimmer as very high notes.
        """
        # Count notes above threshold
        high_notes = [n for n in notes if n.pitch > threshold_pitch]
        high_ratio = len(high_notes) / (len(notes) + 1e-10)

        # If >30% of notes are suspiciously high, likely artifacts
        if high_ratio > 0.3:
            return [n for n in notes if n.pitch <= threshold_pitch]

        return notes

    def _smooth_timing(
        self,
        notes: list[note_seq.NoteSequence.Note],
        quantize_ms: float = 50,
    ) -> list[note_seq.NoteSequence.Note]:
        """
        Quantize timing to remove jitter from AI artifacts.

        Suno audio sometimes has unstable transients.
        Returns new Note objects — does NOT mutate the input list.
        """
        quantize_sec = quantize_ms / 1000.0
        smoothed = []

        for note in notes:
            smoothed.append(
                note_seq.NoteSequence.Note(
                    pitch=note.pitch,
                    start_time=round(note.start_time / quantize_sec) * quantize_sec,
                    end_time=round(note.end_time / quantize_sec) * quantize_sec,
                    velocity=note.velocity,
                ),
            )

        return smoothed


# Convenience function for integration with main pipeline
def process_suno_audio(audio_path: str, output_path: str | None = None) -> tuple[str, bool, dict]:
    """
    Detect and preprocess Suno audio in one call.

    Args:
        audio_path: Input audio
        output_path: Output preprocessed audio (auto-generated if None)

    Returns:
        (processed_audio_path, is_suno, metrics)

    """
    detector = SunoArtifactDetector()
    is_suno, metrics = detector.analyze(audio_path)

    if is_suno:
        preprocessor = SunoAudioPreprocessor()
        if output_path is None:
            import os

            base, ext = os.path.splitext(audio_path)
            output_path = f"{base}_processed{ext}"
        processed_path = preprocessor.process(audio_path, output_path)
        return processed_path, is_suno, metrics
    # No processing needed
    return audio_path, is_suno, metrics
