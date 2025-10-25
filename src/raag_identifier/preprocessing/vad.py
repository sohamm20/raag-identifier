"""
Voice Activity Detection (VAD) for removing silence and non-vocal segments.
Implements energy-based thresholding and harmonic content detection.
"""

import numpy as np
import librosa
from typing import List, Tuple, Optional
import webrtcvad


class VoiceActivityDetector:
    """
    Voice Activity Detection using energy thresholding and harmonic content.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        frame_length: int = 2048,
        hop_length: int = 512,
        energy_threshold: float = 0.02,
        use_webrtc: bool = False,
        webrtc_mode: int = 3,
    ):
        """
        Args:
            sample_rate: Audio sample rate
            frame_length: Frame length for analysis
            hop_length: Hop length between frames
            energy_threshold: Energy threshold (fraction of max energy)
            use_webrtc: Whether to use WebRTC VAD
            webrtc_mode: WebRTC VAD aggressiveness (0-3, 3 is most aggressive)
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.use_webrtc = use_webrtc

        if use_webrtc and sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError("WebRTC VAD requires sample rate in [8000, 16000, 32000, 48000]")

        if use_webrtc:
            self.vad = webrtcvad.Vad(webrtc_mode)
        else:
            self.vad = None

    def compute_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute energy per frame.

        Args:
            audio: Audio signal

        Returns:
            Energy per frame
        """
        # Compute RMS energy per frame
        energy = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]

        return energy

    def compute_harmonic_ratio(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute harmonic-to-percussive ratio as indicator of vocal content.

        Args:
            audio: Audio signal

        Returns:
            Harmonic ratio per frame
        """
        # Separate harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(audio)

        # Compute energy for each
        harmonic_energy = librosa.feature.rms(
            y=harmonic,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]

        percussive_energy = librosa.feature.rms(
            y=percussive,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]

        # Compute ratio (avoid division by zero)
        total_energy = harmonic_energy + percussive_energy + 1e-8
        harmonic_ratio = harmonic_energy / total_energy

        return harmonic_ratio

    def detect_voice_activity_energy(
        self,
        audio: np.ndarray,
        use_harmonic: bool = True,
        harmonic_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Detect voice activity using energy and harmonic content.

        Args:
            audio: Audio signal
            use_harmonic: Whether to use harmonic ratio
            harmonic_threshold: Threshold for harmonic ratio

        Returns:
            Boolean mask indicating voice activity per frame
        """
        # Compute energy
        energy = self.compute_energy(audio)

        # Normalize energy
        if np.max(energy) > 0:
            energy_normalized = energy / np.max(energy)
        else:
            energy_normalized = energy

        # Energy-based detection
        voice_frames = energy_normalized > self.energy_threshold

        # Optional: refine with harmonic content
        if use_harmonic:
            harmonic_ratio = self.compute_harmonic_ratio(audio)
            harmonic_frames = harmonic_ratio > harmonic_threshold
            voice_frames = voice_frames & harmonic_frames

        return voice_frames

    def detect_voice_activity_webrtc(self, audio: np.ndarray) -> np.ndarray:
        """
        Detect voice activity using WebRTC VAD.

        Args:
            audio: Audio signal (will be resampled if needed)

        Returns:
            Boolean mask indicating voice activity
        """
        if self.vad is None:
            raise ValueError("WebRTC VAD not initialized")

        # WebRTC VAD requires 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)

        # Frame size must be 10, 20, or 30 ms
        frame_duration_ms = 30
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)

        # Pad audio to fit frame size
        num_frames = int(np.ceil(len(audio_int16) / frame_size))
        padded_length = num_frames * frame_size
        audio_padded = np.zeros(padded_length, dtype=np.int16)
        audio_padded[:len(audio_int16)] = audio_int16

        # Process frames
        voice_frames = []
        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio_padded[start:end].tobytes()

            # Check if frame contains speech
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            voice_frames.append(is_speech)

        return np.array(voice_frames)

    def get_voice_segments(
        self,
        audio: np.ndarray,
        use_harmonic: bool = True,
        min_segment_length: float = 0.5,
    ) -> List[Tuple[int, int]]:
        """
        Get start and end indices of voice segments.

        Args:
            audio: Audio signal
            use_harmonic: Whether to use harmonic content
            min_segment_length: Minimum segment length in seconds

        Returns:
            List of (start_sample, end_sample) tuples
        """
        if self.use_webrtc:
            voice_frames = self.detect_voice_activity_webrtc(audio)
            frame_size = int(self.sample_rate * 30 / 1000)  # 30ms frames
        else:
            voice_frames = self.detect_voice_activity_energy(audio, use_harmonic)
            frame_size = self.hop_length

        # Find segments
        segments = []
        in_segment = False
        start_frame = 0

        for i, is_voice in enumerate(voice_frames):
            if is_voice and not in_segment:
                start_frame = i
                in_segment = True
            elif not is_voice and in_segment:
                # Segment ended
                start_sample = start_frame * frame_size
                end_sample = i * frame_size
                segments.append((start_sample, end_sample))
                in_segment = False

        # Handle last segment
        if in_segment:
            start_sample = start_frame * frame_size
            end_sample = len(audio)
            segments.append((start_sample, end_sample))

        # Filter segments by minimum length
        min_samples = int(min_segment_length * self.sample_rate)
        filtered_segments = [
            (start, end) for start, end in segments
            if (end - start) >= min_samples
        ]

        return filtered_segments

    def remove_silence(
        self,
        audio: np.ndarray,
        use_harmonic: bool = True,
        min_segment_length: float = 0.5,
        pad_duration: float = 0.1,
    ) -> np.ndarray:
        """
        Remove silence and non-vocal segments from audio.

        Args:
            audio: Input audio signal
            use_harmonic: Whether to use harmonic content
            min_segment_length: Minimum segment length in seconds
            pad_duration: Padding around segments in seconds

        Returns:
            Audio with silence removed
        """
        segments = self.get_voice_segments(audio, use_harmonic, min_segment_length)

        if len(segments) == 0:
            # No voice detected, return short silence
            return np.zeros(int(0.1 * self.sample_rate))

        # Add padding
        pad_samples = int(pad_duration * self.sample_rate)
        padded_segments = []
        for start, end in segments:
            start_padded = max(0, start - pad_samples)
            end_padded = min(len(audio), end + pad_samples)
            padded_segments.append((start_padded, end_padded))

        # Merge overlapping segments
        merged_segments = []
        current_start, current_end = padded_segments[0]

        for start, end in padded_segments[1:]:
            if start <= current_end:
                # Overlapping or adjacent, merge
                current_end = max(current_end, end)
            else:
                # Non-overlapping, save current and start new
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end

        merged_segments.append((current_start, current_end))

        # Concatenate voice segments
        voice_audio = np.concatenate([
            audio[start:end] for start, end in merged_segments
        ])

        return voice_audio


def apply_vad_to_file(
    audio: np.ndarray,
    sample_rate: int = 22050,
    use_webrtc: bool = False,
    energy_threshold: float = 0.02,
    min_segment_length: float = 0.5,
) -> np.ndarray:
    """
    Convenience function to apply VAD to an audio file.

    Args:
        audio: Audio signal
        sample_rate: Sample rate
        use_webrtc: Whether to use WebRTC VAD
        energy_threshold: Energy threshold
        min_segment_length: Minimum segment length in seconds

    Returns:
        Audio with silence removed
    """
    vad = VoiceActivityDetector(
        sample_rate=sample_rate,
        use_webrtc=use_webrtc,
        energy_threshold=energy_threshold,
    )

    return vad.remove_silence(
        audio,
        use_harmonic=True,
        min_segment_length=min_segment_length,
    )
