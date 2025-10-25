"""
Feature extraction for Raag classification.
Implements Constant-Q Transform (CQT) and spectrograms optimized for pitched Indian classical music.
"""

import numpy as np
import librosa
from typing import Optional, Tuple, List
import warnings


class FeatureExtractor:
    """
    Extract time-frequency representations suitable for Raag classification.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_bins: int = 84,  # 7 octaves
        bins_per_octave: int = 12,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        use_cqt: bool = True,
        n_mels: int = 128,
        n_fft: int = 2048,
    ):
        """
        Args:
            sample_rate: Audio sample rate
            n_bins: Number of frequency bins for CQT
            bins_per_octave: Bins per octave for CQT (12 for semitone resolution)
            hop_length: Hop length between frames
            fmin: Minimum frequency for CQT (defaults to C1)
            use_cqt: Whether to use CQT (True) or Mel spectrogram (False)
            n_mels: Number of mel bands for mel spectrogram
            n_fft: FFT size for mel spectrogram
        """
        self.sample_rate = sample_rate
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.hop_length = hop_length
        self.fmin = fmin or librosa.note_to_hz('C1')
        self.use_cqt = use_cqt
        self.n_mels = n_mels
        self.n_fft = n_fft

    def extract_cqt(
        self,
        audio: np.ndarray,
        log_scale: bool = True,
    ) -> np.ndarray:
        """
        Extract Constant-Q Transform features.

        CQT is ideal for music analysis as it has logarithmic frequency spacing
        that matches musical pitch perception.

        Args:
            audio: Audio signal
            log_scale: Whether to apply log scaling

        Returns:
            CQT features with shape [n_bins, time_frames]
        """
        # Compute CQT
        cqt = librosa.cqt(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
        )

        # Convert to magnitude
        cqt_mag = np.abs(cqt)

        # Apply log scaling
        if log_scale:
            cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)
            return cqt_db
        else:
            return cqt_mag

    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        log_scale: bool = True,
    ) -> np.ndarray:
        """
        Extract Mel spectrogram features.

        Args:
            audio: Audio signal
            log_scale: Whether to apply log scaling

        Returns:
            Mel spectrogram with shape [n_mels, time_frames]
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
        )

        # Apply log scaling
        if log_scale:
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_db
        else:
            return mel_spec

    def extract_features(
        self,
        audio: np.ndarray,
        include_delta: bool = True,
        include_delta_delta: bool = False,
    ) -> np.ndarray:
        """
        Extract primary features (CQT or Mel) with optional deltas.

        Args:
            audio: Audio signal
            include_delta: Include delta features (velocity)
            include_delta_delta: Include delta-delta features (acceleration)

        Returns:
            Feature array with shape [n_features, time_frames]
        """
        # Extract primary features
        if self.use_cqt:
            features = self.extract_cqt(audio)
        else:
            features = self.extract_mel_spectrogram(audio)

        # Stack with deltas if requested
        feature_list = [features]

        if include_delta:
            delta = librosa.feature.delta(features)
            feature_list.append(delta)

        if include_delta_delta:
            delta_delta = librosa.feature.delta(features, order=2)
            feature_list.append(delta_delta)

        # Stack along frequency dimension
        if len(feature_list) > 1:
            features_stacked = np.vstack(feature_list)
            return features_stacked
        else:
            return features

    def extract_additional_features(
        self,
        audio: np.ndarray,
    ) -> dict:
        """
        Extract additional features useful for Raag classification.

        Args:
            audio: Audio signal

        Returns:
            Dictionary of additional features
        """
        features = {}

        # Chroma features (pitch class distribution)
        chroma = librosa.feature.chroma_cqt(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        features['chroma'] = chroma

        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        features['spectral_contrast'] = spectral_contrast

        # Tonnetz (tonal centroid features)
        tonnetz = librosa.feature.tonnetz(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        features['tonnetz'] = tonnetz

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            hop_length=self.hop_length,
        )
        features['zcr'] = zcr

        return features


class AudioSegmenter:
    """
    Segment audio into fixed-length chunks with optional overlap.
    """

    def __init__(
        self,
        segment_duration: float = 5.0,
        overlap_duration: float = 2.5,
        sample_rate: int = 22050,
    ):
        """
        Args:
            segment_duration: Duration of each segment in seconds
            overlap_duration: Overlap between segments in seconds
            sample_rate: Audio sample rate
        """
        self.segment_duration = segment_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = sample_rate

        self.segment_samples = int(segment_duration * sample_rate)
        self.overlap_samples = int(overlap_duration * sample_rate)
        self.hop_samples = self.segment_samples - self.overlap_samples

    def segment_audio(
        self,
        audio: np.ndarray,
        pad: bool = True,
    ) -> List[np.ndarray]:
        """
        Segment audio into fixed-length chunks.

        Args:
            audio: Audio signal
            pad: Whether to pad the last segment if too short

        Returns:
            List of audio segments
        """
        segments = []

        # Handle short audio
        if len(audio) < self.segment_samples:
            if pad:
                padded = np.zeros(self.segment_samples)
                padded[:len(audio)] = audio
                segments.append(padded)
            else:
                # Skip if too short
                pass
            return segments

        # Extract segments with overlap
        for start in range(0, len(audio) - self.segment_samples + 1, self.hop_samples):
            end = start + self.segment_samples
            segment = audio[start:end]
            segments.append(segment)

        # Handle remaining audio
        if pad and (len(audio) - start - self.segment_samples) > 0:
            last_segment = np.zeros(self.segment_samples)
            remaining = audio[start + self.hop_samples:]
            last_segment[:len(remaining)] = remaining
            segments.append(last_segment)

        return segments

    def segment_features(
        self,
        features: np.ndarray,
        segment_duration: Optional[float] = None,
    ) -> List[np.ndarray]:
        """
        Segment feature array along time dimension.

        Args:
            features: Feature array with shape [n_features, time_frames]
            segment_duration: Optional override for segment duration

        Returns:
            List of feature segments
        """
        if segment_duration is None:
            segment_duration = self.segment_duration

        # Assume features are computed with hop_length
        # Estimate frame rate (features per second)
        # This is approximate; actual frame rate depends on hop_length used
        # For safety, this should be passed as a parameter
        # For now, we'll segment by number of frames

        # For a 5-second segment with hop_length=512 and sr=22050
        # frames = 5 * 22050 / 512 ≈ 215 frames
        frames_per_segment = int(
            segment_duration * self.sample_rate / 512  # Assuming hop_length=512
        )

        segments = []
        n_frames = features.shape[1]

        for start in range(0, n_frames, frames_per_segment):
            end = start + frames_per_segment
            if end > n_frames:
                # Pad last segment
                segment = np.zeros((features.shape[0], frames_per_segment))
                segment[:, :n_frames - start] = features[:, start:]
            else:
                segment = features[:, start:end]
            segments.append(segment)

        return segments

    def normalize_segment(
        self,
        segment: np.ndarray,
        method: str = 'peak',
    ) -> np.ndarray:
        """
        Normalize audio segment amplitude.

        Args:
            segment: Audio segment
            method: Normalization method ('peak' or 'rms')

        Returns:
            Normalized segment
        """
        if method == 'peak':
            # Peak normalization
            max_val = np.max(np.abs(segment))
            if max_val > 0:
                normalized = segment / max_val
            else:
                normalized = segment
        elif method == 'rms':
            # RMS normalization
            rms = np.sqrt(np.mean(segment ** 2))
            if rms > 0:
                normalized = segment / rms
            else:
                normalized = segment
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized


def extract_and_save_features(
    audio: np.ndarray,
    output_path: str,
    sample_rate: int = 22050,
    use_cqt: bool = True,
    segment_duration: float = 5.0,
    normalize: bool = True,
) -> None:
    """
    Convenience function to extract features and save to file.

    Args:
        audio: Audio signal
        output_path: Path to save features (.npy file)
        sample_rate: Sample rate
        use_cqt: Whether to use CQT
        segment_duration: Segment duration in seconds
        normalize: Whether to normalize segments
    """
    # Extract features
    extractor = FeatureExtractor(sample_rate=sample_rate, use_cqt=use_cqt)
    features = extractor.extract_features(audio)

    # Save features
    np.save(output_path, features)

    return features
