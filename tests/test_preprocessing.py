"""
Unit tests for preprocessing modules.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from raag_identifier.preprocessing import (
    VoiceActivityDetector,
    FeatureExtractor,
    AudioSegmenter,
    AudioAugmentation,
    SpecAugment,
)


@pytest.fixture
def sample_audio():
    """Generate sample audio signal."""
    duration = 5.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    # Mix of frequencies
    audio = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    return audio, sr


def test_vad_initialization():
    """Test VAD initialization."""
    vad = VoiceActivityDetector(sample_rate=22050)
    assert vad.sample_rate == 22050
    assert vad.energy_threshold == 0.02


def test_vad_compute_energy(sample_audio):
    """Test energy computation."""
    audio, sr = sample_audio
    vad = VoiceActivityDetector(sample_rate=sr)

    energy = vad.compute_energy(audio)

    assert energy.shape[0] > 0
    assert np.all(energy >= 0)


def test_vad_remove_silence(sample_audio):
    """Test silence removal."""
    audio, sr = sample_audio

    # Add silence
    silence = np.zeros(int(sr * 2))
    audio_with_silence = np.concatenate([silence, audio, silence])

    vad = VoiceActivityDetector(sample_rate=sr)
    processed = vad.remove_silence(audio_with_silence)

    # Processed audio should be shorter
    assert len(processed) < len(audio_with_silence)


def test_feature_extractor_cqt(sample_audio):
    """Test CQT feature extraction."""
    audio, sr = sample_audio
    extractor = FeatureExtractor(sample_rate=sr, use_cqt=True)

    features = extractor.extract_cqt(audio)

    assert features.shape[0] == extractor.n_bins
    assert features.shape[1] > 0


def test_feature_extractor_mel(sample_audio):
    """Test Mel spectrogram extraction."""
    audio, sr = sample_audio
    extractor = FeatureExtractor(sample_rate=sr, use_cqt=False)

    features = extractor.extract_mel_spectrogram(audio)

    assert features.shape[0] == extractor.n_mels
    assert features.shape[1] > 0


def test_feature_extractor_with_deltas(sample_audio):
    """Test feature extraction with deltas."""
    audio, sr = sample_audio
    extractor = FeatureExtractor(sample_rate=sr, use_cqt=True)

    features = extractor.extract_features(audio, include_delta=True, include_delta_delta=False)

    # Should have 2x the bins (features + deltas)
    assert features.shape[0] == extractor.n_bins * 2


def test_audio_segmenter(sample_audio):
    """Test audio segmentation."""
    audio, sr = sample_audio
    segmenter = AudioSegmenter(segment_duration=2.0, overlap_duration=1.0, sample_rate=sr)

    segments = segmenter.segment_audio(audio, pad=True)

    assert len(segments) > 0
    # Each segment should be 2 seconds
    assert all(len(seg) == int(2.0 * sr) for seg in segments)


def test_audio_segmenter_normalize(sample_audio):
    """Test segment normalization."""
    audio, sr = sample_audio
    segmenter = AudioSegmenter(sample_rate=sr)

    # Peak normalization
    normalized = segmenter.normalize_segment(audio, method='peak')
    assert np.max(np.abs(normalized)) <= 1.0


def test_audio_augmentation_pitch_shift(sample_audio):
    """Test pitch shifting."""
    audio, sr = sample_audio
    augmenter = AudioAugmentation(sample_rate=sr)

    shifted = augmenter.pitch_shift(audio, n_steps=2)

    assert shifted.shape == audio.shape
    assert not np.allclose(shifted, audio)


def test_audio_augmentation_time_stretch(sample_audio):
    """Test time stretching."""
    audio, sr = sample_audio
    augmenter = AudioAugmentation(sample_rate=sr)

    stretched = augmenter.time_stretch(audio, rate=1.1)

    # Stretched audio should be shorter (faster)
    assert len(stretched) < len(audio)


def test_audio_augmentation_add_noise(sample_audio):
    """Test noise addition."""
    audio, sr = sample_audio
    augmenter = AudioAugmentation(sample_rate=sr)

    noisy = augmenter.add_noise(audio, noise_level=0.01)

    assert noisy.shape == audio.shape
    assert not np.allclose(noisy, audio)
    assert np.max(np.abs(noisy)) <= 1.0  # Should be clipped


def test_spec_augment():
    """Test SpecAugment."""
    # Create sample spectrogram
    spec = np.random.randn(84, 100)

    augmenter = SpecAugment(freq_mask_param=10, time_mask_param=10)

    # Frequency mask
    masked = augmenter.freq_mask(spec)
    assert masked.shape == spec.shape
    assert np.sum(masked == 0) > np.sum(spec == 0)  # More zeros after masking

    # Time mask
    masked = augmenter.time_mask(spec)
    assert masked.shape == spec.shape
    assert np.sum(masked == 0) > np.sum(spec == 0)


def test_spec_augment_full(sample_audio):
    """Test full SpecAugment pipeline."""
    audio, sr = sample_audio

    # Extract features
    extractor = FeatureExtractor(sample_rate=sr, use_cqt=True)
    spec = extractor.extract_cqt(audio)

    # Apply SpecAugment
    augmenter = SpecAugment()
    augmented = augmenter.augment(spec)

    assert augmented.shape == spec.shape
    # Should have some masked regions
    assert not np.allclose(augmented, spec)
