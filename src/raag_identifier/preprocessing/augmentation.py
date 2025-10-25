"""
Data augmentation for Raag audio classification.
Implements pitch shifting, time stretching, noise addition, and random cropping.
"""

import numpy as np
import librosa
from typing import Optional, Tuple
import torch


class AudioAugmentation:
    """
    Audio data augmentation techniques for Raag classification.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        pitch_shift_range: Tuple[float, float] = (-2.0, 2.0),
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        noise_level: float = 0.005,
        random_crop: bool = True,
    ):
        """
        Args:
            sample_rate: Audio sample rate
            pitch_shift_range: Range for pitch shifting in semitones (min, max)
            time_stretch_range: Range for time stretching ratio (min, max)
            noise_level: Level of additive Gaussian noise
            random_crop: Whether to apply random cropping
        """
        self.sample_rate = sample_rate
        self.pitch_shift_range = pitch_shift_range
        self.time_stretch_range = time_stretch_range
        self.noise_level = noise_level
        self.random_crop = random_crop

    def pitch_shift(
        self,
        audio: np.ndarray,
        n_steps: Optional[float] = None,
    ) -> np.ndarray:
        """
        Shift audio pitch by n semitones.

        Args:
            audio: Input audio
            n_steps: Number of semitones to shift (random if None)

        Returns:
            Pitch-shifted audio
        """
        if n_steps is None:
            n_steps = np.random.uniform(*self.pitch_shift_range)

        # Use librosa pitch shift
        shifted = librosa.effects.pitch_shift(
            y=audio,
            sr=self.sample_rate,
            n_steps=n_steps,
        )

        return shifted

    def time_stretch(
        self,
        audio: np.ndarray,
        rate: Optional[float] = None,
    ) -> np.ndarray:
        """
        Stretch or compress audio in time.

        Args:
            audio: Input audio
            rate: Stretch rate (random if None). >1 speeds up, <1 slows down

        Returns:
            Time-stretched audio
        """
        if rate is None:
            rate = np.random.uniform(*self.time_stretch_range)

        # Use librosa time stretch
        stretched = librosa.effects.time_stretch(y=audio, rate=rate)

        return stretched

    def add_noise(
        self,
        audio: np.ndarray,
        noise_level: Optional[float] = None,
    ) -> np.ndarray:
        """
        Add Gaussian white noise to audio.

        Args:
            audio: Input audio
            noise_level: Noise standard deviation (uses default if None)

        Returns:
            Audio with added noise
        """
        if noise_level is None:
            noise_level = self.noise_level

        # Generate Gaussian noise
        noise = np.random.normal(0, noise_level, audio.shape)

        # Add to audio
        noisy_audio = audio + noise

        # Clip to prevent overflow
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

        return noisy_audio

    def random_crop_audio(
        self,
        audio: np.ndarray,
        crop_ratio: float = 0.9,
    ) -> np.ndarray:
        """
        Randomly crop audio to a fraction of its length.

        Args:
            audio: Input audio
            crop_ratio: Ratio of audio to keep (0 to 1)

        Returns:
            Cropped audio
        """
        crop_length = int(len(audio) * crop_ratio)

        if crop_length >= len(audio):
            return audio

        # Random start position
        max_start = len(audio) - crop_length
        start = np.random.randint(0, max_start + 1)

        cropped = audio[start:start + crop_length]

        return cropped

    def augment(
        self,
        audio: np.ndarray,
        apply_pitch_shift: bool = True,
        apply_time_stretch: bool = True,
        apply_noise: bool = True,
        apply_crop: bool = False,
    ) -> np.ndarray:
        """
        Apply multiple augmentation techniques.

        Args:
            audio: Input audio
            apply_pitch_shift: Whether to apply pitch shifting
            apply_time_stretch: Whether to apply time stretching
            apply_noise: Whether to add noise
            apply_crop: Whether to apply random cropping

        Returns:
            Augmented audio
        """
        augmented = audio.copy()

        # Apply augmentations in sequence
        if apply_pitch_shift and np.random.rand() > 0.5:
            augmented = self.pitch_shift(augmented)

        if apply_time_stretch and np.random.rand() > 0.5:
            augmented = self.time_stretch(augmented)

        if apply_noise and np.random.rand() > 0.5:
            augmented = self.add_noise(augmented)

        if apply_crop and np.random.rand() > 0.5:
            augmented = self.random_crop_audio(augmented)

        return augmented


class SpecAugment:
    """
    SpecAugment for feature-level augmentation.
    Implements frequency masking and time masking on spectrograms.
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 20,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        """
        Args:
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
            n_freq_masks: Number of frequency masks
            n_time_masks: Number of time masks
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def freq_mask(
        self,
        spec: np.ndarray,
        mask_param: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply frequency masking to spectrogram.

        Args:
            spec: Spectrogram with shape [freq, time]
            mask_param: Maximum mask width

        Returns:
            Masked spectrogram
        """
        if mask_param is None:
            mask_param = self.freq_mask_param

        spec_masked = spec.copy()
        n_freq = spec.shape[0]

        # Random mask width
        f = np.random.randint(0, mask_param)

        # Random start position
        f0 = np.random.randint(0, n_freq - f)

        # Apply mask
        spec_masked[f0:f0 + f, :] = 0

        return spec_masked

    def time_mask(
        self,
        spec: np.ndarray,
        mask_param: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply time masking to spectrogram.

        Args:
            spec: Spectrogram with shape [freq, time]
            mask_param: Maximum mask width

        Returns:
            Masked spectrogram
        """
        if mask_param is None:
            mask_param = self.time_mask_param

        spec_masked = spec.copy()
        n_time = spec.shape[1]

        # Random mask width
        t = np.random.randint(0, mask_param)

        # Random start position
        t0 = np.random.randint(0, n_time - t)

        # Apply mask
        spec_masked[:, t0:t0 + t] = 0

        return spec_masked

    def augment(self, spec: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment (frequency and time masking).

        Args:
            spec: Spectrogram with shape [freq, time]

        Returns:
            Augmented spectrogram
        """
        augmented = spec.copy()

        # Apply frequency masks
        for _ in range(self.n_freq_masks):
            augmented = self.freq_mask(augmented)

        # Apply time masks
        for _ in range(self.n_time_masks):
            augmented = self.time_mask(augmented)

        return augmented


class AugmentationTransform:
    """
    PyTorch-compatible transform for data augmentation.
    """

    def __init__(
        self,
        audio_augment: Optional[AudioAugmentation] = None,
        spec_augment: Optional[SpecAugment] = None,
        apply_prob: float = 0.5,
    ):
        """
        Args:
            audio_augment: Audio augmentation instance
            spec_augment: Spectrogram augmentation instance
            apply_prob: Probability of applying augmentation
        """
        self.audio_augment = audio_augment
        self.spec_augment = spec_augment
        self.apply_prob = apply_prob

    def __call__(self, x):
        """
        Apply augmentation transform.

        Args:
            x: Input (audio or spectrogram)

        Returns:
            Augmented input
        """
        if np.random.rand() > self.apply_prob:
            return x

        # Check if input is audio (1D) or spectrogram (2D)
        if isinstance(x, torch.Tensor):
            x_np = x.numpy()
        else:
            x_np = x

        if x_np.ndim == 1:
            # Audio augmentation
            if self.audio_augment:
                x_aug = self.audio_augment.augment(x_np)
            else:
                x_aug = x_np
        elif x_np.ndim == 2:
            # Spectrogram augmentation
            if self.spec_augment:
                x_aug = self.spec_augment.augment(x_np)
            else:
                x_aug = x_np
        else:
            x_aug = x_np

        # Convert back to tensor if needed
        if isinstance(x, torch.Tensor):
            return torch.from_numpy(x_aug).float()
        else:
            return x_aug
