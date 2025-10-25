"""
Dataset loader for Raag classification.
Extracts labels from filenames and supports multiple audio formats.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
import soundfile as sf


class RaagDataset(Dataset):
    """
    Dataset for Raag audio classification.

    Expects audio files with raag labels in filenames (case-insensitive):
    - yaman_1.mp3, yaman_2.wav
    - bhairavi_1.mp3, bhairavi_2.flac
    - puriya_dhanashree_1.mp3, puriya_dhanashree_2.wav

    Also supports legacy formats:
    - Yaman_001.wav, Bhairav_vocal_02.mp3, PuriyaDhanashree_05.flac
    """

    SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac']
    RAAG_LABELS = ['yaman', 'bhairav', 'bhairavi', 'puriya_dhanashree', 'puriyadhanashree']
    # Normalize label variants to canonical forms
    LABEL_MAP = {
        'yaman': 'Yaman',
        'bhairav': 'Bhairav',
        'bhairavi': 'Bhairav',  # Common alternate spelling
        'puriya_dhanashree': 'Puriya_Dhanashree',
        'puriyadhanashree': 'Puriya_Dhanashree',
        'puriya dhanashree': 'Puriya_Dhanashree',
    }

    def __init__(
        self,
        data_dir: str,
        feature_dir: Optional[str] = None,
        mode: str = 'train',
        transform=None,
        target_sr: int = 22050,
    ):
        """
        Args:
            data_dir: Directory containing audio files
            feature_dir: Directory containing preprocessed features (.npy files)
            mode: 'train', 'val', or 'test'
            transform: Optional transforms to apply
            target_sr: Target sample rate for audio
        """
        self.data_dir = Path(data_dir)
        self.feature_dir = Path(feature_dir) if feature_dir else None
        self.mode = mode
        self.transform = transform
        self.target_sr = target_sr

        # Load file list and labels
        self.samples = self._load_samples()

        # Create label to index mapping
        self.unique_labels = sorted(set(self.LABEL_MAP.values()))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def _extract_label_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract raag label from filename.

        Args:
            filename: Audio filename

        Returns:
            Normalized raag label or None if not found
        """
        filename_lower = filename.lower()

        # Try to match known raag labels
        for label in self.RAAG_LABELS:
            pattern = re.escape(label.lower())
            if re.search(pattern, filename_lower):
                return self.LABEL_MAP.get(label.lower(), label)

        # Try additional patterns
        for key, value in self.LABEL_MAP.items():
            if key in filename_lower:
                return value

        return None

    def _load_samples(self) -> List[Tuple[Path, str]]:
        """
        Load all audio files with valid labels.

        Returns:
            List of (file_path, label) tuples
        """
        samples = []

        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        # Recursively find all audio files
        for ext in self.SUPPORTED_FORMATS:
            for audio_file in self.data_dir.rglob(f"*{ext}"):
                label = self._extract_label_from_filename(audio_file.name)
                if label:
                    samples.append((audio_file, label))

        if len(samples) == 0:
            raise ValueError(f"No valid audio files found in {self.data_dir}")

        print(f"Loaded {len(samples)} samples for {self.mode} mode")
        return samples

    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Use librosa for loading (handles various formats)
            audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {file_path}: {e}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'feature', 'label', 'label_idx', 'file_path'
        """
        file_path, label = self.samples[idx]

        # Load preprocessed features if available
        if self.feature_dir:
            feature_file = self.feature_dir / f"{file_path.stem}.npy"
            if feature_file.exists():
                feature = np.load(feature_file)
            else:
                # Fall back to loading raw audio
                audio, sr = self.load_audio(file_path)
                feature = audio
        else:
            # Load raw audio
            audio, sr = self.load_audio(file_path)
            feature = audio

        # Apply transforms if provided
        if self.transform:
            feature = self.transform(feature)

        # Convert to tensor if numpy array
        if isinstance(feature, np.ndarray):
            feature = torch.from_numpy(feature).float()

        return {
            'feature': feature,
            'label': label,
            'label_idx': self.label_to_idx[label],
            'file_path': str(file_path),
        }

    def get_label_distribution(self) -> Dict[str, int]:
        """
        Get distribution of labels in dataset.

        Returns:
            Dictionary mapping label to count
        """
        label_counts = {}
        for _, label in self.samples:
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts


def create_data_splits(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split dataset into train/val/test sets.

    Args:
        data_dir: Directory containing audio files
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    from sklearn.model_selection import train_test_split

    # Create temporary dataset to get all files
    temp_dataset = RaagDataset(data_dir, mode='all')
    all_samples = temp_dataset.samples

    # Extract files and labels
    files = [str(path) for path, _ in all_samples]
    labels = [label for _, label in all_samples]

    # First split: train vs (val + test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        files, labels,
        test_size=(val_ratio + test_ratio),
        random_state=random_seed,
        stratify=labels
    )

    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(
        temp_files,
        test_size=(1 - val_ratio_adjusted),
        random_state=random_seed,
        stratify=temp_labels
    )

    print(f"Data split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    return train_files, val_files, test_files
