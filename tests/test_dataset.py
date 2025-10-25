"""
Unit tests for dataset loader.
"""

import pytest
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from raag_identifier.data import RaagDataset, create_data_splits


@pytest.fixture
def sample_audio():
    """Generate sample audio data."""
    duration = 2.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return audio, sr


@pytest.fixture
def temp_dataset_dir(sample_audio):
    """Create temporary dataset directory with sample files."""
    audio, sr = sample_audio

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create sample files for each raag
        files = [
            'Yaman_001.wav',
            'Yaman_002.wav',
            'Bhairav_001.wav',
            'Bhairav_002.wav',
            'PuriyaDhanashree_001.wav',
            'PuriyaDhanashree_002.wav',
        ]

        for filename in files:
            filepath = tmpdir / filename
            sf.write(filepath, audio, sr)

        yield str(tmpdir)


def test_label_extraction():
    """Test label extraction from filenames."""
    dataset = RaagDataset.__new__(RaagDataset)

    assert dataset._extract_label_from_filename('Yaman_001.wav') == 'Yaman'
    assert dataset._extract_label_from_filename('Bhairav_vocal_02.mp3') == 'Bhairav'
    assert dataset._extract_label_from_filename('PuriyaDhanashree_05.flac') == 'Puriya_Dhanashree'
    assert dataset._extract_label_from_filename('unknown_file.wav') is None


def test_dataset_loading(temp_dataset_dir):
    """Test dataset loading."""
    dataset = RaagDataset(data_dir=temp_dataset_dir, mode='train')

    assert len(dataset) == 6
    assert len(dataset.unique_labels) == 3
    assert set(dataset.unique_labels) == {'Yaman', 'Bhairav', 'Puriya_Dhanashree'}


def test_dataset_getitem(temp_dataset_dir):
    """Test getting items from dataset."""
    dataset = RaagDataset(data_dir=temp_dataset_dir, mode='train')

    item = dataset[0]

    assert 'feature' in item
    assert 'label' in item
    assert 'label_idx' in item
    assert 'file_path' in item

    # Check types
    assert isinstance(item['feature'].numpy(), np.ndarray)
    assert isinstance(item['label'], str)
    assert isinstance(item['label_idx'], int)


def test_label_distribution(temp_dataset_dir):
    """Test label distribution calculation."""
    dataset = RaagDataset(data_dir=temp_dataset_dir, mode='train')

    distribution = dataset.get_label_distribution()

    assert len(distribution) == 3
    assert distribution['Yaman'] == 2
    assert distribution['Bhairav'] == 2
    assert distribution['Puriya_Dhanashree'] == 2


def test_data_splits(temp_dataset_dir):
    """Test train/val/test split creation."""
    train_files, val_files, test_files = create_data_splits(
        data_dir=temp_dataset_dir,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=42
    )

    # Check split sizes (approximately)
    assert len(train_files) >= 3  # ~60% of 6
    assert len(val_files) >= 1    # ~20% of 6
    assert len(test_files) >= 1   # ~20% of 6

    # Check no overlap
    all_files = set(train_files + val_files + test_files)
    assert len(all_files) == len(train_files) + len(val_files) + len(test_files)
