"""
Generate synthetic sample dataset for smoke testing.
Creates labeled audio files with characteristic frequencies for each raag.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import argparse


# Characteristic frequency patterns for each raag (simplified)
# In reality, raags are defined by note relationships, but this is for testing
RAAG_PATTERNS = {
    'Yaman': {
        # Yaman: Major scale-like, happy mood
        'frequencies': [261.63, 293.66, 329.63, 369.99, 415.30, 466.16],  # C D E F# G A
        'weights': [1.0, 0.8, 1.0, 0.7, 0.9, 0.8],
    },
    'Bhairav': {
        # Bhairav: Serious, morning raag with distinctive intervals
        'frequencies': [261.63, 277.18, 329.63, 349.23, 392.00, 415.30],  # C Db E F G Ab
        'weights': [1.0, 0.9, 0.8, 0.7, 0.9, 0.7],
    },
    'Puriya_Dhanashree': {
        # Puriya Dhanashree: Evening raag
        'frequencies': [261.63, 277.18, 311.13, 369.99, 392.00, 466.16],  # C Db Eb F# G A
        'weights': [1.0, 0.8, 0.7, 0.9, 0.8, 0.9],
    },
}


def generate_raag_audio(
    raag_name: str,
    duration: float = 10.0,
    sample_rate: int = 22050,
    add_noise: bool = True,
    noise_level: float = 0.05,
) -> np.ndarray:
    """
    Generate synthetic audio with characteristic frequencies of a raag.

    Args:
        raag_name: Name of raag ('Yaman', 'Bhairav', 'Puriya_Dhanashree')
        duration: Duration in seconds
        sample_rate: Sample rate
        add_noise: Whether to add noise
        noise_level: Level of noise to add

    Returns:
        Audio signal
    """
    if raag_name not in RAAG_PATTERNS:
        raise ValueError(f"Unknown raag: {raag_name}")

    pattern = RAAG_PATTERNS[raag_name]
    frequencies = pattern['frequencies']
    weights = pattern['weights']

    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate composite signal
    signal = np.zeros_like(t)

    for freq, weight in zip(frequencies, weights):
        # Add some vibrato and temporal variation
        vibrato_rate = np.random.uniform(4, 7)  # Hz
        vibrato_depth = np.random.uniform(0.01, 0.03)  # fraction

        # Modulated frequency
        freq_modulated = freq * (1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t))

        # Amplitude envelope (fade in/out)
        envelope = np.ones_like(t)
        fade_samples = int(0.1 * sample_rate)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        # Random amplitude variation
        amp_variation = 1 + 0.1 * np.sin(2 * np.pi * np.random.uniform(0.1, 0.5) * t)

        # Add component
        component = weight * amp_variation * envelope * np.sin(2 * np.pi * freq_modulated * t)
        signal += component

    # Normalize
    signal = signal / np.max(np.abs(signal))

    # Add harmonics for richness
    for i in range(1, 4):
        harmonic_freq = frequencies[0] * (i + 1)
        harmonic_weight = 0.3 / (i + 1)
        signal += harmonic_weight * np.sin(2 * np.pi * harmonic_freq * t)

    # Normalize again
    signal = signal / np.max(np.abs(signal))

    # Add noise
    if add_noise:
        noise = np.random.normal(0, noise_level, signal.shape)
        signal = signal + noise
        signal = np.clip(signal, -1.0, 1.0)

    # Add some silence at the beginning and end (for VAD testing)
    silence_duration = 0.5  # seconds
    silence = np.zeros(int(silence_duration * sample_rate))
    signal = np.concatenate([silence, signal, silence])

    return signal


def generate_dataset(
    output_dir: str,
    n_samples_per_raag: int = 10,
    duration_range: tuple = (8.0, 15.0),
    sample_rate: int = 22050,
):
    """
    Generate complete sample dataset.

    Args:
        output_dir: Directory to save dataset
        n_samples_per_raag: Number of samples per raag
        duration_range: Range of durations (min, max) in seconds
        sample_rate: Sample rate
    """
    output_path = Path(output_dir)

    # Create train/val/test splits
    for split in ['train', 'val', 'test']:
        split_dir = output_path / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Determine number of samples for this split
        if split == 'train':
            n_samples = int(n_samples_per_raag * 0.7)
        elif split == 'val':
            n_samples = int(n_samples_per_raag * 0.15)
        else:  # test
            n_samples = int(n_samples_per_raag * 0.15)

        # Generate samples for each raag
        for raag_name in RAAG_PATTERNS.keys():
            for i in range(n_samples):
                # Random duration
                duration = np.random.uniform(*duration_range)

                # Generate audio
                audio = generate_raag_audio(
                    raag_name=raag_name,
                    duration=duration,
                    sample_rate=sample_rate,
                    add_noise=True,
                )

                # Save file
                # Use various formats
                formats = ['.wav', '.mp3', '.flac']
                ext = formats[i % len(formats)]

                # Use lowercase with underscores (matching real dataset format)
                raag_name_lower = raag_name.lower().replace(' ', '_')
                filename = f"{raag_name_lower}_{i+1}{ext}"
                filepath = split_dir / filename

                sf.write(filepath, audio, sample_rate)

        print(f"Generated {split} split with {n_samples * len(RAAG_PATTERNS)} samples")

    print(f"\nDataset generated successfully in {output_path}")
    print(f"Total samples: {n_samples_per_raag * len(RAAG_PATTERNS)}")
    print(f"  Train: {int(n_samples_per_raag * 0.7) * len(RAAG_PATTERNS)}")
    print(f"  Val: {int(n_samples_per_raag * 0.15) * len(RAAG_PATTERNS)}")
    print(f"  Test: {int(n_samples_per_raag * 0.15) * len(RAAG_PATTERNS)}")


def main():
    parser = argparse.ArgumentParser(description='Generate sample Raag dataset')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                        help='Output directory for dataset')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Number of samples per raag (total across all splits)')
    parser.add_argument('--min-duration', type=float, default=8.0,
                        help='Minimum audio duration in seconds')
    parser.add_argument('--max-duration', type=float, default=15.0,
                        help='Maximum audio duration in seconds')
    parser.add_argument('--sample-rate', type=int, default=22050,
                        help='Audio sample rate')

    args = parser.parse_args()

    print("="*60)
    print("SYNTHETIC RAAG DATASET GENERATOR")
    print("="*60)
    print(f"\nGenerating dataset with:")
    print(f"  Raags: {', '.join(RAAG_PATTERNS.keys())}")
    print(f"  Samples per raag: {args.n_samples}")
    print(f"  Duration range: {args.min_duration}s - {args.max_duration}s")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print(f"  Output directory: {args.output_dir}")
    print()

    generate_dataset(
        output_dir=args.output_dir,
        n_samples_per_raag=args.n_samples,
        duration_range=(args.min_duration, args.max_duration),
        sample_rate=args.sample_rate,
    )

    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)
    print("\nTo train a model with this dataset, run:")
    print(f"  python train.py --config config/train_config.yaml")
    print("\nNote: This is synthetic data for testing purposes only.")
    print("For real raag classification, use actual recordings.")
    print("="*60)


if __name__ == '__main__':
    main()
