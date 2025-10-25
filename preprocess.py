"""
Preprocessing script for Raag audio dataset.
Applies VAD, feature extraction, and saves processed data.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from raag_identifier.preprocessing import (
    VoiceActivityDetector,
    FeatureExtractor,
    AudioSegmenter
)


def preprocess_audio_file(
    input_path: Path,
    output_dir: Path,
    config: dict,
    apply_vad: bool = True,
    extract_features: bool = True,
    save_audio: bool = False,
):
    """
    Preprocess a single audio file.

    Args:
        input_path: Path to input audio file
        output_dir: Directory to save processed files
        config: Configuration dictionary
        apply_vad: Whether to apply VAD
        extract_features: Whether to extract and save features
        save_audio: Whether to save processed audio
    """
    # Load audio
    sample_rate = config['audio']['sample_rate']
    audio, sr = librosa.load(input_path, sr=sample_rate, mono=True)

    # Apply VAD
    if apply_vad and config['vad']['enabled']:
        vad = VoiceActivityDetector(
            sample_rate=sample_rate,
            use_webrtc=config['vad']['use_webrtc'],
            energy_threshold=config['vad']['energy_threshold'],
        )
        audio = vad.remove_silence(
            audio,
            use_harmonic=config['vad']['use_harmonic'],
            min_segment_length=config['vad']['min_segment_length'],
        )

    # Save processed audio if requested
    if save_audio:
        audio_output_dir = output_dir / 'audio'
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_output_dir / f"{input_path.stem}_processed.npy"
        np.save(audio_path, audio)

    # Extract features
    if extract_features:
        feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            use_cqt=config['features']['use_cqt'],
            n_bins=config['features']['n_bins'],
            bins_per_octave=config['features']['bins_per_octave'],
        )

        features = feature_extractor.extract_features(
            audio,
            include_delta=config['features']['include_delta'],
            include_delta_delta=config['features']['include_delta_delta'],
        )

        # Save features
        features_output_dir = output_dir / 'features'
        features_output_dir.mkdir(parents=True, exist_ok=True)
        features_path = features_output_dir / f"{input_path.stem}.npy"
        np.save(features_path, features)

        return features.shape

    return None


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    config: dict,
    apply_vad: bool = True,
    extract_features: bool = True,
    save_audio: bool = False,
):
    """
    Preprocess entire dataset.

    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save processed files
        config: Configuration dictionary
        apply_vad: Whether to apply VAD
        extract_features: Whether to extract features
        save_audio: Whether to save processed audio
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_path.rglob(f'*{ext}'))

    if len(audio_files) == 0:
        print(f"No audio files found in {input_dir}")
        return

    print(f"Found {len(audio_files)} audio files")

    # Process files
    stats = {
        'total_files': len(audio_files),
        'processed_files': 0,
        'failed_files': 0,
        'feature_shapes': [],
    }

    for audio_file in tqdm(audio_files, desc='Processing files'):
        try:
            # Determine output subdirectory (preserve relative structure)
            rel_path = audio_file.relative_to(input_path)
            file_output_dir = output_path / rel_path.parent

            # Preprocess
            shape = preprocess_audio_file(
                input_path=audio_file,
                output_dir=file_output_dir,
                config=config,
                apply_vad=apply_vad,
                extract_features=extract_features,
                save_audio=save_audio,
            )

            if shape is not None:
                stats['feature_shapes'].append(shape)

            stats['processed_files'] += 1

        except Exception as e:
            print(f"\nError processing {audio_file}: {e}")
            stats['failed_files'] += 1

    # Save statistics
    stats_path = output_path / 'preprocessing_stats.json'

    # Convert numpy arrays to lists for JSON serialization
    stats['feature_shapes'] = [list(shape) for shape in stats['feature_shapes']]

    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Total files: {stats['total_files']}")
    print(f"Processed: {stats['processed_files']}")
    print(f"Failed: {stats['failed_files']}")
    if stats['feature_shapes']:
        print(f"Feature shape range: {min(stats['feature_shapes'])} to {max(stats['feature_shapes'])}")
    print(f"\nResults saved to: {output_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Preprocess Raag audio dataset')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing audio files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for processed files')
    parser.add_argument('--no-vad', action='store_true',
                        help='Disable voice activity detection')
    parser.add_argument('--no-features', action='store_true',
                        help='Skip feature extraction')
    parser.add_argument('--save-audio', action='store_true',
                        help='Save processed audio files')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Preprocess
    preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config=config,
        apply_vad=not args.no_vad,
        extract_features=not args.no_features,
        save_audio=args.save_audio,
    )


if __name__ == '__main__':
    main()
