#!/usr/bin/env python3
"""
Split audio files into 5-second segments for data augmentation.
"""

import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def split_audio_file(input_path, output_dir, segment_duration=5.0, overlap_duration=0.0, sample_rate=22050):
    """
    Split an audio file into segments.

    Args:
        input_path: Path to input audio file
        output_dir: Directory to save segments
        segment_duration: Duration of each segment in seconds
        overlap_duration: Overlap between segments in seconds
        sample_rate: Target sample rate

    Returns:
        Number of segments created
    """
    # Load audio
    audio, sr = librosa.load(input_path, sr=sample_rate, mono=True)

    # Calculate segment parameters
    segment_samples = int(segment_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    hop_samples = segment_samples - overlap_samples

    # Extract raag label from filename
    filename = Path(input_path).stem
    if 'yaman' in filename.lower():
        raag = 'yaman'
    elif 'bhairav' in filename.lower():
        raag = 'bhairavi'  # Using consistent naming
    elif 'puriya' in filename.lower():
        raag = 'puriya_dhanashree'
    else:
        raag = 'unknown'

    # Create output directory for this raag
    raag_dir = Path(output_dir) / raag
    raag_dir.mkdir(parents=True, exist_ok=True)

    segments_created = 0

    # Split into segments
    for i, start in enumerate(range(0, len(audio) - segment_samples + 1, hop_samples)):
        end = start + segment_samples
        segment = audio[start:end]

        # Only save if segment is full length
        if len(segment) == segment_samples:
            output_filename = f"{filename}_segment_{i:04d}.wav"
            output_path = raag_dir / output_filename

            # Save segment
            sf.write(output_path, segment, sample_rate)
            segments_created += 1

    return segments_created


def main():
    parser = argparse.ArgumentParser(description='Split audio files into segments')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--segment-duration', type=float, default=5.0, help='Segment duration in seconds')
    parser.add_argument('--overlap-duration', type=float, default=0.0, help='Overlap duration in seconds')
    parser.add_argument('--sample-rate', type=int, default=22050, help='Sample rate')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_extensions = ['.mp3', '.wav', '.flac']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f'*{ext}'))

    print(f"Found {len(audio_files)} audio files")

    total_segments = 0

    for audio_file in tqdm(audio_files, desc="Processing files"):
        segments = split_audio_file(
            audio_file,
            output_dir,
            args.segment_duration,
            args.overlap_duration,
            args.sample_rate
        )
        total_segments += segments
        print(f"{audio_file.name}: {segments} segments created")

    print(f"\nTotal segments created: {total_segments}")

    # Print summary by raag
    for raag_dir in output_dir.iterdir():
        if raag_dir.is_dir():
            count = len(list(raag_dir.glob('*.wav')))
            print(f"{raag_dir.name}: {count} segments")


if __name__ == '__main__':
    main()