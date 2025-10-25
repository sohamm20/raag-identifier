"""
Inference script for Raag classification.
Accepts single audio file or directory and outputs predictions.
"""

import os
import sys
import argparse
import json
import yaml
import torch
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from raag_identifier.models import create_model, create_crnn_model
from raag_identifier.preprocessing import (
    FeatureExtractor,
    AudioSegmenter,
    VoiceActivityDetector
)


class RaagInference:
    """Raag classifier inference engine."""

    def __init__(
        self,
        model_path: str,
        config: dict,
        device: str = 'cpu',
        use_vad: bool = True,
    ):
        """
        Args:
            model_path: Path to model checkpoint
            config: Configuration dictionary
            device: Device to run on ('cpu' or 'cuda')
            use_vad: Whether to use voice activity detection
        """
        self.device = torch.device(device)
        self.config = config
        self.use_vad = use_vad

        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Get model config
        if 'config' in checkpoint:
            model_config = checkpoint['config']
        else:
            model_config = config

        # Create model
        n_classes = 3  # Yaman, Bhairav, Puriya_Dhanashree
        model_type = model_config.get('model', {}).get('type', 'simple')

        if model_type in ['simple', 'resnet']:
            self.model = create_model(
                model_type=model_type,
                n_classes=n_classes,
            )
        else:
            self.model = create_crnn_model(
                model_type=model_type,
                n_classes=n_classes,
            )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Class labels
        self.class_labels = ['Yaman', 'Bhairav', 'Puriya_Dhanashree']

        # Feature extractor
        audio_config = config.get('audio', {})
        feature_config = config.get('features', {})

        self.sample_rate = audio_config.get('sample_rate', 22050)

        self.feature_extractor = FeatureExtractor(
            sample_rate=self.sample_rate,
            use_cqt=feature_config.get('use_cqt', True),
            n_bins=feature_config.get('n_bins', 84),
            bins_per_octave=feature_config.get('bins_per_octave', 12),
        )

        # Segmenter
        segment_config = config.get('segmentation', {})
        self.segmenter = AudioSegmenter(
            segment_duration=segment_config.get('segment_duration', 5.0),
            overlap_duration=segment_config.get('overlap_duration', 2.5),
            sample_rate=self.sample_rate,
        )

        # VAD
        if use_vad:
            self.vad = VoiceActivityDetector(sample_rate=self.sample_rate)
        else:
            self.vad = None

        print("Model loaded successfully!")

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Preprocessed audio
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Apply VAD if enabled
        if self.vad:
            audio = self.vad.remove_silence(audio)

        return audio

    def predict_segment(self, features: np.ndarray) -> tuple:
        """
        Predict raag for a single feature segment.

        Args:
            features: Feature array

        Returns:
            (predicted_class, confidence)
        """
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)

        predicted_class = self.class_labels[predicted.item()]
        confidence_score = confidence.item()

        return predicted_class, confidence_score

    def predict_file(self, audio_path: str, segment: bool = True) -> list:
        """
        Predict raag for an audio file.

        Args:
            audio_path: Path to audio file
            segment: Whether to segment audio

        Returns:
            List of prediction dictionaries
        """
        # Preprocess audio
        audio = self.preprocess_audio(audio_path)

        results = []

        if segment:
            # Segment audio
            segments = self.segmenter.segment_audio(audio, pad=True)

            for i, seg in enumerate(segments):
                # Extract features
                features = self.feature_extractor.extract_features(seg)

                # Predict
                predicted_class, confidence = self.predict_segment(features)

                # Calculate time boundaries
                segment_start = i * self.segmenter.hop_samples / self.sample_rate
                segment_end = segment_start + self.segmenter.segment_duration

                results.append({
                    'file': str(audio_path),
                    'segment_start': float(segment_start),
                    'segment_end': float(segment_end),
                    'prediction': predicted_class,
                    'confidence': float(confidence),
                })
        else:
            # Single prediction for entire file
            features = self.feature_extractor.extract_features(audio)
            predicted_class, confidence = self.predict_segment(features)

            results.append({
                'file': str(audio_path),
                'segment_start': 0.0,
                'segment_end': len(audio) / self.sample_rate,
                'prediction': predicted_class,
                'confidence': float(confidence),
            })

        return results

    def predict_directory(self, input_dir: str, segment: bool = True) -> list:
        """
        Predict raag for all audio files in a directory.

        Args:
            input_dir: Path to directory
            segment: Whether to segment audio

        Returns:
            List of prediction dictionaries
        """
        input_path = Path(input_dir)
        audio_extensions = ['.wav', '.mp3', '.flac']

        # Find all audio files
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.rglob(f'*{ext}'))

        if len(audio_files) == 0:
            print(f"No audio files found in {input_dir}")
            return []

        print(f"Found {len(audio_files)} audio files")

        # Process files
        all_results = []
        for audio_file in tqdm(audio_files, desc='Processing files'):
            try:
                results = self.predict_file(audio_file, segment=segment)
                all_results.extend(results)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

        return all_results


def main():
    parser = argparse.ArgumentParser(description='Raag Classifier Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to audio file or directory')
    parser.add_argument('--output', type=str, default='predictions.jsonl',
                        help='Path to output file (JSONL format)')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--no-segment', action='store_true',
                        help='Disable audio segmentation (predict entire file)')
    parser.add_argument('--no-vad', action='store_true',
                        help='Disable voice activity detection')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run inference on')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create inference engine
    inference = RaagInference(
        model_path=args.model,
        config=config,
        device=args.device,
        use_vad=not args.no_vad,
    )

    # Run inference
    input_path = Path(args.input)

    if input_path.is_file():
        print(f"Processing file: {input_path}")
        results = inference.predict_file(
            str(input_path),
            segment=not args.no_segment
        )
    elif input_path.is_dir():
        print(f"Processing directory: {input_path}")
        results = inference.predict_directory(
            str(input_path),
            segment=not args.no_segment
        )
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"\nResults saved to {output_path}")

    # Print summary
    if results:
        print(f"\nProcessed {len(results)} segments")
        print("\nSample predictions:")
        for result in results[:5]:
            print(f"  {result['file']} [{result['segment_start']:.1f}s - {result['segment_end']:.1f}s]")
            print(f"    Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")


if __name__ == '__main__':
    main()
