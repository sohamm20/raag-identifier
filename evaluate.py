"""
Evaluation script for Raag classification model.
Generates comprehensive metrics and reports.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from raag_identifier.data import RaagDataset
from raag_identifier.models import create_model, create_crnn_model
from raag_identifier.preprocessing import FeatureExtractor
from raag_identifier.utils import MetricsCalculator
from torch.utils.data import DataLoader


def evaluate(config, model_path, data_dir, output_dir):
    """
    Evaluate model on test set.

    Args:
        config: Configuration dictionary
        model_path: Path to model checkpoint
        data_dir: Path to test data
        output_dir: Path to save results
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load test dataset
    print(f"Loading test data from {data_dir}...")
    test_dataset = RaagDataset(
        data_dir=data_dir,
        mode='test',
        target_sr=config['audio']['sample_rate']
    )

    # Custom collate function to handle variable-length audio
    def custom_collate(batch):
        return batch

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate,
    )

    print(f"Test samples: {len(test_dataset)}")

    # Feature extractor
    feature_extractor = FeatureExtractor(
        sample_rate=config['audio']['sample_rate'],
        use_cqt=config['features']['use_cqt'],
        n_bins=config['features']['n_bins'],
        bins_per_octave=config['features']['bins_per_octave'],
    )

    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    # Get model config
    if 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        model_config = config

    # Create model
    n_classes = len(test_dataset.unique_labels)
    model_type = model_config.get('model', {}).get('type', 'simple')

    if model_type in ['simple', 'resnet']:
        model = create_model(
            model_type=model_type,
            n_classes=n_classes,
        )
    else:
        model = create_crnn_model(
            model_type=model_type,
            n_classes=n_classes,
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")

    # Evaluate
    print("\nEvaluating...")
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            # Preprocess
            features_list = []
            labels = []

            for item in batch:
                audio = item['feature'].numpy()
                label = item['label_idx']

                # Extract features
                if audio.ndim == 1:
                    features = feature_extractor.extract_features(audio)
                else:
                    features = audio

                features_list.append(features)
                labels.append(label)

            # Pad features
            max_time = max(f.shape[-1] for f in features_list)
            padded_features = []

            for features in features_list:
                if features.shape[-1] < max_time:
                    pad_width = ((0, 0), (0, max_time - features.shape[-1]))
                    features = np.pad(features, pad_width, mode='constant')
                padded_features.append(features)

            # Convert to tensor
            inputs = torch.FloatTensor(np.array(padded_features)).to(device)
            labels_tensor = torch.LongTensor(labels)

            # Forward pass
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_tensor.numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics_calc = MetricsCalculator(class_names=test_dataset.unique_labels)
    metrics = metrics_calc.calculate_metrics(all_labels, all_preds)

    # Print metrics
    metrics_calc.print_metrics(metrics)

    # Save metrics report
    print("\nSaving results...")
    metrics_calc.save_metrics_report(metrics, output_dir=str(output_path), prefix='test')

    print(f"\nEvaluation complete! Results saved to {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Raag classifier')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation',
                        help='Path to output directory')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Evaluate
    evaluate(
        config=config,
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
