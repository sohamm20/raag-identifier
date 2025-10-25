"""
Training script for Raag classification model.
"""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from raag_identifier.data import RaagDataset
from raag_identifier.models import create_model, create_crnn_model
from raag_identifier.preprocessing import FeatureExtractor, AudioSegmenter
from raag_identifier.utils import MetricsCalculator, plot_training_history


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_batch(batch, feature_extractor, device):
    """
    Preprocess a batch of audio data.

    Args:
        batch: Batch from dataloader
        feature_extractor: FeatureExtractor instance
        device: Torch device

    Returns:
        features, labels
    """
    features_list = []
    labels = []

    for item in batch:
        audio = item['feature'].numpy()
        label = item['label_idx']

        # Extract features if audio is raw
        if audio.ndim == 1:
            features = feature_extractor.extract_features(audio)
        else:
            features = audio

        features_list.append(features)
        labels.append(label)

    # Pad features to same length
    max_time = max(f.shape[-1] for f in features_list)
    padded_features = []

    for features in features_list:
        if features.shape[-1] < max_time:
            pad_width = ((0, 0), (0, max_time - features.shape[-1]))
            features = np.pad(features, pad_width, mode='constant')
        padded_features.append(features)

    # Stack and convert to tensor
    features_tensor = torch.FloatTensor(np.array(padded_features)).to(device)
    labels_tensor = torch.LongTensor(labels).to(device)

    return features_tensor, labels_tensor


def train_epoch(model, dataloader, criterion, optimizer, feature_extractor, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, batch in enumerate(pbar):
        # Get data
        inputs, labels = preprocess_batch(batch, feature_extractor, device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, feature_extractor, device, epoch):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Get data
            inputs, labels = preprocess_batch(batch, feature_extractor, device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Store predictions
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, all_preds, all_labels


def train(config):
    """Main training function."""

    # Set seed
    set_seed(config['random_seed'])

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    output_dir = Path(config['output_dir'])
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tensorboard
    writer = SummaryWriter(log_dir=str(log_dir))

    # Check if we need to create train/val/test splits
    train_dir = Path(config['data']['train_dir'])
    val_dir = Path(config['data']['val_dir'])

    # If train_dir and val_dir are the same, we need to auto-split
    need_split = (train_dir == val_dir)

    if need_split:
        print("Auto-splitting dataset into train/val sets...")
        from raag_identifier.data import create_data_splits

        train_files, val_files, test_files = create_data_splits(
            data_dir=str(train_dir),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=config['random_seed']
        )

        # For now, we'll use all files and split in dataset
        print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    # Create datasets
    print("Loading datasets...")
    train_dataset = RaagDataset(
        data_dir=config['data']['train_dir'],
        mode='train',
        target_sr=config['audio']['sample_rate']
    )

    val_dataset = RaagDataset(
        data_dir=config['data']['val_dir'],
        mode='val',
        target_sr=config['audio']['sample_rate']
    )

    # If auto-split, we need to filter the datasets
    if need_split:
        # Filter train dataset
        train_dataset.samples = [(p, l) for p, l in train_dataset.samples if str(p) in train_files]
        # Filter val dataset
        val_dataset.samples = [(p, l) for p, l in val_dataset.samples if str(p) in val_files]

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create feature extractor
    feature_extractor = FeatureExtractor(
        sample_rate=config['audio']['sample_rate'],
        use_cqt=config['features']['use_cqt'],
        n_bins=config['features']['n_bins'],
        bins_per_octave=config['features']['bins_per_octave'],
    )

    # Create model
    print(f"Creating model: {config['model']['type']}")
    n_classes = len(train_dataset.unique_labels)

    if config['model']['type'] in ['simple', 'resnet']:
        model = create_model(
            model_type=config['model']['type'],
            n_classes=n_classes,
            dropout=config['model']['dropout']
        )
    else:
        model = create_crnn_model(
            model_type=config['model']['type'],
            n_classes=n_classes,
            dropout=config['model']['dropout']
        )

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config['training']['lr_patience'],
        verbose=True
    )

    # Training loop
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }

    early_stop_counter = 0

    print("\nStarting training...")
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*60}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            feature_extractor, device, epoch
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion,
            feature_extractor, device, epoch
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config,
            }, checkpoint_path)
            print(f"  Best model saved! Val Acc: {val_acc:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= config['training']['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    # Save final model
    final_checkpoint = checkpoint_dir / 'final_model.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, final_checkpoint)

    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Plot training history
    plot_path = output_dir / 'training_history.png'
    plot_training_history(history, save_path=str(plot_path))

    writer.close()

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Models saved to: {checkpoint_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Train Raag classifier')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, help='Override data directory')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line arguments
    if args.data_dir:
        config['data']['train_dir'] = args.data_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # Train
    train(config)


if __name__ == '__main__':
    main()
