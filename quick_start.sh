#!/bin/bash
# Quick start script for Raag Identifier

set -e

echo "======================================================================"
echo "RAAG IDENTIFIER - QUICK START"
echo "======================================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Generate sample dataset
echo ""
echo "Generating sample dataset for testing..."
python generate_sample_dataset.py --output-dir data/raw --n-samples 20

# Run tests
echo ""
echo "Running unit tests..."
pytest tests/ -v

# Train a small model
echo ""
echo "Training a simple model (this may take a few minutes)..."
echo "Using Simple CNN for quick training..."

# Create a quick training config
mkdir -p config
cat > config/quick_start_config.yaml << EOL
random_seed: 42
data:
  train_dir: "data/raw/train"
  val_dir: "data/raw/val"
  test_dir: "data/raw/test"
  processed_dir: "data/processed"
audio:
  sample_rate: 22050
  target_duration: 5.0
features:
  use_cqt: true
  n_bins: 84
  bins_per_octave: 12
  hop_length: 512
  include_delta: false
  include_delta_delta: false
segmentation:
  segment_duration: 5.0
  overlap_duration: 2.5
  normalize: true
  normalization_method: "peak"
vad:
  enabled: true
  use_webrtc: false
  energy_threshold: 0.02
  use_harmonic: true
  harmonic_threshold: 0.5
  min_segment_length: 0.5
augmentation:
  enabled: false
model:
  type: "simple"
  dropout: 0.3
training:
  epochs: 10
  batch_size: 4
  learning_rate: 0.001
  weight_decay: 0.0001
  num_workers: 0
  lr_patience: 3
  early_stopping_patience: 5
  grad_clip: 1.0
validation:
  val_frequency: 1
  save_best_only: true
cross_validation:
  enabled: false
output_dir: "outputs/quick_start"
logging:
  use_tensorboard: true
  log_interval: 10
EOL

python train.py --config config/quick_start_config.yaml

# Evaluate
echo ""
echo "Evaluating model..."
python evaluate.py \
    --model outputs/quick_start/checkpoints/best_model.pth \
    --data-dir data/raw/test \
    --output-dir outputs/quick_start/evaluation

# Run inference
echo ""
echo "Running inference on test data..."
python inference.py \
    --model outputs/quick_start/checkpoints/best_model.pth \
    --input data/raw/test \
    --output outputs/quick_start/predictions.jsonl \
    --config config/quick_start_config.yaml

echo ""
echo "======================================================================"
echo "QUICK START COMPLETE!"
echo "======================================================================"
echo ""
echo "Results:"
echo "  - Model saved: outputs/quick_start/checkpoints/best_model.pth"
echo "  - Evaluation metrics: outputs/quick_start/evaluation/"
echo "  - Predictions: outputs/quick_start/predictions.jsonl"
echo "  - Training logs: outputs/quick_start/logs/"
echo ""
echo "Next steps:"
echo "  1. View training progress: tensorboard --logdir outputs/quick_start/logs"
echo "  2. Check evaluation metrics in outputs/quick_start/evaluation/"
echo "  3. For better results, train on real raag recordings"
echo "  4. Experiment with different models (resnet, crnn) in config/train_config.yaml"
echo ""
echo "======================================================================"
