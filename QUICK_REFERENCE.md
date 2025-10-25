# Quick Reference Guide

## Your Current Setup

You have 6 audio files in `data/raw/`:
- `yaman_1.mp3`, `yaman_2.mp3`
- `bhairavi_1.mp3`, `bhairavi_2.mp3`
- `puriya_dhanashree_1.mp3`, `puriya_dhanashree_2.mp3`

The system is configured to work with this format directly!

## Quick Commands

### 1. Train a Model (Quick Test)
```bash
# Simple CNN, small config for quick training
python3 train.py --config config/train_config.yaml --epochs 5 --batch-size 2
```

### 2. Run Inference on Your Files
```bash
# After training, run inference
python3 inference.py \
    --model outputs/experiment_1/checkpoints/best_model.pth \
    --input data/raw/yaman_1.mp3 \
    --output predictions.jsonl \
    --config config/train_config.yaml
```

### 3. Batch Inference on All Files
```bash
python3 inference.py \
    --model outputs/experiment_1/checkpoints/best_model.pth \
    --input data/raw \
    --output predictions.jsonl \
    --config config/train_config.yaml
```

## File Naming Convention

The system recognizes your naming format automatically:

**Format:** `{raag_name}_{number}.{extension}`

**Supported raag names** (case-insensitive):
- `yaman`
- `bhairav` or `bhairavi` (both recognized as "Bhairav")
- `puriya_dhanashree` or `puriyadhanashree`

**Supported extensions:** `.mp3`, `.wav`, `.flac`

## Training with Your 6 Files

Since you only have 6 files, the system will auto-split them:
- **Train:** ~4 files (70%)
- **Validation:** ~1 file (15%)
- **Test:** ~1 file (15%)

### Recommended Settings for Small Dataset

Edit `config/train_config.yaml`:
```yaml
training:
  epochs: 20              # More epochs for small dataset
  batch_size: 2           # Small batch for small dataset
  early_stopping_patience: 10

model:
  type: "simple"          # Use Simple CNN (faster, less prone to overfitting)
  dropout: 0.5            # Higher dropout to prevent overfitting

augmentation:
  enabled: true           # Important! Helps with small dataset
```

## Expected Workflow

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train model:**
   ```bash
   python3 train.py --config config/train_config.yaml
   ```

3. **View training progress:**
   ```bash
   # In another terminal
   tensorboard --logdir outputs/experiment_1/logs
   # Then open http://localhost:6006
   ```

4. **Evaluate:**
   ```bash
   python3 evaluate.py \
       --model outputs/experiment_1/checkpoints/best_model.pth \
       --data-dir data/raw \
       --output-dir outputs/evaluation
   ```

5. **Run inference:**
   ```bash
   python3 inference.py \
       --model outputs/experiment_1/checkpoints/best_model.pth \
       --input data/raw/yaman_1.mp3 \
       --output predictions.jsonl
   ```

## Important Notes for Small Dataset

⚠️ **With only 6 files (2 per raag), results may be limited:**
- Training will work, but accuracy may not be optimal
- Recommend collecting at least 20-30 files per raag for better results
- Data augmentation is crucial with small datasets
- Use Simple CNN model (avoid overfitting)
- Higher dropout (0.5) helps prevent memorization

## Adding More Files

To improve the model, add more files to `data/raw/`:
```bash
data/raw/
├── yaman_1.mp3
├── yaman_2.mp3
├── yaman_3.mp3        # Add more Yaman recordings
├── bhairavi_1.mp3
├── bhairavi_2.mp3
├── bhairavi_3.mp3     # Add more Bhairavi recordings
├── puriya_dhanashree_1.mp3
├── puriya_dhanashree_2.mp3
├── puriya_dhanashree_3.mp3  # Add more Puriya Dhanashree
└── ...
```

The system will automatically recognize new files with the same naming pattern.

## Output Files

After training, you'll find:
- **Model checkpoint:** `outputs/experiment_1/checkpoints/best_model.pth`
- **Training logs:** `outputs/experiment_1/logs/`
- **Training history:** `outputs/experiment_1/training_history.json`
- **Plots:** `outputs/experiment_1/training_history.png`

After evaluation:
- **Metrics:** `outputs/evaluation/test_metrics.json`
- **Confusion matrix:** `outputs/evaluation/test_confusion_matrix.png`
- **Per-class metrics:** `outputs/evaluation/test_metrics.csv`

After inference:
- **Predictions:** `predictions.jsonl` (one JSON object per line)

## Example Prediction Output

```json
{"file": "data/raw/yaman_1.mp3", "segment_start": 0.0, "segment_end": 5.0, "prediction": "Yaman", "confidence": 0.92}
{"file": "data/raw/yaman_1.mp3", "segment_start": 2.5, "segment_end": 7.5, "prediction": "Yaman", "confidence": 0.89}
```

## Troubleshooting

**Issue:** "No valid audio files found"
- Check file names match pattern: `{raag}__{number}.{ext}`
- Ensure files are in `data/raw/`

**Issue:** "Out of memory"
- Reduce `batch_size` to 1 or 2
- Use `model: type: "simple"` instead of "resnet"

**Issue:** "Poor accuracy"
- Add more training files (recommended: 20-30 per raag)
- Enable data augmentation
- Train for more epochs
- Use higher dropout

## Next Steps

1. **Start with what you have:** Train on your 6 files to test the system
2. **Collect more data:** Add more recordings to improve accuracy
3. **Experiment:** Try different models and hyperparameters
4. **Deploy:** Use the trained model for real predictions

---

For full documentation, see [README.md](README.md)
