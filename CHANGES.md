# Changes Made for Custom Dataset Format

## Summary

Updated the Raag Identifier to work with your specific file naming convention:
- `yaman_1.mp3`, `yaman_2.mp3`
- `bhairavi_1.mp3`, `bhairavi_2.mp3`
- `puriya_dhanashree_1.mp3`, `puriya_dhanashree_2.mp3`

## Files Modified

### 1. **src/raag_identifier/data/dataset.py**
**Changes:**
- Updated `RAAG_LABELS` to include lowercase variants: `yaman`, `bhairav`, `bhairavi`, `puriya_dhanashree`
- Added `bhairavi` → `Bhairav` mapping (common alternate spelling)
- Updated documentation to show the new naming convention as default
- Maintained backward compatibility with legacy formats

**Before:**
```python
RAAG_LABELS = ['Yaman', 'Bhairav', 'Puriya_Dhanashree']
LABEL_MAP = {
    'yaman': 'Yaman',
    'bhairav': 'Bhairav',
    'puriya_dhanashree': 'Puriya_Dhanashree',
}
```

**After:**
```python
RAAG_LABELS = ['yaman', 'bhairav', 'bhairavi', 'puriya_dhanashree']
LABEL_MAP = {
    'yaman': 'Yaman',
    'bhairav': 'Bhairav',
    'bhairavi': 'Bhairav',  # Common alternate spelling
    'puriya_dhanashree': 'Puriya_Dhanashree',
    'puriyadhanashree': 'Puriya_Dhanashree',
}
```

### 2. **generate_sample_dataset.py**
**Changes:**
- Updated filename generation to use lowercase with underscores
- Changed from `Yaman_train_001.wav` to `yaman_1.wav` format

**Before:**
```python
filename = f"{raag_name}_{split}_{i:03d}{ext}"
```

**After:**
```python
raag_name_lower = raag_name.lower().replace(' ', '_')
filename = f"{raag_name_lower}_{i+1}{ext}"
```

### 3. **README.md**
**Changes:**
- Updated Dataset Preparation section to show lowercase naming as default
- Added clear naming convention documentation
- Simplified structure to match single-directory setup

**New content:**
```markdown
### Dataset Preparation

Organize your audio files with raag labels in filenames (case-insensitive):

data/raw/
├── yaman_1.mp3
├── yaman_2.mp3
├── bhairavi_1.mp3
└── ...

**Naming convention:** `{raag_name}_{number}.{ext}`
- Raag names: `yaman`, `bhairavi`, `puriya_dhanashree`
- Case-insensitive matching
```

### 4. **config/train_config.yaml**
**Changes:**
- Updated default data paths to point to `data/raw` instead of separate train/val/test directories
- Added comments explaining auto-split functionality

**Before:**
```yaml
data:
  train_dir: "data/raw/train"
  val_dir: "data/raw/val"
  test_dir: "data/raw/test"
```

**After:**
```yaml
data:
  train_dir: "data/raw"  # Main data directory
  val_dir: "data/raw"    # Can be same as train_dir for auto-split
  test_dir: "data/raw"   # Can be same as train_dir for auto-split
```

### 5. **train.py**
**Changes:**
- Added auto-split logic when train_dir and val_dir are the same
- Automatically splits all files in `data/raw/` into train/val/test sets
- Maintains reproducibility with random seed

**New code:**
```python
# Check if we need to create train/val/test splits
train_dir = Path(config['data']['train_dir'])
val_dir = Path(config['data']['val_dir'])

# If train_dir and val_dir are the same, we need to auto-split
need_split = (train_dir == val_dir)

if need_split:
    print("Auto-splitting dataset into train/val sets...")
    train_files, val_files, test_files = create_data_splits(...)
    # Filter datasets accordingly
```

## New Files Created

### **QUICK_REFERENCE.md**
- Quick start guide specifically for your 6-file setup
- Commands to run training and inference
- Tips for working with small datasets
- Expected workflow and outputs

## Compatibility

✅ **Backward Compatible:**
- Still supports old naming: `Yaman_001.wav`, `Bhairav_vocal_02.mp3`
- Still works with train/val/test subdirectories
- All existing functionality preserved

✅ **New Default:**
- Primary format: `yaman_1.mp3`, `bhairavi_2.wav`, etc.
- Works directly with files in `data/raw/` (no subdirectories needed)
- Auto-split for train/val/test

## Testing

Your 6 files are correctly recognized:
```
bhairavi_1.mp3         -> Bhairav
bhairavi_2.mp3         -> Bhairav
puriya_dhanashree_1.mp3 -> Puriya_Dhanashree
puriya_dhanashree_2.mp3 -> Puriya_Dhanashree
yaman_1.mp3            -> Yaman
yaman_2.mp3            -> Yaman
```

## Usage

### Quick Start with Your Files

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train (your files will auto-split):**
   ```bash
   python3 train.py --config config/train_config.yaml --epochs 10 --batch-size 2
   ```

3. **Inference on a single file:**
   ```bash
   python3 inference.py \
       --model outputs/experiment_1/checkpoints/best_model.pth \
       --input data/raw/yaman_1.mp3 \
       --output predictions.jsonl
   ```

4. **Batch inference on all files:**
   ```bash
   python3 inference.py \
       --model outputs/experiment_1/checkpoints/best_model.pth \
       --input data/raw \
       --output predictions.jsonl
   ```

## Recommendations for Your Dataset

With only 6 files (2 per raag), consider:

1. **Enable data augmentation** (already enabled in config)
2. **Use Simple CNN model** (less prone to overfitting)
3. **Higher dropout** (0.5 recommended)
4. **More epochs** (20-30) since dataset is small
5. **Small batch size** (2-4)
6. **Add more files** when possible (20-30 per raag recommended)

## Summary

✅ System now works out-of-the-box with your file naming convention
✅ Auto-splits your 6 files into train/val/test
✅ Maintains backward compatibility
✅ Ready to train and run inference immediately

See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for detailed usage instructions.
