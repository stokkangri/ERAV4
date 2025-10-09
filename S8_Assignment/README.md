# ResNet-50 CIFAR-100 Training Pipeline

A complete, modular training pipeline for training ResNet-50 from scratch on the CIFAR-100 dataset. This implementation includes learning rate finding, OneCycle scheduling, and comprehensive training utilities.

## Project Structure

```
.
├── models/
│   └── resnet50.py           # ResNet-50 model implementation
├── dataset/
│   └── cifar100_loader.py    # CIFAR-100 data loader with augmentation
├── utils/
│   ├── lr_finder.py          # Learning rate finder utility
│   └── train_test.py         # Training and testing functions
├── train_resnet50.py         # Main training script (auto-download)
├── train_binary_data.py      # Training script for binary files
├── train_notebook.py         # Notebook-compatible training module
├── visualize_cifar100.py     # CIFAR-100 visualization tool
└── cifar100_viewer.py        # Simple CIFAR-100 viewer
```

## Features

- **ResNet-50 Architecture**: Modified for CIFAR-100 (32x32 images)
- **Data Augmentation**: Random crops, horizontal flips, color jitter, and Cutout
- **Learning Rate Finder**: Automatic optimal learning rate detection
- **OneCycle Scheduler**: State-of-the-art learning rate scheduling
- **Mixed Training Support**: SGD and Adam optimizers
- **Checkpoint Management**: Save/resume training with best model tracking
- **Comprehensive Logging**: Training curves, metrics, and progress visualization
- **Notebook Support**: Jupyter notebook-friendly interface
- **Binary Data Support**: Direct loading from CIFAR-100 binary files

## Installation

```bash
# Install required packages
pip install torch torchvision matplotlib numpy scipy tqdm pandas

# For LR finder (optional but recommended)
pip install torch-lr-finder
```

## Quick Start

### Command Line Training

#### Option 1: Using Downloaded Binary Files
If you have downloaded CIFAR-100 binary files (train, test, meta):

```bash
# Basic training with binary files
python train_binary_data.py train test --meta meta

# Find optimal learning rate first, then train
python train_binary_data.py train test --meta meta --find-lr --epochs 100

# Custom configuration
python train_binary_data.py train test \
    --meta meta \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.1
```

#### Option 2: Automatic Download
The script will automatically download CIFAR-100 if not present:

```bash
# Basic training with automatic download
python train_resnet50.py

# Find optimal learning rate first, then train
python train_resnet50.py --find-lr --epochs 100

# Custom configuration
python train_resnet50.py \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.1 \
    --scheduler onecycle \
    --weight-decay 5e-4

# Resume from checkpoint
python train_resnet50.py --resume
```

### Notebook Training

```python
# Import the notebook trainer
from train_notebook import NotebookTrainer, quick_train

# Option 1: Quick training with automatic LR finding
trainer = quick_train(epochs=50, find_lr_first=True)

# Option 2: Custom configuration
trainer = NotebookTrainer(
    batch_size=128,
    epochs=100,
    learning_rate=0.1,
    scheduler_type='onecycle'
)
trainer.setup()

# Find optimal learning rate
suggested_lr = trainer.find_lr()

# Train the model
history = trainer.train(plot_interval=5)

# Save the best model
trainer.save_checkpoint('best_model.pth')

# Evaluate on test set
results = trainer.evaluate()
```

### Visualizing CIFAR-100 Data

```bash
# View CIFAR-100 images from binary files
python cifar100_viewer.py train meta 16

# Or use the full-featured visualizer
python visualize_cifar100.py test meta 25
```

## Training Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 128 | Batch size for training |
| `--lr` | 0.1 | Initial learning rate |
| `--momentum` | 0.9 | SGD momentum |
| `--weight-decay` | 5e-4 | Weight decay (L2 penalty) |
| `--scheduler` | onecycle | LR scheduler (onecycle/cosine/step/none) |
| `--max-lr` | 0.1 | Maximum LR for OneCycle scheduler |
| `--find-lr` | False | Find optimal LR before training |
| `--resume` | False | Resume from last checkpoint |
| `--num-workers` | 4 | Number of data loading workers |
| `--no-augment` | False | Disable data augmentation |

### Schedulers

1. **OneCycle** (Recommended): Cycles learning rate from low → high → low
2. **Cosine**: Cosine annealing from initial LR to minimum
3. **Step**: Decrease LR by factor at specific intervals
4. **None**: No learning rate scheduling

## Model Architecture

The ResNet-50 implementation is adapted for CIFAR-100:
- Input: 32×32×3 images
- First conv: 3×3 kernel with stride 1 (no maxpool)
- Bottleneck blocks: [3, 4, 6, 3]
- Output: 100 classes
- Parameters: ~23.7M

## Data Augmentation

Training augmentations:
- Random crop (32×32 with padding 4)
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Cutout (16×16 patches)
- Normalization with CIFAR-100 statistics

## Learning Rate Finder

The LR finder helps identify the optimal learning rate:

```python
from utils.lr_finder import find_optimal_lr
from models.resnet50 import resnet50

model = resnet50(num_classes=100)
suggested_lr, lr_finder = find_optimal_lr(
    model, 
    train_loader,
    num_iter=200
)
```

## Training History

Training metrics are automatically saved:
- Loss curves (train/test)
- Accuracy curves (top-1 and top-5)
- Learning rate schedule
- Best model checkpoint
- Training configuration

## Expected Performance

With default settings (100 epochs, OneCycle):
- Top-1 Test Accuracy: ~75-78%
- Top-5 Test Accuracy: ~93-95%
- Training Time: ~2-3 hours on GPU

## Tips for Better Results

1. **Use OneCycle Scheduler**: Provides best results with faster convergence
2. **Find Optimal LR**: Use `--find-lr` flag for automatic LR detection
3. **Data Augmentation**: Keep augmentation enabled for better generalization
4. **Longer Training**: Train for 200+ epochs for best results
5. **Weight Decay**: Use 5e-4 for good regularization

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 64`
- Use gradient accumulation (modify code)

### Slow Training
- Increase workers: `--num-workers 8`
- Ensure GPU is being used
- Use mixed precision training (requires code modification)

### Poor Accuracy
- Verify data augmentation is enabled
- Use learning rate finder
- Train for more epochs
- Try different schedulers

## Citation

If you use this code, please cite:

```
ResNet: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
OneCycle: Smith, "Super-Convergence: Very Fast Training Using Large Learning Rates", 2018
CIFAR-100: Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009
```

## License

This project is open source and available under the MIT License.

## Training log
============================================================
ResNet-50 CIFAR-100 Training with Binary Data
============================================================
Log file: logs/training_20251008_065424.log
============================================================
Configuration saved to: logs/config_20251008_065424.json
Loaded 100 fine label names

Creating datasets...
Loading data from data/train...
Detected CIFAR-100 fine labels
Loaded 50000 images
Loading data from data/test...
Detected CIFAR-100 fine labels
Loaded 10000 images
Train batches: 390
Test batches: 79

Initializing ResNet-50 model...
Total parameters: 23,705,252

Finding optimal learning rate...
Stopping early, loss diverged at lr=5.05e-07
LR range test complete. Best loss: 0.2517
Model and optimizer reset to initial state
Suggested LR: 3.37e-07
Suggested LR seems unreasonable, using default: 1.00e-01

============================================================
Training from epoch 0 to 130
============================================================
  New best model saved!

Epoch 1/130 Summary:
  Train: Loss=4.4699, Acc=4.02%
  Test:  Loss=3.9429, Acc=8.97%, Top5=27.69%
  LR=1.01e-02, Best=8.97%
------------------------------------------------------------
  New best model saved!

Epoch 2/130 Summary:
  Train: Loss=3.8889, Acc=9.31%
  Test:  Loss=3.6276, Acc=14.22%, Top5=37.46%
  LR=1.06e-02, Best=14.22%
------------------------------------------------------------
  New best model saved!

Epoch 3/130 Summary:
  Train: Loss=3.5911, Acc=14.42%
  Test:  Loss=3.4166, Acc=17.43%, Top5=45.14%
  LR=1.13e-02, Best=17.43%
------------------------------------------------------------
  New best model saved!

Epoch 4/130 Summary:
  Train: Loss=3.2983, Acc=19.66%
  Test:  Loss=2.9971, Acc=24.39%, Top5=55.73%
  LR=1.23e-02, Best=24.39%
------------------------------------------------------------
  New best model saved!

Epoch 5/130 Summary:
  Train: Loss=3.0098, Acc=24.89%
  Test:  Loss=2.9233, Acc=27.27%, Top5=58.31%
  LR=1.36e-02, Best=27.27%
------------------------------------------------------------
  New best model saved!

Epoch 6/130 Summary:
  Train: Loss=2.7274, Acc=30.07%
  Test:  Loss=2.6122, Acc=32.85%, Top5=65.18%
  LR=1.52e-02, Best=32.85%
------------------------------------------------------------
  New best model saved!

Epoch 7/130 Summary:
  Train: Loss=2.4755, Acc=35.21%
  Test:  Loss=2.5179, Acc=36.84%, Top5=68.66%
  LR=1.70e-02, Best=36.84%
------------------------------------------------------------
  New best model saved!

Epoch 8/130 Summary:
  Train: Loss=2.2654, Acc=39.88%
  Test:  Loss=2.0948, Acc=44.34%, Top5=75.96%
  LR=1.90e-02, Best=44.34%
------------------------------------------------------------

Epoch 9/130 Summary:
  Train: Loss=2.0912, Acc=43.57%
  Test:  Loss=2.1239, Acc=43.67%, Top5=75.65%
  LR=2.13e-02, Best=44.34%
------------------------------------------------------------
  New best model saved!
  Checkpoint saved: checkpoint_epoch_10_binary.pth

Epoch 10/130 Summary:
  Train: Loss=1.9785, Acc=46.17%
  Test:  Loss=2.0635, Acc=45.62%, Top5=78.10%
  LR=2.38e-02, Best=45.62%
------------------------------------------------------------
  New best model saved!

Epoch 11/130 Summary:
  Train: Loss=1.8696, Acc=48.48%
  Test:  Loss=2.0077, Acc=47.39%, Top5=77.70%
  LR=2.65e-02, Best=47.39%
------------------------------------------------------------
  New best model saved!

Epoch 12/130 Summary:
  Train: Loss=1.7897, Acc=50.58%
  Test:  Loss=1.7964, Acc=50.48%, Top5=81.47%
  LR=2.94e-02, Best=50.48%
------------------------------------------------------------
  New best model saved!

Epoch 13/130 Summary:
  Train: Loss=1.7134, Acc=52.13%
  Test:  Loss=1.7403, Acc=51.98%, Top5=82.87%
  LR=3.25e-02, Best=51.98%
------------------------------------------------------------

Epoch 14/130 Summary:
  Train: Loss=1.6665, Acc=53.56%
  Test:  Loss=1.9667, Acc=49.27%, Top5=78.85%
  LR=3.57e-02, Best=51.98%
------------------------------------------------------------

Epoch 15/130 Summary:
  Train: Loss=1.6185, Acc=54.51%
  Test:  Loss=1.8570, Acc=50.30%, Top5=81.02%
  LR=3.90e-02, Best=51.98%
------------------------------------------------------------
  New best model saved!

Epoch 16/130 Summary:
  Train: Loss=1.5677, Acc=55.88%
  Test:  Loss=1.6857, Acc=54.05%, Top5=83.45%
  LR=4.25e-02, Best=54.05%
------------------------------------------------------------
  New best model saved!

Epoch 17/130 Summary:
  Train: Loss=1.5387, Acc=56.50%
  Test:  Loss=1.6442, Acc=54.76%, Top5=83.77%
  LR=4.60e-02, Best=54.76%
------------------------------------------------------------

Epoch 18/130 Summary:
  Train: Loss=1.5057, Acc=57.52%
  Test:  Loss=1.8089, Acc=51.73%, Top5=81.98%
  LR=4.96e-02, Best=54.76%
------------------------------------------------------------
  New best model saved!

Epoch 19/130 Summary:
  Train: Loss=1.4824, Acc=58.18%
  Test:  Loss=1.5747, Acc=56.40%, Top5=85.63%
  LR=5.32e-02, Best=56.40%
------------------------------------------------------------
  Checkpoint saved: checkpoint_epoch_20_binary.pth

Epoch 20/130 Summary:
  Train: Loss=1.4644, Acc=58.52%
  Test:  Loss=1.6355, Acc=55.51%, Top5=84.07%
  LR=5.68e-02, Best=56.40%
------------------------------------------------------------

Epoch 21/130 Summary:
  Train: Loss=1.4413, Acc=59.24%
  Test:  Loss=1.7540, Acc=52.12%, Top5=82.48%
  LR=6.04e-02, Best=56.40%
------------------------------------------------------------

Epoch 22/130 Summary:
  Train: Loss=1.4287, Acc=59.34%
  Test:  Loss=1.6782, Acc=54.20%, Top5=84.07%
  LR=6.40e-02, Best=56.40%
------------------------------------------------------------

Epoch 23/130 Summary:
  Train: Loss=1.4141, Acc=59.85%
  Test:  Loss=1.7019, Acc=53.58%, Top5=83.54%
  LR=6.75e-02, Best=56.40%
------------------------------------------------------------

Epoch 24/130 Summary:
  Train: Loss=1.3973, Acc=60.31%
  Test:  Loss=1.7488, Acc=53.07%, Top5=82.22%
  LR=7.10e-02, Best=56.40%
------------------------------------------------------------

Epoch 25/130 Summary:
  Train: Loss=1.3864, Acc=60.68%
  Test:  Loss=1.6236, Acc=56.17%, Top5=84.33%
  LR=7.43e-02, Best=56.40%
------------------------------------------------------------

Epoch 26/130 Summary:
  Train: Loss=1.3767, Acc=61.02%
  Test:  Loss=1.9069, Acc=50.82%, Top5=79.89%
  LR=7.75e-02, Best=56.40%
------------------------------------------------------------

Epoch 27/130 Summary:
  Train: Loss=1.3654, Acc=61.20%
  Test:  Loss=1.7565, Acc=52.90%, Top5=82.33%
  LR=8.06e-02, Best=56.40%
------------------------------------------------------------

Epoch 28/130 Summary:
  Train: Loss=1.3580, Acc=61.23%
  Test:  Loss=1.7527, Acc=52.92%, Top5=82.76%
  LR=8.35e-02, Best=56.40%
------------------------------------------------------------

Epoch 29/130 Summary:
  Train: Loss=1.3368, Acc=61.80%
  Test:  Loss=1.6765, Acc=55.11%, Top5=83.28%
  LR=8.62e-02, Best=56.40%
------------------------------------------------------------
  Checkpoint saved: checkpoint_epoch_30_binary.pth

Epoch 30/130 Summary:
  Train: Loss=1.3417, Acc=61.71%
  Test:  Loss=1.7207, Acc=55.00%, Top5=83.85%
  LR=8.87e-02, Best=56.40%
------------------------------------------------------------
  New best model saved!

Epoch 31/130 Summary:
  Train: Loss=1.3293, Acc=62.07%
  Test:  Loss=1.5275, Acc=58.74%, Top5=85.96%
  LR=9.10e-02, Best=58.74%
------------------------------------------------------------

Epoch 32/130 Summary:
  Train: Loss=1.3251, Acc=62.02%
  Test:  Loss=1.6455, Acc=55.61%, Top5=85.02%
  LR=9.30e-02, Best=58.74%
------------------------------------------------------------

Epoch 33/130 Summary:
  Train: Loss=1.3157, Acc=62.29%
  Test:  Loss=1.6411, Acc=55.38%, Top5=84.29%
  LR=9.48e-02, Best=58.74%
------------------------------------------------------------

Epoch 34/130 Summary:
  Train: Loss=1.3073, Acc=62.59%
  Test:  Loss=1.8717, Acc=52.27%, Top5=82.02%
  LR=9.64e-02, Best=58.74%
------------------------------------------------------------

Epoch 35/130 Summary:
  Train: Loss=1.2900, Acc=63.11%
  Test:  Loss=1.6651, Acc=55.39%, Top5=84.12%
  LR=9.77e-02, Best=58.74%
------------------------------------------------------------

Epoch 36/130 Summary:
  Train: Loss=1.2842, Acc=63.21%
  Test:  Loss=1.6108, Acc=58.03%, Top5=84.86%
  LR=9.87e-02, Best=58.74%
------------------------------------------------------------

Epoch 37/130 Summary:
  Train: Loss=1.2724, Acc=63.64%
  Test:  Loss=1.6007, Acc=55.96%, Top5=84.86%
  LR=9.94e-02, Best=58.74%
------------------------------------------------------------

Epoch 38/130 Summary:
  Train: Loss=1.2728, Acc=63.74%
  Test:  Loss=1.6498, Acc=55.93%, Top5=84.25%
  LR=9.99e-02, Best=58.74%
------------------------------------------------------------

Epoch 39/130 Summary:
  Train: Loss=1.2605, Acc=63.75%
  Test:  Loss=1.6113, Acc=56.64%, Top5=85.05%
  LR=1.00e-01, Best=58.74%
------------------------------------------------------------
  Checkpoint saved: checkpoint_epoch_40_binary.pth

Epoch 40/130 Summary:
  Train: Loss=1.2524, Acc=63.98%
  Test:  Loss=1.7349, Acc=54.72%, Top5=83.96%
  LR=1.00e-01, Best=58.74%
------------------------------------------------------------

Epoch 41/130 Summary:
  Train: Loss=1.2539, Acc=63.93%
  Test:  Loss=1.5629, Acc=58.49%, Top5=85.69%
  LR=9.99e-02, Best=58.74%
------------------------------------------------------------

Epoch 42/130 Summary:
  Train: Loss=1.2277, Acc=64.69%
  Test:  Loss=1.6405, Acc=56.63%, Top5=84.83%
  LR=9.97e-02, Best=58.74%
------------------------------------------------------------

Epoch 43/130 Summary:
  Train: Loss=1.2196, Acc=65.20%
  Test:  Loss=1.6606, Acc=56.38%, Top5=84.68%
  LR=9.95e-02, Best=58.74%
------------------------------------------------------------
  New best model saved!

Epoch 44/130 Summary:
  Train: Loss=1.2308, Acc=64.95%
  Test:  Loss=1.4347, Acc=60.91%, Top5=87.47%
  LR=9.93e-02, Best=60.91%
------------------------------------------------------------

Epoch 45/130 Summary:
  Train: Loss=1.1987, Acc=65.29%
  Test:  Loss=1.5720, Acc=58.30%, Top5=85.57%
  LR=9.89e-02, Best=60.91%
------------------------------------------------------------

Epoch 46/130 Summary:
  Train: Loss=1.2078, Acc=65.41%
  Test:  Loss=1.5755, Acc=58.45%, Top5=86.13%
  LR=9.85e-02, Best=60.91%
------------------------------------------------------------

Epoch 47/130 Summary:
  Train: Loss=1.1999, Acc=65.53%
  Test:  Loss=1.7263, Acc=55.38%, Top5=83.21%
  LR=9.81e-02, Best=60.91%
------------------------------------------------------------

Epoch 48/130 Summary:
  Train: Loss=1.1938, Acc=65.73%
  Test:  Loss=1.6052, Acc=56.97%, Top5=85.40%
  LR=9.76e-02, Best=60.91%
------------------------------------------------------------

Epoch 49/130 Summary:
  Train: Loss=1.1952, Acc=65.59%
  Test:  Loss=1.5847, Acc=58.19%, Top5=84.99%
  LR=9.70e-02, Best=60.91%
------------------------------------------------------------
  Checkpoint saved: checkpoint_epoch_50_binary.pth

Epoch 50/130 Summary:
  Train: Loss=1.1766, Acc=66.10%
  Test:  Loss=1.4689, Acc=60.33%, Top5=86.95%
  LR=9.64e-02, Best=60.91%
------------------------------------------------------------

Epoch 51/130 Summary:
  Train: Loss=1.1706, Acc=66.32%
  Test:  Loss=1.5162, Acc=58.40%, Top5=86.22%
  LR=9.58e-02, Best=60.91%
------------------------------------------------------------

Epoch 52/130 Summary:
  Train: Loss=1.1631, Acc=66.40%
  Test:  Loss=1.5881, Acc=57.74%, Top5=85.51%
  LR=9.50e-02, Best=60.91%
------------------------------------------------------------

Epoch 53/130 Summary:
  Train: Loss=1.1562, Acc=66.39%
  Test:  Loss=1.5882, Acc=58.30%, Top5=85.62%
  LR=9.43e-02, Best=60.91%
------------------------------------------------------------

Epoch 54/130 Summary:
  Train: Loss=1.1568, Acc=66.40%
  Test:  Loss=1.6168, Acc=57.36%, Top5=84.45%
  LR=9.34e-02, Best=60.91%
------------------------------------------------------------

Epoch 55/130 Summary:
  Train: Loss=1.1536, Acc=66.70%
  Test:  Loss=1.6939, Acc=56.94%, Top5=84.36%
  LR=9.26e-02, Best=60.91%
------------------------------------------------------------

Epoch 56/130 Summary:
  Train: Loss=1.1355, Acc=67.18%
  Test:  Loss=1.7019, Acc=57.25%, Top5=84.68%
  LR=9.16e-02, Best=60.91%
------------------------------------------------------------

Epoch 57/130 Summary:
  Train: Loss=1.1304, Acc=67.31%
  Test:  Loss=1.4831, Acc=60.26%, Top5=87.25%
  LR=9.07e-02, Best=60.91%
------------------------------------------------------------

Epoch 58/130 Summary:
  Train: Loss=1.1255, Acc=67.33%
  Test:  Loss=1.5011, Acc=59.32%, Top5=87.00%
  LR=8.96e-02, Best=60.91%
------------------------------------------------------------

Epoch 59/130 Summary:
  Train: Loss=1.1258, Acc=67.39%
  Test:  Loss=1.5710, Acc=58.85%, Top5=85.59%
  LR=8.85e-02, Best=60.91%
------------------------------------------------------------
  Checkpoint saved: checkpoint_epoch_60_binary.pth

Epoch 60/130 Summary:
  Train: Loss=1.1199, Acc=67.62%
  Test:  Loss=1.4782, Acc=60.47%, Top5=86.67%
  LR=8.74e-02, Best=60.91%
------------------------------------------------------------

Epoch 61/130 Summary:
  Train: Loss=1.1014, Acc=68.04%
  Test:  Loss=1.4614, Acc=60.59%, Top5=88.09%
  LR=8.63e-02, Best=60.91%
------------------------------------------------------------

Epoch 62/130 Summary:
  Train: Loss=1.1035, Acc=67.93%
  Test:  Loss=1.7142, Acc=55.66%, Top5=83.58%
  LR=8.50e-02, Best=60.91%
------------------------------------------------------------

Epoch 63/130 Summary:
  Train: Loss=1.0947, Acc=68.28%
  Test:  Loss=1.5426, Acc=59.63%, Top5=86.71%
  LR=8.38e-02, Best=60.91%
------------------------------------------------------------

Epoch 64/130 Summary:
  Train: Loss=1.0884, Acc=68.44%
  Test:  Loss=1.5319, Acc=59.19%, Top5=86.46%
  LR=8.25e-02, Best=60.91%
------------------------------------------------------------

Epoch 65/130 Summary:
  Train: Loss=1.0932, Acc=68.10%
  Test:  Loss=1.6237, Acc=57.90%, Top5=85.19%
  LR=8.12e-02, Best=60.91%
------------------------------------------------------------
  New best model saved!

Epoch 66/130 Summary:
  Train: Loss=1.0722, Acc=68.78%
  Test:  Loss=1.4115, Acc=61.44%, Top5=87.32%
  LR=7.98e-02, Best=61.44%
------------------------------------------------------------
  New best model saved!

Epoch 67/130 Summary:
  Train: Loss=1.0670, Acc=68.95%
  Test:  Loss=1.3758, Acc=62.94%, Top5=88.29%
  LR=7.84e-02, Best=62.94%
------------------------------------------------------------

Epoch 68/130 Summary:
  Train: Loss=1.0607, Acc=68.92%
  Test:  Loss=1.6036, Acc=58.30%, Top5=85.71%
  LR=7.70e-02, Best=62.94%
------------------------------------------------------------

Epoch 69/130 Summary:
  Train: Loss=1.0615, Acc=69.18%
  Test:  Loss=1.4809, Acc=60.89%, Top5=87.09%
  LR=7.55e-02, Best=62.94%
------------------------------------------------------------
  Checkpoint saved: checkpoint_epoch_70_binary.pth

Epoch 70/130 Summary:
  Train: Loss=1.0528, Acc=69.17%
  Test:  Loss=1.4156, Acc=62.61%, Top5=87.75%
  LR=7.40e-02, Best=62.94%
------------------------------------------------------------

Epoch 71/130 Summary:
  Train: Loss=1.0424, Acc=69.36%
  Test:  Loss=1.4485, Acc=60.75%, Top5=87.48%
  LR=7.25e-02, Best=62.94%
------------------------------------------------------------

Epoch 72/130 Summary:
  Train: Loss=1.0327, Acc=69.92%
  Test:  Loss=1.5861, Acc=58.55%, Top5=86.30%
  LR=7.09e-02, Best=62.94%
------------------------------------------------------------

Epoch 73/130 Summary:
  Train: Loss=1.0262, Acc=70.07%
  Test:  Loss=1.4227, Acc=61.58%, Top5=87.88%
  LR=6.93e-02, Best=62.94%
------------------------------------------------------------

Epoch 74/130 Summary:
  Train: Loss=1.0158, Acc=70.40%
  Test:  Loss=1.3906, Acc=62.75%, Top5=87.88%
  LR=6.77e-02, Best=62.94%
------------------------------------------------------------

Epoch 75/130 Summary:
  Train: Loss=1.0123, Acc=70.20%
  Test:  Loss=1.4221, Acc=61.84%, Top5=87.40%
  LR=6.61e-02, Best=62.94%
------------------------------------------------------------

Epoch 76/130 Summary:
  Train: Loss=0.9905, Acc=70.97%
  Test:  Loss=1.4201, Acc=62.29%, Top5=87.91%
  LR=6.45e-02, Best=62.94%
------------------------------------------------------------

Epoch 77/130 Summary:
  Train: Loss=1.0016, Acc=70.34%
  Test:  Loss=1.4470, Acc=61.69%, Top5=87.76%
  LR=6.28e-02, Best=62.94%
------------------------------------------------------------

Epoch 78/130 Summary:
  Train: Loss=0.9883, Acc=70.98%
  Test:  Loss=1.3691, Acc=62.92%, Top5=87.92%
  LR=6.11e-02, Best=62.94%
------------------------------------------------------------

Epoch 79/130 Summary:
  Train: Loss=0.9721, Acc=71.67%
  Test:  Loss=1.4873, Acc=61.24%, Top5=87.52%
  LR=5.94e-02, Best=62.94%
------------------------------------------------------------
  New best model saved!
  Checkpoint saved: checkpoint_epoch_80_binary.pth

Epoch 80/130 Summary:
  Train: Loss=0.9649, Acc=71.61%
  Test:  Loss=1.3117, Acc=64.44%, Top5=89.26%
  LR=5.77e-02, Best=64.44%
------------------------------------------------------------

Epoch 81/130 Summary:
  Train: Loss=0.9490, Acc=72.14%
  Test:  Loss=1.4679, Acc=60.82%, Top5=87.30%
  LR=5.60e-02, Best=64.44%
------------------------------------------------------------

Epoch 82/130 Summary:
  Train: Loss=0.9421, Acc=72.19%
  Test:  Loss=1.7811, Acc=55.45%, Top5=83.39%
  LR=5.43e-02, Best=64.44%
------------------------------------------------------------

Epoch 83/130 Summary:
  Train: Loss=0.9362, Acc=72.39%
  Test:  Loss=1.3404, Acc=63.82%, Top5=88.85%
  LR=5.26e-02, Best=64.44%
------------------------------------------------------------
  New best model saved!

Epoch 84/130 Summary:
  Train: Loss=0.9191, Acc=72.69%
  Test:  Loss=1.2876, Acc=64.75%, Top5=89.43%
  LR=5.09e-02, Best=64.75%
------------------------------------------------------------
  New best model saved!

Epoch 85/130 Summary:
  Train: Loss=0.9005, Acc=73.52%
  Test:  Loss=1.2856, Acc=65.49%, Top5=89.34%
  LR=4.91e-02, Best=65.49%
------------------------------------------------------------

Epoch 86/130 Summary:
  Train: Loss=0.9123, Acc=73.03%
  Test:  Loss=1.2841, Acc=64.85%, Top5=89.73%
  LR=4.74e-02, Best=65.49%
------------------------------------------------------------

Epoch 87/130 Summary:
  Train: Loss=0.8780, Acc=73.89%
  Test:  Loss=1.3039, Acc=64.84%, Top5=89.49%
  LR=4.57e-02, Best=65.49%
------------------------------------------------------------

Epoch 88/130 Summary:
  Train: Loss=0.8779, Acc=73.85%
  Test:  Loss=1.3369, Acc=64.20%, Top5=89.32%
  LR=4.40e-02, Best=65.49%
------------------------------------------------------------

Epoch 89/130 Summary:
  Train: Loss=0.8491, Acc=74.91%
  Test:  Loss=1.3674, Acc=63.79%, Top5=88.52%
  LR=4.23e-02, Best=65.49%
------------------------------------------------------------
  Checkpoint saved: checkpoint_epoch_90_binary.pth

Epoch 90/130 Summary:
  Train: Loss=0.8510, Acc=74.66%
  Test:  Loss=1.2803, Acc=65.30%, Top5=89.50%
  LR=4.06e-02, Best=65.49%
------------------------------------------------------------
  New best model saved!

Epoch 91/130 Summary:
  Train: Loss=0.8447, Acc=74.90%
  Test:  Loss=1.2846, Acc=65.93%, Top5=90.26%
  LR=3.89e-02, Best=65.93%
------------------------------------------------------------

Epoch 92/130 Summary:
  Train: Loss=0.8038, Acc=76.05%
  Test:  Loss=1.2823, Acc=65.44%, Top5=89.63%
  LR=3.72e-02, Best=65.93%
------------------------------------------------------------

Epoch 93/130 Summary:
  Train: Loss=0.7846, Acc=76.57%
  Test:  Loss=1.3256, Acc=65.13%, Top5=89.62%
  LR=3.55e-02, Best=65.93%
------------------------------------------------------------

Epoch 94/130 Summary:
  Train: Loss=0.7753, Acc=76.61%
  Test:  Loss=1.3698, Acc=64.28%, Top5=88.56%
  LR=3.39e-02, Best=65.93%
------------------------------------------------------------
  New best model saved!

Epoch 95/130 Summary:
  Train: Loss=0.7552, Acc=77.30%
  Test:  Loss=1.2221, Acc=66.43%, Top5=90.21%
  LR=3.23e-02, Best=66.43%
------------------------------------------------------------

Epoch 96/130 Summary:
  Train: Loss=0.7429, Acc=77.72%
  Test:  Loss=1.2725, Acc=65.83%, Top5=90.33%
  LR=3.07e-02, Best=66.43%
------------------------------------------------------------
  New best model saved!

Epoch 97/130 Summary:
  Train: Loss=0.7184, Acc=78.55%
  Test:  Loss=1.1862, Acc=68.57%, Top5=90.86%
  LR=2.91e-02, Best=68.57%
------------------------------------------------------------

Epoch 98/130 Summary:
  Train: Loss=0.7006, Acc=78.83%
  Test:  Loss=1.2460, Acc=67.12%, Top5=90.27%
  LR=2.75e-02, Best=68.57%
------------------------------------------------------------

Epoch 99/130 Summary:
  Train: Loss=0.6762, Acc=79.61%
  Test:  Loss=1.1741, Acc=68.41%, Top5=91.06%
  LR=2.60e-02, Best=68.57%
------------------------------------------------------------
  Checkpoint saved: checkpoint_epoch_100_binary.pth

Epoch 100/130 Summary:
  Train: Loss=0.6481, Acc=80.50%
  Test:  Loss=1.1927, Acc=67.94%, Top5=90.88%
  LR=2.45e-02, Best=68.57%
------------------------------------------------------------
  New best model saved!

Epoch 101/130 Summary:
  Train: Loss=0.6295, Acc=80.96%
  Test:  Loss=1.1888, Acc=68.79%, Top5=91.27%
  LR=2.30e-02, Best=68.79%
------------------------------------------------------------
  New best model saved!

Epoch 102/130 Summary:
  Train: Loss=0.5968, Acc=81.72%
  Test:  Loss=1.1589, Acc=69.53%, Top5=91.37%
  LR=2.16e-02, Best=69.53%
------------------------------------------------------------

Epoch 103/130 Summary:
  Train: Loss=0.5755, Acc=82.28%
  Test:  Loss=1.2704, Acc=67.31%, Top5=90.21%
  LR=2.02e-02, Best=69.53%
------------------------------------------------------------

Epoch 104/130 Summary:
  Train: Loss=0.5397, Acc=83.33%
  Test:  Loss=1.1664, Acc=69.41%, Top5=91.35%
  LR=1.88e-02, Best=69.53%
------------------------------------------------------------

Epoch 105/130 Summary:
  Train: Loss=0.5252, Acc=83.89%
  Test:  Loss=1.1910, Acc=69.15%, Top5=90.93%
  LR=1.75e-02, Best=69.53%
------------------------------------------------------------
  New best model saved!

Epoch 106/130 Summary:
  Train: Loss=0.4847, Acc=85.13%
  Test:  Loss=1.1039, Acc=71.12%, Top5=92.44%
  LR=1.62e-02, Best=71.12%
------------------------------------------------------------
  New best model saved!

Epoch 107/130 Summary:
  Train: Loss=0.4502, Acc=86.20%
  Test:  Loss=1.1056, Acc=71.31%, Top5=91.87%
  LR=1.50e-02, Best=71.31%
------------------------------------------------------------
  New best model saved!

Epoch 108/130 Summary:
  Train: Loss=0.4296, Acc=86.76%
  Test:  Loss=1.0989, Acc=71.82%, Top5=92.21%
  LR=1.37e-02, Best=71.82%
------------------------------------------------------------
  New best model saved!

Epoch 109/130 Summary:
  Train: Loss=0.3891, Acc=88.05%
  Test:  Loss=1.0498, Acc=72.76%, Top5=92.66%
  LR=1.26e-02, Best=72.76%
------------------------------------------------------------
  Checkpoint saved: checkpoint_epoch_110_binary.pth

Epoch 110/130 Summary:
  Train: Loss=0.3499, Acc=89.38%
  Test:  Loss=1.0780, Acc=72.66%, Top5=92.59%
  LR=1.15e-02, Best=72.76%
------------------------------------------------------------
  New best model saved!

Epoch 111/130 Summary:
  Train: Loss=0.3032, Acc=90.73%
  Test:  Loss=1.0165, Acc=73.24%, Top5=93.29%
  LR=1.04e-02, Best=73.24%
------------------------------------------------------------
  New best model saved!

Epoch 112/130 Summary:
  Train: Loss=0.2680, Acc=92.00%
  Test:  Loss=1.0167, Acc=74.53%, Top5=93.37%
  LR=9.35e-03, Best=74.53%
------------------------------------------------------------

Epoch 113/130 Summary:
  Train: Loss=0.2379, Acc=92.87%
  Test:  Loss=0.9928, Acc=74.50%, Top5=93.36%
  LR=8.37e-03, Best=74.53%
------------------------------------------------------------
  New best model saved!

Epoch 114/130 Summary:
  Train: Loss=0.2043, Acc=93.96%
  Test:  Loss=0.9955, Acc=75.19%, Top5=93.34%
  LR=7.44e-03, Best=75.19%
------------------------------------------------------------

Epoch 115/130 Summary:
  Train: Loss=0.1749, Acc=94.97%
  Test:  Loss=0.9798, Acc=75.16%, Top5=93.69%
  LR=6.56e-03, Best=75.19%
------------------------------------------------------------
  New best model saved!

Epoch 116/130 Summary:
  Train: Loss=0.1459, Acc=95.95%
  Test:  Loss=0.9378, Acc=76.53%, Top5=94.00%
  LR=5.73e-03, Best=76.53%
------------------------------------------------------------
  New best model saved!

Epoch 117/130 Summary:
  Train: Loss=0.1199, Acc=96.77%
  Test:  Loss=0.9161, Acc=77.01%, Top5=94.14%
  LR=4.96e-03, Best=77.01%
------------------------------------------------------------

Epoch 118/130 Summary:
  Train: Loss=0.0992, Acc=97.53%
  Test:  Loss=0.9187, Acc=77.01%, Top5=94.20%
  LR=4.24e-03, Best=77.01%
------------------------------------------------------------
  New best model saved!

Epoch 119/130 Summary:
  Train: Loss=0.0812, Acc=98.10%
  Test:  Loss=0.8871, Acc=77.72%, Top5=94.50%
  LR=3.57e-03, Best=77.72%
------------------------------------------------------------
  New best model saved!
  Checkpoint saved: checkpoint_epoch_120_binary.pth

Epoch 120/130 Summary:
  Train: Loss=0.0642, Acc=98.62%
  Test:  Loss=0.8879, Acc=77.85%, Top5=94.46%
  LR=2.96e-03, Best=77.85%
------------------------------------------------------------
  New best model saved!

Epoch 121/130 Summary:
  Train: Loss=0.0587, Acc=98.81%
  Test:  Loss=0.8899, Acc=78.17%, Top5=94.48%
  LR=2.40e-03, Best=78.17%
------------------------------------------------------------
  New best model saved!

Epoch 122/130 Summary:
  Train: Loss=0.0493, Acc=99.07%
  Test:  Loss=0.8780, Acc=78.39%, Top5=94.46%
  LR=1.90e-03, Best=78.39%
------------------------------------------------------------
  New best model saved!

Epoch 123/130 Summary:
  Train: Loss=0.0436, Acc=99.22%
  Test:  Loss=0.8667, Acc=78.42%, Top5=94.48%
  LR=1.46e-03, Best=78.42%
------------------------------------------------------------

Epoch 124/130 Summary:
  Train: Loss=0.0391, Acc=99.31%
  Test:  Loss=0.8670, Acc=78.41%, Top5=94.54%
  LR=1.08e-03, Best=78.42%
------------------------------------------------------------
  New best model saved!

Epoch 125/130 Summary:
  Train: Loss=0.0368, Acc=99.40%
  Test:  Loss=0.8647, Acc=78.57%, Top5=94.51%
  LR=7.52e-04, Best=78.57%
------------------------------------------------------------

Epoch 126/130 Summary:
  Train: Loss=0.0334, Acc=99.51%
  Test:  Loss=0.8599, Acc=78.52%, Top5=94.64%
  LR=4.85e-04, Best=78.57%
------------------------------------------------------------
  New best model saved!

Epoch 127/130 Summary:
  Train: Loss=0.0329, Acc=99.50%
  Test:  Loss=0.8580, Acc=78.72%, Top5=94.48%
  LR=2.77e-04, Best=78.72%
------------------------------------------------------------

Epoch 128/130 Summary:
  Train: Loss=0.0312, Acc=99.56%
  Test:  Loss=0.8623, Acc=78.62%, Top5=94.56%
  LR=1.29e-04, Best=78.72%
------------------------------------------------------------

Epoch 129/130 Summary:
  Train: Loss=0.0312, Acc=99.54%
  Test:  Loss=0.8632, Acc=78.46%, Top5=94.47%
  LR=3.96e-05, Best=78.72%
------------------------------------------------------------
  Checkpoint saved: checkpoint_epoch_130_binary.pth

Epoch 130/130 Summary:
  Train: Loss=0.0305, Acc=99.58%
  Test:  Loss=0.8567, Acc=78.67%, Top5=94.49%
  LR=1.00e-05, Best=78.72%
------------------------------------------------------------

============================================================
Training Completed!
Best Test Accuracy: 78.72%
============================================================
Training history saved to: logs/history_20251008_091322.json
Training curves saved to: logs/training_curves_20251008_091322.png
Summary report saved to: logs/summary_20251008_184053.txt
