# ImageNet Training with ResNet-50

This repository contains a complete pipeline for training ResNet-50 on ImageNet-1k dataset with various options for experimentation and production training.

## Features

- ✅ **Flexible Dataset Support**:
  - Full ImageNet-1k (1000 classes)
  - Tiny ImageNet (200 classes, 64x64 images)
  - Configurable subset sampling for quick experiments
  
- ✅ **Model Options**:
  - Standard ResNet-50 architecture
  - Strided convolution instead of MaxPool (default)
  - Pretrained weights support
  - Multi-GPU training support

- ✅ **Training Features**:
  - Learning Rate Finder
  - Multiple schedulers (OneCycle, Cosine, Step)
  - Gradient accumulation
  - Mixed precision training ready
  - Comprehensive logging and visualization

- ✅ **Google Colab Support**:
  - Ready-to-use Jupyter notebook
  - Automatic setup and configuration
  - Options for small/medium/full dataset training

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd S9_Assignment
```

### 2. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib tqdm scipy
```

### 3. Download Dataset

See [IMAGENET_DOWNLOAD_INSTRUCTIONS.md](IMAGENET_DOWNLOAD_INSTRUCTIONS.md) for detailed instructions.

For quick testing, use Tiny ImageNet:
```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d data/
```

### 4. Train the Model

#### Option A: Using Command Line

```bash
# Small subset for testing (1% of ImageNet)
python train_imagenet.py --data-dir data/imagenet --subset-percent 0.01 --epochs 10 --find-lr

# Tiny ImageNet
python train_imagenet.py --data-dir data/tiny-imagenet-200 --tiny-imagenet --epochs 50

# Full ImageNet with pretrained weights
python train_imagenet.py --data-dir data/imagenet --pretrained --epochs 90

# Use standard MaxPool instead of Conv (Conv is now default)
python train_imagenet.py --data-dir data/imagenet --no-replace-maxpool-with-conv --epochs 90
```

#### Option B: Using Configuration Files

```bash
# Small subset training
python train_with_config.py --config configs/config_small_subset.yaml

# Full ImageNet training
python train_with_config.py --config configs/config_full_imagenet.yaml

# Tiny ImageNet
python train_with_config.py --config configs/config_tiny_imagenet.yaml

# Fine-tuning pretrained model
python train_with_config.py --config configs/config_pretrained_finetune.yaml
```

#### Option C: Using Google Colab

1. Upload the notebook `notebooks/train_imagenet_colab.ipynb` to Google Colab
2. Follow the instructions in the notebook
3. Choose your dataset size and training configuration
4. Run all cells

## Project Structure

```
S9_Assignment/
├── dataset/
│   └── imagenet_loader.py      # ImageNet data loader with subset support
├── models/
│   └── resnet50_imagenet.py    # ResNet-50 model (with MaxPool/Conv option)
├── utils/
│   ├── lr_finder.py            # Learning rate finder utility
│   └── train_test.py           # Training and testing functions
├── configs/
│   ├── config_small_subset.yaml     # 1% ImageNet config
│   ├── config_full_imagenet.yaml    # Full ImageNet config
│   ├── config_tiny_imagenet.yaml    # Tiny ImageNet config
│   └── config_pretrained_finetune.yaml  # Fine-tuning config
├── notebooks/
│   └── train_imagenet_colab.ipynb   # Google Colab notebook
├── train_imagenet.py           # Main training script
├── IMAGENET_DOWNLOAD_INSTRUCTIONS.md  # Dataset download guide
└── README.md                   # This file
```

## Training Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 90 | Number of training epochs |
| `--batch-size` | 256 | Batch size for training |
| `--lr` | 0.1 | Initial learning rate |
| `--momentum` | 0.9 | SGD momentum |
| `--weight-decay` | 1e-4 | Weight decay (L2 penalty) |
| `--data-dir` | ./data/imagenet | Path to dataset |
| `--subset-percent` | None | Use only this percentage of data (0-1) |
| `--tiny-imagenet` | False | Use Tiny ImageNet dataset |
| `--pretrained` | False | Use pretrained weights |
| `--no-replace-maxpool-with-conv` | False | Use standard MaxPool instead of strided Conv |
| `--scheduler` | cosine | LR scheduler (onecycle/cosine/step/none) |
| `--find-lr` | False | Run LR finder before training |
| `--resume` | False | Resume from checkpoint |
| `--no-multi-gpu` | False | Disable multi-GPU training |

### Learning Rate Finder

The LR finder helps identify the optimal learning rate:

```python
python train_imagenet.py --find-lr --lr-finder-iterations 200
```

This will:
1. Run a range test from 1e-7 to 10
2. Plot loss vs learning rate
3. Suggest an optimal learning rate
4. Use the suggested LR for training

### Multi-GPU Training

Multi-GPU training is enabled by default if multiple GPUs are available:

```python
# Use all available GPUs
python train_imagenet.py

# Disable multi-GPU
python train_imagenet.py --no-multi-gpu
```

## Results and Checkpoints

Training outputs are organized as follows:

```
checkpoints/
├── imagenet_full/
│   ├── best_model.pth          # Best model based on validation accuracy
│   ├── last_checkpoint.pth     # Latest checkpoint for resuming
│   └── checkpoint_epoch_N.pth  # Periodic checkpoints
logs/
├── imagenet_full/
│   ├── training_history.json   # Training metrics history
│   └── config.json            # Training configuration
plots/
├── imagenet_full/
│   ├── lr_finder.png          # LR finder plot
│   └── training_curves.png    # Training progress plots
```

## Performance Tips

1. **Data Loading**:
   - Use SSD for dataset storage
   - Increase `num_workers` based on CPU cores
   - Enable `pin_memory` for GPU training

2. **Memory Optimization**:
   - Reduce batch size if running out of GPU memory
   - Use gradient accumulation for effective larger batch sizes
   - Enable mixed precision training (if supported)

3. **Training Speed**:
   - Use OneCycle scheduler for faster convergence
   - Start with pretrained weights when possible
   - Use subset training for hyperparameter tuning

## Expected Results

| Dataset | Model | Epochs | Top-1 Acc | Top-5 Acc |
|---------|-------|--------|-----------|-----------|
| ImageNet-1k | ResNet-50 | 90 | ~76% | ~93% |
| ImageNet-1k | ResNet-50 (pretrained) | 20 | ~76.5% | ~93.2% |
| Tiny ImageNet | ResNet-50 | 50 | ~65% | ~85% |
| ImageNet 1% | ResNet-50 | 10 | ~25% | ~50% |

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Enable gradient accumulation
- Use a smaller model or subset of data

### Slow Data Loading
- Increase number of workers
- Use SSD instead of HDD
- Reduce image augmentations

### Poor Convergence
- Run LR finder to find optimal learning rate
- Try different schedulers
- Check data augmentation settings

## Citation

If you use this code in your research, please cite:

```bibtex
@article{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  journal={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ResNet architecture from the original paper by He et al.
- ImageNet dataset from Stanford Vision Lab
- PyTorch framework and torchvision library