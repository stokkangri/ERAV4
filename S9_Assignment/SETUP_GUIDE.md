# Environment Setup Guide for ResNet50 Training

## Prerequisites
- Python 3.8 or higher
- CUDA 11.7+ (for GPU training, optional)
- 16GB+ RAM recommended
- 100GB+ free disk space for ImageNet dataset

## Quick Setup

### 1. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv resnet50_env
source resnet50_env/bin/activate  # On Windows: resnet50_env\Scripts\activate

# Or using conda
conda create -n resnet50 python=3.9
conda activate resnet50
```

### 2. Install Requirements

#### Option A: Automatic Installation (Recommended)
```bash
# This script will detect your CUDA version and install appropriate PyTorch
./install_requirements.sh
```

#### Option B: Manual Installation
```bash
# For GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

#### Option C: Using Conda
```bash
# For GPU
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU
conda install pytorch torchvision cpuonly -c pytorch

# Install remaining packages
pip install tensorboard torch-lr-finder scipy pandas matplotlib
```

## Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

## Common Issues and Solutions

### 1. CUDA Version Mismatch
If you get CUDA errors, ensure your PyTorch CUDA version matches your system CUDA:
```bash
# Check system CUDA
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu[YOUR_CUDA_VERSION]
```

### 2. Out of Memory Errors
- Reduce batch size: `--batch-size 64` or even `--batch-size 32`
- Reduce number of workers: `--workers 2`
- Enable gradient checkpointing (requires code modification)

### 3. ImportError: No module named 'torch_lr_finder'
```bash
pip install torch-lr-finder
```

### 4. matplotlib Backend Issues on Server
The script automatically uses 'Agg' backend for server environments. If you still have issues:
```bash
export MPLBACKEND=Agg
```

## AWS EC2 Setup

### 1. Choose Instance Type
- **Development/Testing**: g4dn.xlarge (1x T4 GPU, cheaper)
- **Training**: p3.2xlarge (1x V100 GPU)
- **Fast Training**: p3.8xlarge (4x V100 GPUs)

### 2. Deep Learning AMI
Use AWS Deep Learning AMI which comes with CUDA and PyTorch pre-installed:
```bash
# Activate PyTorch environment
source activate pytorch
```

### 3. Install Additional Requirements
```bash
pip install tensorboard torch-lr-finder
```

## Docker Setup (Optional)

Create a `Dockerfile`:
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "train_resnet50.py"]
```

Build and run:
```bash
docker build -t resnet50-training .
docker run --gpus all -v /path/to/imagenet:/data resnet50-training \
    --data-path /data/train --val-path /data/val
```

## Package Versions

Tested with:
- Python: 3.8, 3.9, 3.10
- PyTorch: 2.0.0+
- CUDA: 11.7, 11.8, 12.1
- Ubuntu: 20.04, 22.04
- macOS: 12.0+ (CPU only)

## Next Steps

1. Download ImageNet dataset (see IMAGENET_DOWNLOAD_INSTRUCTIONS.md)
2. Test the installation:
   ```bash
   python train_resnet50.py --help
   ```
3. Start training:
   ```bash
   ./run_training.sh