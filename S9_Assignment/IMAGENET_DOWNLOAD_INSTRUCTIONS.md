# ImageNet-1k Download Instructions

This guide provides detailed instructions for downloading and preparing the ImageNet-1k dataset for training.

## Dataset Overview

ImageNet-1k (ILSVRC2012) contains:
- 1,000 object classes
- ~1.28 million training images
- 50,000 validation images
- 100,000 test images (labels not publicly available)

## Download Options

### Option 1: Official ImageNet Website (Recommended)

1. **Register for an account** at https://image-net.org/signup
2. **Navigate to** https://image-net.org/download-images
3. **Download the following files**:
   - `ILSVRC2012_img_train.tar` (~138 GB)
   - `ILSVRC2012_img_val.tar` (~6.3 GB)
   - `ILSVRC2012_devkit_t12.tar.gz` (contains metadata and tools)

### Option 2: Academic Torrents

If you have trouble accessing the official website:
- Training data: https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2
- Validation data: https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5

### Option 3: Kaggle (Requires Kaggle Account)

1. Install Kaggle API: `pip install kaggle`
2. Setup API credentials: https://github.com/Kaggle/kaggle-api#api-credentials
3. Download using:
```bash
kaggle competitions download -c imagenet-object-localization-challenge
```

## Data Preparation

### Step 1: Create Directory Structure

```bash
mkdir -p data/imagenet/{train,val}
```

### Step 2: Extract Training Data

```bash
# Extract main training archive
tar -xf ILSVRC2012_img_train.tar -C data/imagenet/train/

# Extract individual class archives
cd data/imagenet/train
for f in *.tar; do
    class_name="${f%.tar}"
    mkdir -p "$class_name"
    tar -xf "$f" -C "$class_name"
    rm "$f"  # Optional: remove tar file to save space
done
cd -
```

### Step 3: Extract and Organize Validation Data

```bash
# Extract validation archive
tar -xf ILSVRC2012_img_val.tar -C data/imagenet/val/

# Download validation labels
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
chmod +x valprep.sh

# Organize validation images into class folders
cd data/imagenet/val
bash ../../../valprep.sh
cd -
```

### Alternative: Python Script for Validation Organization

```python
import os
import shutil
from pathlib import Path

# Download this file first
# wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt

val_dir = Path('data/imagenet/val')
with open('imagenet_2012_validation_synset_labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Create class directories and move images
for i, label in enumerate(labels):
    class_dir = val_dir / label
    class_dir.mkdir(exist_ok=True)
    
    # Validation images are named ILSVRC2012_val_00000001.JPEG to ILSVRC2012_val_00050000.JPEG
    img_name = f'ILSVRC2012_val_{i+1:08d}.JPEG'
    src = val_dir / img_name
    dst = class_dir / img_name
    
    if src.exists():
        shutil.move(str(src), str(dst))
```

## Disk Space Requirements

- Training data (extracted): ~140 GB
- Validation data (extracted): ~6.5 GB
- **Total**: ~150 GB

## Smaller Alternatives for Testing

### 1. Tiny ImageNet

A smaller subset with 200 classes and 64x64 images:

```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

- 200 classes (subset of ImageNet)
- 500 training images per class
- 50 validation images per class
- 50 test images per class
- Total size: ~240 MB (compressed)

### 2. ImageNet Subset

Use the provided data loader with `subset_percent` parameter:

```python
from dataset.imagenet_loader import create_imagenet_loaders

# Use only 1% of ImageNet data
train_loader, val_loader, stats = create_imagenet_loaders(
    data_dir='data/imagenet',
    subset_percent=0.01  # 1% of data
)
```

### 3. ImageNette/ImageWoof

Smaller curated subsets by fast.ai:

```bash
# ImageNette (10 easy classes)
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar -xf imagenette2.tgz

# ImageWoof (10 difficult dog breeds)
wget https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz
tar -xf imagewoof2.tgz
```

## Verification

After extraction, verify your directory structure:

```bash
# Check number of training classes
ls data/imagenet/train | wc -l  # Should be 1000

# Check number of validation classes
ls data/imagenet/val | wc -l  # Should be 1000

# Check total number of training images
find data/imagenet/train -name "*.JPEG" | wc -l  # Should be 1,281,167

# Check total number of validation images
find data/imagenet/val -name "*.JPEG" | wc -l  # Should be 50,000
```

## Class Mapping

Download the class index mapping:

```bash
wget https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/imagenet_class_index.json -O data/imagenet/imagenet_class_index.json
```

## Usage with Our Code

Once the data is prepared, you can use it with our training script:

```bash
# Full ImageNet training
python train_imagenet.py --data-dir data/imagenet --epochs 90

# Small subset for testing
python train_imagenet.py --data-dir data/imagenet --subset-percent 0.01 --epochs 10

# Tiny ImageNet
python train_imagenet.py --data-dir data/tiny-imagenet-200 --tiny-imagenet --epochs 50
```

## Google Colab Setup

For Google Colab users:

1. **Upload to Google Drive**: Due to size constraints, upload the dataset to Google Drive
2. **Mount Drive in Colab**:
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. **Create symbolic link**:
```python
!ln -s /content/drive/MyDrive/imagenet /content/imagenet
```

## Tips for Faster Downloads

1. **Use a fast internet connection**: University networks often have better bandwidth
2. **Use wget with resume capability**:
```bash
wget -c <url>  # -c flag allows resuming partial downloads
```
3. **Use aria2 for parallel downloads**:
```bash
aria2c -x 16 -s 16 <url>  # 16 parallel connections
```
4. **Consider using a cloud instance**: Download directly to cloud storage

## Common Issues

### Issue 1: Access Denied
- Make sure you're logged into your ImageNet account
- Some institutions have site licenses - check with your IT department

### Issue 2: Slow Download Speeds
- Try downloading during off-peak hours
- Use academic torrents as an alternative
- Consider using a VPN if your ISP throttles large downloads

### Issue 3: Corrupted Archives
- Verify checksums if provided
- Use `tar -tf archive.tar | head` to test archive integrity
- Re-download if necessary

### Issue 4: Out of Disk Space
- ImageNet requires ~150GB after extraction
- Consider using external storage or cloud storage
- Delete tar files after extraction to save space

## License and Citation

ImageNet is available for non-commercial research purposes only. If you use ImageNet in your research, please cite:

```bibtex
@article{imagenet_cvpr09,
    Author = {Deng, J. and Dong, W. and Socher, R. and Li, L.-J. and Li, K. and Fei-Fei, L.},
    Title = {{ImageNet: A Large-Scale Hierarchical Image Database}},
    Journal = {CVPR09},
    Year = {2009}
}