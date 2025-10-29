#!/bin/bash

echo "PyTorch Installation Script for ResNet50 Training"
echo "================================================"
echo ""

# Function to detect CUDA version
detect_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1-2)
        echo "Detected CUDA version: $cuda_version"
        
        # Map CUDA version to PyTorch index
        case $cuda_version in
            "11.7") cuda_index="cu117" ;;
            "11.8") cuda_index="cu118" ;;
            "12.1") cuda_index="cu121" ;;
            "12.2") cuda_index="cu121" ;;  # Use 12.1 for 12.2
            *) cuda_index="cu118" ;;  # Default to 11.8
        esac
        return 0
    else
        echo "No CUDA detected. Will install CPU version."
        return 1
    fi
}

# Detect system and CUDA
if detect_cuda; then
    echo "Installing PyTorch with CUDA support ($cuda_index)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/$cuda_index
else
    echo "Installing PyTorch CPU version..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo ""
echo "Installing other requirements..."
pip install tensorboard>=2.13.0
pip install torch-lr-finder>=0.2.1
pip install numpy>=1.24.0
pip install scipy>=1.10.0
pip install pandas>=2.0.0
pip install matplotlib>=3.7.0

echo ""
echo "Installation complete!"
echo ""

# Verify installation
python -c "
import torch
import torchvision
print('PyTorch version:', torch.__version__)
print('Torchvision version:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
"

echo ""
echo "To test the training script, run:"
echo "  python train_resnet50.py --help"