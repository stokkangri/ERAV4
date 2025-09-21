# Session 5 - MNIST CNN Architecture and Training

## Model Architecture

### Network Structure
The model is a Convolutional Neural Network designed for MNIST digit classification with the following architecture:

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Initial feature extraction
        self.conv0 = nn.Conv2d(1, 4, kernel_size=3, padding=1)      # 28x28x1 -> 28x28x4
        self.bn0 = nn.BatchNorm2d(4)
        self.dropout0 = nn.Dropout(0.15)
        
        # First block
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, padding=1)      # 28x28x4 -> 28x28x8
        self.bn1 = nn.BatchNorm2d(8)
        self.dropout1 = nn.Dropout(0.125)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)       # 28x28x8 -> 14x14x8
        
        # Second block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)     # 14x14x8 -> 14x14x16
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(0.1)
        
        # Third block
        self.conv3 = nn.Conv2d(16, 22, kernel_size=3, padding=1)    # 14x14x16 -> 14x14x22
        self.bn3 = nn.BatchNorm2d(22)
        self.dropout3 = nn.Dropout(0.1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)       # 14x14x22 -> 7x7x22
        
        # Fourth block
        self.conv4 = nn.Conv2d(22, 30, kernel_size=3)               # 7x7x22 -> 5x5x30
        self.bn_conv4 = nn.BatchNorm2d(30)
        self.dropout_conv4 = nn.Dropout(0.1)
        
        # Fifth block
        self.conv5 = nn.Conv2d(30, 30, kernel_size=3)               # 5x5x30 -> 3x3x30
        self.bn5 = nn.BatchNorm2d(30)
        self.dropout5 = nn.Dropout(0.1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 3x3x30 -> 2x2x30
        
        # Dimension reduction
        self.conv1x1_reduce_1 = nn.Conv2d(30, 10, kernel_size=1)    # 2x2x30 -> 2x2x10
        self.bn_conv1x1_reduce_1 = nn.BatchNorm2d(10)
        self.dropout_conv1x1_reduce_1 = nn.Dropout(0.1)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1,1))                      # 2x2x10 -> 1x1x10
        self.dropout = nn.Dropout(0.1)
```

### Key Architecture Features

1. **Progressive Channel Expansion**: 1 → 4 → 8 → 16 → 22 → 30 channels
2. **Strategic Pooling**: Three MaxPool layers for spatial dimension reduction
3. **Batch Normalization**: Applied after every convolutional layer for stable training
4. **Dropout Regularization**: Variable dropout rates (0.15 to 0.1) for preventing overfitting
5. **1x1 Convolution**: Used for channel reduction before final classification
6. **Global Average Pooling**: Reduces spatial dimensions to 1x1 before final output

### Receptive Field Analysis

The model achieves a **GLOBAL receptive field** covering the entire 28x28 input image through:
- Multiple 3x3 convolutions building local receptive fields
- Three MaxPool2d operations doubling the receptive field jump
- Final Global Average Pooling ensuring each output sees the entire feature map

Key RF progression:
- Initial layers: RF grows from 1 → 3 → 5 → 7
- After first MaxPool: Jump increases to 2
- Progressive expansion through deeper layers
- Final GAP: Achieves global receptive field (28x28)

## Training Configuration

### Data Augmentation
Enhanced augmentation pipeline for training:
```python
train_transform = transforms.Compose([
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=5, translate=(0.1,0.1), scale=(0.9, 1.1), shear=5),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### Training Parameters
- **Optimizer**: SGD with momentum=0.9
- **Learning Rate**: OneCycleLR scheduler with max_lr=0.1
- **Batch Size**: 128
- **Epochs**: 20
- **Loss Function**: Negative Log Likelihood (NLL)
- **Dataset Split**: 50,000 training / 10,000 validation samples

### Model Statistics
- **Total Parameters**: ~18,746 parameters
- **Model Size**: ~0.07 MB
- **Input Size**: 28x28x1 (MNIST grayscale images)
- **Output**: 10 classes (digits 0-9)

## Advanced Features

### 1. Receptive Field Calculator
Custom implementation that extends torchsummary to display receptive field progression through the network:
- Tracks RF, Jump, and Output Size for each layer
- Handles various layer types (Conv2d, MaxPool2d, BatchNorm, etc.)
- Supports Global Average Pooling RF calculation

### 2. Model Shape Debugger
Debugging tool for tracing tensor shapes through forward pass:
- Identifies size mismatches
- Layer-by-layer shape analysis
- Automatic error detection and diagnosis

### 3. Failure Analysis System
Comprehensive failure case analysis during testing:
- Captures misclassified samples
- Visualizes failure cases with predictions
- Tracks confusion matrix and most confused pairs
- Provides failure rate statistics

## Performance Metrics Tracking

The training loop tracks:
- **Training Loss & Accuracy**: Per-epoch metrics
- **Validation Loss & Accuracy**: Evaluated after each epoch
- **Learning Rate**: Monitored through OneCycleLR scheduler
- **Failure Analysis**: Detailed misclassification patterns

## Visualization

The notebook includes:
1. **Training Progress Plots**: 2x2 grid showing train/test loss and accuracy curves
2. **Sample Visualizations**: Display of augmented training samples
3. **Failure Case Analysis**: Visual representation of misclassified digits
4. **Confusion Analysis**: Most commonly confused digit pairs

## Key Innovations

1. **Adaptive Dropout**: Variable dropout rates decreasing with depth
2. **Global Context**: GAP ensures global receptive field for better feature aggregation
3. **Efficient Architecture**: Achieves good performance with <20K parameters
4. **Comprehensive Analysis Tools**: Built-in debugging and visualization capabilities

## Usage

```python
# Initialize model
model = Net().to(device)

# View architecture with RF analysis
summary_with_rf(model, input_size=(1, 28, 28), device=device, debug=True)

# Train model
for epoch in range(1, epochs+1):
    train(model, device, train_loader, optimizer, scheduler, epoch)
    test_loss, accuracy, analyzer = test(model, device, val_loader, epoch)
```

## Results
The model achieves competitive accuracy on MNIST with:
- Efficient parameter usage
- Robust to variations through augmentation
- Comprehensive failure analysis for model improvement
- Global receptive field ensuring complete feature coverage