# Fixing Overfitting in ImageNet Training

## Current Issues
- Training accuracy: 80%
- Validation accuracy: 50% (stuck)
- Training loss: 2 (flat)
- Using 110 epochs with OneCycleLR (way too many!)

## Immediate Fixes

### 1. Reduce Epochs
```python
config['epochs'] = 30  # OneCycleLR works best with 20-50 epochs
```

### 2. Use Stronger Augmentations
Update your notebook to use the improved loader:
```python
from dataset.imagenet_loader_improved import create_imagenet_loaders

train_loader, val_loader, dataset_stats = create_imagenet_loaders(
    data_dir=config['data_dir'],
    batch_size=config['batch_size'],
    num_workers=config['num_workers'],
    subset_percent=config.get('subset_percent', None),
    tiny_imagenet=config.get('tiny_imagenet', False),
    augment_train=True,
    augmentation_strength='strong'  # Use strong augmentations
)
```

### 3. Add Dropout to Model
If your ResNet50 doesn't have dropout, add it:
```python
# In your model, before the final FC layer
self.dropout = nn.Dropout(0.2)  # 20% dropout
```

### 4. Use Label Smoothing
```python
# Replace CrossEntropyLoss with label smoothing
from torch.nn import CrossEntropyLoss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        loss = -log_preds.sum(dim=-1)
        nll = F.nll_loss(log_preds, target, reduction='none')
        smooth_loss = loss / n_classes
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth_loss
        return loss.mean()

# Use it in your training
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
```

### 5. Reduce Learning Rate
```python
# After LR finder, reduce the suggested LR
config['learning_rate'] = suggested_lr * 0.5  # Use 50% of suggested LR
config['max_lr'] = suggested_lr * 0.5
```

### 6. Add Weight Decay
```python
config['weight_decay'] = 5e-4  # Increase from 1e-4
```

### 7. Use Mixup or CutMix
```python
# Add to your training loop
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# In training loop:
if use_mixup:
    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
    outputs = model(inputs)
    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
```

### 8. Monitor and Early Stop
```python
# Add early stopping
patience = 10
no_improve_epochs = 0

if val_acc > best_val_acc:
    best_val_acc = val_acc
    no_improve_epochs = 0
    # Save model
else:
    no_improve_epochs += 1
    if no_improve_epochs >= patience:
        print("Early stopping triggered")
        break
```

## Diagnostic Steps

1. **Check if model is actually learning features**:
   - Visualize first layer filters
   - Check if validation loss is decreasing at all

2. **Verify data loading**:
   - Ensure train/val splits are correct
   - Check that augmentations are being applied

3. **Learning rate schedule**:
   - Plot the learning rate over epochs
   - Ensure it's not too high or too low

## Expected Results
With these changes, you should see:
- Validation accuracy improving to 60-70% within 20-30 epochs
- Training and validation accuracies closer together
- Training loss decreasing properly (not stuck at 2)

## If Still Not Working
1. Start with pretrained weights: `config['pretrained'] = True`
2. Use a smaller learning rate: `config['learning_rate'] = 0.01`
3. Check for data loading issues or corrupted images
4. Try gradient clipping: `clip_grad_norm = 1.0`