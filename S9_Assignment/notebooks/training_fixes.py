# Critical fixes for your training notebook

# 1. CORRECT IMPORTS (replace line 97)
from dataset.imagenet_loader_improved import create_imagenet_loaders

# 2. CORRECT DATA LOADER CREATION (replace lines 285-299)
train_loader, val_loader, dataset_stats = create_imagenet_loaders(
    data_dir=config['data_dir'],
    batch_size=config['batch_size'],
    num_workers=config['num_workers'],
    subset_percent=config.get('subset_percent', None),
    tiny_imagenet=config.get('tiny_imagenet', False),  # Fixed parameter
    augment_train=True,
    augmentation_strength='strong'  # Add strong augmentations
)

# 3. LEARNING RATE ADJUSTMENT (after line 401)
# After LR finder suggests a rate:
config['learning_rate'] = suggested_lr * 0.3  # Reduce to 30%
config['max_lr'] = suggested_lr * 0.3

# 4. ADD GRADIENT CLIPPING (modify line 533)
train_loss, train_acc = train_epoch(
    model, device, train_loader, optimizer, criterion,
    scheduler if config['scheduler'] == 'onecycle' else None,
    epoch, accumulation_steps=1, clip_grad_norm=1.0, verbose=True  # Add gradient clipping
)

# 5. OPTIONAL: Add Label Smoothing
import torch.nn.functional as F

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

# Replace line 413:
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# 6. REDUCE WEIGHT DECAY (line 150)
config['weight_decay'] = 5e-4  # Increase from 1e-4

# 7. CHECK YOUR DATA PATH
# Make sure this path is correct (line 126):
# 'data_dir': '/home/xpz1/Downloads/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/'
# The directory should contain 'train' and 'val' subdirectories

# 8. DEBUGGING: Add this after creating data loaders to verify
print(f"\nVerifying data loader:")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")

# Test loading one batch
try:
    sample_batch = next(iter(train_loader))
    print(f"Sample batch shape: {sample_batch[0].shape}")
    print(f"Labels shape: {sample_batch[1].shape}")
    print(f"Label values range: {sample_batch[1].min()} to {sample_batch[1].max()}")
except Exception as e:
    print(f"ERROR loading batch: {e}")