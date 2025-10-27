# Pretrained Model Test - Add this to your notebook

## Add this as a new cell after creating your data loaders:

```python
# Test with pretrained model to verify data loader
import torchvision.models as models

print("Testing with pretrained ResNet50 to verify data loader...")

# Load pretrained model
pretrained_model = models.resnet50(pretrained=True)
pretrained_model = pretrained_model.to(device)
pretrained_model.eval()

# Test on validation set
num_batches = 5
total_correct = 0
total_correct_top5 = 0
total_samples = 0

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(val_loader):
        if batch_idx >= num_batches:
            break
            
        images, labels = images.to(device), labels.to(device)
        outputs = pretrained_model(images)
        
        # Top-1 accuracy
        _, predictions = outputs.max(1)
        correct = predictions.eq(labels).sum().item()
        total_correct += correct
        
        # Top-5 accuracy
        _, pred_top5 = outputs.topk(5, 1, largest=True, sorted=True)
        correct_top5 = pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).sum().item()
        total_correct_top5 += correct_top5
        
        total_samples += labels.size(0)
        
        print(f"Batch {batch_idx+1}: Top-1: {100.*correct/labels.size(0):.1f}%, "
              f"Top-5: {100.*correct_top5/labels.size(0):.1f}%")

print(f"\nOverall on {total_samples} samples:")
print(f"Top-1 Accuracy: {100.*total_correct/total_samples:.2f}%")
print(f"Top-5 Accuracy: {100.*total_correct_top5/total_samples:.2f}%")

print("\nExpected: ~76% top-1, ~93% top-5 for ImageNet")
print("If accuracy is very low, check data loading!")
```

## Alternative: Quick single batch test

```python
# Quick test on single batch
import torchvision.models as models

# Load pretrained model
model_pretrained = models.resnet50(pretrained=True).to(device).eval()

# Get one batch
images, labels = next(iter(val_loader))
images, labels = images.to(device), labels.to(device)

# Inference
with torch.no_grad():
    outputs = model_pretrained(images)
    _, preds = outputs.max(1)
    acc = 100. * preds.eq(labels).sum().item() / labels.size(0)

print(f"Pretrained ResNet50 accuracy on batch: {acc:.2f}%")
print(f"Label range: {labels.min().item()}-{labels.max().item()} (should be 0-999)")
print(f"If accuracy < 50%, there's likely a data loading issue!")

# Show first 5 predictions
for i in range(min(5, len(labels))):
    print(f"Image {i}: True={labels[i].item()}, Pred={preds[i].item()}, "
          f"Correct={'✓' if labels[i]==preds[i] else '✗'}")
```

## To use YOUR model with pretrained weights:

```python
# Test your model with pretrained=True
model_test = resnet50(
    num_classes=1000,
    pretrained=True,  # This loads pretrained weights
    replace_maxpool_with_conv=False  # Use False for standard pretrained weights
).to(device).eval()

# Test on one batch
with torch.no_grad():
    outputs = model_test(images)
    _, preds = outputs.max(1)
    acc = 100. * preds.eq(labels).sum().item() / labels.size(0)

print(f"Your model with pretrained weights: {acc:.2f}%")
```

## What to look for:
1. **Good signs:**
   - Pretrained model gets 60-80% accuracy on validation batches
   - Labels are in range 0-999
   - Your model with pretrained=True performs similarly

2. **Problems to check:**
   - If accuracy is 0% or near 0%: Labels might be shifted or wrong
   - If accuracy is exactly 0.1% (1/1000): Model might be outputting same class
   - If labels are outside 0-999: Data loading issue
   - If images look wrong when visualized: Preprocessing issue