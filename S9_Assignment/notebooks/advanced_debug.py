# Advanced debugging for high loss and low accuracy issues

import torch
import torch.nn as nn
import numpy as np

print("="*60)
print("ADVANCED DEBUGGING FOR IMAGENET TRAINING")
print("="*60)

# 1. VERIFY INITIAL LOSS IS CORRECT
print("\n1. INITIAL LOSS CHECK:")
num_classes = dataset_stats['num_classes']
expected_initial_loss = -np.log(1.0 / num_classes)
print(f"   Expected initial loss for {num_classes} classes: {expected_initial_loss:.3f}")
print(f"   Your reported loss: ~7.0")
print(f"   ✓ This is NORMAL for random initialization!")

# 2. CHECK FIRST BATCH IN DETAIL
print("\n2. DETAILED FIRST BATCH CHECK:")
# Add this at the beginning of your training loop
first_batch = next(iter(train_loader))
images, labels = first_batch[0].to(device), first_batch[1].to(device)

# Check batch statistics
print(f"   Batch shape: {images.shape}")
print(f"   Labels shape: {labels.shape}")
print(f"   Unique labels in batch: {len(torch.unique(labels))}")
print(f"   Label distribution: min={labels.min().item()}, max={labels.max().item()}")

# 3. TEST MODEL OUTPUT AND LOSS
print("\n3. MODEL OUTPUT TEST:")
model.eval()
with torch.no_grad():
    outputs = model(images)
    print(f"   Output shape: {outputs.shape}")
    print(f"   Output stats: mean={outputs.mean().item():.3f}, std={outputs.std().item():.3f}")
    
    # Check if outputs are reasonable
    probs = torch.softmax(outputs, dim=1)
    max_probs, predictions = probs.max(dim=1)
    print(f"   Max probability stats: mean={max_probs.mean().item():.3f}, std={max_probs.std().item():.3f}")
    
    # Calculate loss
    loss = criterion(outputs, labels)
    print(f"   Loss on first batch: {loss.item():.3f}")
    
    # Check accuracy
    correct = (predictions == labels).sum().item()
    accuracy = 100.0 * correct / labels.size(0)
    print(f"   Accuracy on first batch: {accuracy:.2f}% ({correct}/{labels.size(0)})")

# 4. GRADIENT CHECK
print("\n4. GRADIENT FLOW CHECK:")
model.train()
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()

# Check if gradients are flowing
grad_norms = []
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.data.norm(2).item()
        grad_norms.append(grad_norm)
        if 'fc' in name or 'classifier' in name:  # Check final layer
            print(f"   {name}: grad_norm={grad_norm:.6f}")

print(f"   Total parameters with gradients: {len(grad_norms)}")
print(f"   Average gradient norm: {np.mean(grad_norms):.6f}")
print(f"   Max gradient norm: {np.max(grad_norms):.6f}")

if np.mean(grad_norms) < 1e-6:
    print("   ⚠️  WARNING: Gradients are too small!")
elif np.max(grad_norms) > 100:
    print("   ⚠️  WARNING: Gradients might be exploding!")
else:
    print("   ✓ Gradient magnitudes look reasonable")

# 5. LEARNING RATE DIAGNOSTIC
print("\n5. LEARNING RATE CHECK:")
print(f"   Current LR: {optimizer.param_groups[0]['lr']:.6f}")
print(f"   Suggested LR from finder: {config.get('max_lr', 'Not set')}")

# Simulate one optimization step
optimizer.step()
optimizer.zero_grad()

# Check if parameters changed
outputs_after = model(images)
loss_after = criterion(outputs_after, labels)
print(f"   Loss after one step: {loss_after.item():.3f}")
print(f"   Loss change: {loss.item() - loss_after.item():.6f}")

if abs(loss.item() - loss_after.item()) < 1e-6:
    print("   ⚠️  WARNING: Loss didn't change - LR might be too small!")
elif loss_after.item() > loss.item() + 0.5:
    print("   ⚠️  WARNING: Loss increased significantly - LR might be too high!")
else:
    print("   ✓ Loss decreased after optimization step")

# 6. DATA AUGMENTATION CHECK
print("\n6. AUGMENTATION IMPACT TEST:")
# Compare augmented vs non-augmented batch
train_dataset = train_loader.dataset
if hasattr(train_dataset, 'transform'):
    print("   ✓ Transforms are applied to training data")
    # You could temporarily disable transforms to compare
else:
    print("   ⚠️  WARNING: No transforms detected!")

# 7. DEBUGGING RECOMMENDATIONS
print("\n7. DEBUGGING RECOMMENDATIONS:")
print("   Based on the analysis above:")

if loss.item() > 6.5 and loss.item() < 7.5:
    print("   • Initial loss is normal for 1000 classes")
    print("   • Check if loss decreases over first 10-20 batches")
    print("   • If not decreasing, try:")
    print("     - Reduce learning rate by 10x")
    print("     - Check data loading (are images/labels matched?)")
    print("     - Verify model architecture (especially final layer)")

# 8. QUICK FIXES TO TRY
print("\n8. QUICK FIXES TO TRY:")
print("   a) Reduce learning rate:")
print("      config['learning_rate'] = suggested_lr * 0.1  # Use 10% instead of 50%")
print("   b) Start with pretrained weights:")
print("      config['pretrained'] = True")
print("   c) Reduce batch size:")
print("      config['batch_size'] = 64  # If GPU memory allows")
print("   d) Disable OneCycleLR initially:")
print("      config['scheduler'] = None  # Use constant LR first")

# 9. MONITOR FIRST EPOCH
print("\n9. FIRST EPOCH MONITORING:")
print("   Add this to your training loop:")
print("""
   if epoch == 0:
       # Log every 10 batches
       if batch_idx % 10 == 0:
           print(f"Batch {batch_idx}: Loss={loss.item():.3f}, "
                 f"Acc={100.*correct/total:.2f}%")
""")

# 10. SANITY CHECK WITH SMALLER DATASET
print("\n10. SANITY CHECK:")
print("   Try training on a tiny subset first:")
print("   config['dataset_type'] = 'small'  # Use 1% of data")
print("   This should overfit quickly if everything is working")