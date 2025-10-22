# Comprehensive debugging script for ImageNet training issues

import torch
import numpy as np

# 1. FIX THE TYPO (line 295)
# Change: tiny_imagenet=config.get('tiny_imagement', False),
# To:     tiny_imagenet=config.get('tiny_imagenet', False),

# 2. After creating data loaders (add after line 311), run this comprehensive check:
print("\n" + "="*50)
print("COMPREHENSIVE DATA LOADER DIAGNOSTICS")
print("="*50)

# Check augmentation strength
aug_strength = dataset_stats.get('augmentation_strength', 'NOT SET')
print(f"\n1. Augmentation strength: {aug_strength}")
if aug_strength != 'strong':
    print("   ⚠️  WARNING: Strong augmentations NOT applied!")
else:
    print("   ✓ Strong augmentations correctly applied")

# Check dataset size
print(f"\n2. Dataset size:")
print(f"   - Num classes: {dataset_stats['num_classes']}")
print(f"   - Train samples: {dataset_stats['num_train_samples']}")
print(f"   - Val samples: {dataset_stats['num_val_samples']}")
print(f"   - Train batches: {len(train_loader)}")
print(f"   - Val batches: {len(val_loader)}")

# Expected initial loss
expected_loss = -np.log(1.0 / dataset_stats['num_classes'])
print(f"\n3. Expected initial loss (random guessing): {expected_loss:.3f}")

# Load and check a batch
try:
    sample_batch = next(iter(train_loader))
    images, labels = sample_batch
    print(f"\n4. Sample batch check:")
    print(f"   - Images shape: {images.shape}")
    print(f"   - Labels shape: {labels.shape}")
    print(f"   - Label range: {labels.min().item()} to {labels.max().item()}")
    print(f"   - Expected range: 0 to {dataset_stats['num_classes']-1}")
    
    # Check if labels are within valid range
    if labels.max().item() >= dataset_stats['num_classes']:
        print(f"   ⚠️  ERROR: Labels exceed num_classes!")
    else:
        print(f"   ✓ Labels are within valid range")
    
    # Check image statistics
    print(f"\n5. Image statistics:")
    print(f"   - Min: {images.min().item():.3f}")
    print(f"   - Max: {images.max().item():.3f}")
    print(f"   - Mean: {images.mean().item():.3f}")
    print(f"   - Std: {images.std().item():.3f}")
    
    # Check if properly normalized
    if images.min().item() < -3 or images.max().item() > 3:
        print("   ⚠️  WARNING: Images might not be properly normalized!")
    else:
        print("   ✓ Images appear to be properly normalized")
        
except Exception as e:
    print(f"\n⚠️  ERROR loading batch: {e}")

# 3. After model creation (add after line 363), test the model:
print("\n" + "="*50)
print("MODEL DIAGNOSTICS")
print("="*50)

# Test forward pass
try:
    with torch.no_grad():
        # Use actual batch if available
        if 'images' in locals():
            test_input = images[:2].to(device)
        else:
            test_input = torch.randn(2, 3, 224, 224).to(device)
        
        test_output = model(test_input)
        print(f"\n1. Model forward pass:")
        print(f"   - Input shape: {test_input.shape}")
        print(f"   - Output shape: {test_output.shape}")
        print(f"   - Output range: [{test_output.min().item():.3f}, {test_output.max().item():.3f}]")
        
        # Check if output dimension matches num_classes
        if test_output.shape[1] != dataset_stats['num_classes']:
            print(f"   ⚠️  ERROR: Output dimension {test_output.shape[1]} doesn't match num_classes {dataset_stats['num_classes']}!")
        else:
            print(f"   ✓ Output dimension matches num_classes")
            
        # Test loss calculation
        if 'labels' in locals():
            test_labels = labels[:2].to(device)
            test_loss = criterion(test_output, test_labels)
            print(f"\n2. Loss calculation test:")
            print(f"   - Test loss: {test_loss.item():.3f}")
            print(f"   - Expected initial loss: ~{expected_loss:.3f}")
            
            if test_loss.item() > expected_loss * 2:
                print("   ⚠️  WARNING: Loss is much higher than expected!")
            elif test_loss.item() < expected_loss * 0.5:
                print("   ⚠️  WARNING: Loss is suspiciously low!")
            else:
                print("   ✓ Loss is in reasonable range")
                
except Exception as e:
    print(f"\n⚠️  ERROR in model forward pass: {e}")

# 4. Check data path
print("\n" + "="*50)
print("DATA PATH CHECK")
print("="*50)
print(f"\nData directory: {config['data_dir']}")
print("Checking if directories exist...")

import os
train_path = os.path.join(config['data_dir'], 'train')
val_path = os.path.join(config['data_dir'], 'val')

if os.path.exists(train_path):
    num_train_classes = len([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    print(f"✓ Train directory exists with {num_train_classes} classes")
else:
    print(f"⚠️  ERROR: Train directory not found at {train_path}")

if os.path.exists(val_path):
    num_val_classes = len([d for d in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, d))])
    print(f"✓ Val directory exists with {num_val_classes} classes")
else:
    print(f"⚠️  ERROR: Val directory not found at {val_path}")

# 5. Learning rate check (after LR finder, around line 420)
print("\n" + "="*50)
print("LEARNING RATE CHECK")
print("="*50)
print(f"\nSuggested LR: {suggested_lr:.2e}")
print(f"Using LR: {config['learning_rate']:.2e} (50% of suggested)")
print(f"Max LR: {config['max_lr']:.2e}")

# 6. Add this in the training loop after first batch:
# if epoch == 0 and batch_idx == 0:
#     print(f"\nFirst batch diagnostics:")
#     print(f"  - Batch loss: {loss.item():.3f}")
#     print(f"  - Expected initial loss: ~{expected_loss:.3f}")
#     if loss.item() > expected_loss * 2:
#         print("  ⚠️  WARNING: Loss is much higher than expected!")
#         print("  Possible issues:")
#         print("  - Model output dimension doesn't match num_classes")
#         print("  - Labels are out of range")
#         print("  - Data loading issue")