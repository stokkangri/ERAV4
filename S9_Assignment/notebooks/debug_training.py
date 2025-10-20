# Debug code to add to your notebook to diagnose the issue

# 1. After creating data loaders (line 307), add:
print("\n=== DATA LOADER DIAGNOSTICS ===")
print(f"Dataset stats: {dataset_stats}")
print(f"Augmentation strength: {dataset_stats.get('augmentation_strength', 'NOT SET')}")

# Load a sample batch
try:
    sample_batch = next(iter(train_loader))
    images, labels = sample_batch
    print(f"\nSample batch loaded successfully!")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label range: {labels.min().item()} to {labels.max().item()}")
    print(f"Expected label range: 0 to {dataset_stats['num_classes']-1}")
    print(f"Image stats - Min: {images.min().item():.3f}, Max: {images.max().item():.3f}")
    
    # Check if images are normalized
    if images.min().item() < -1 or images.max().item() > 2:
        print("WARNING: Images might not be properly normalized!")
except Exception as e:
    print(f"ERROR loading batch: {e}")

# 2. After model creation (line 347), add:
print("\n=== MODEL DIAGNOSTICS ===")
# Test forward pass with dummy data
try:
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    dummy_output = model(dummy_input)
    print(f"Model forward pass successful!")
    print(f"Output shape: {dummy_output.shape}")
    print(f"Output range: {dummy_output.min().item():.3f} to {dummy_output.max().item():.3f}")
except Exception as e:
    print(f"ERROR in model forward pass: {e}")

# 3. Calculate expected initial loss
num_classes = dataset_stats['num_classes']
expected_loss = -np.log(1.0 / num_classes)
print(f"\nExpected initial loss (random guessing): {expected_loss:.3f}")

# 4. After first epoch training, check if loss is reasonable
# Add this in your training loop after first batch:
if epoch == 0 and batch_idx == 0:
    print(f"\nFirst batch loss: {loss.item():.3f}")
    if loss.item() > expected_loss * 2:
        print("WARNING: Loss is much higher than expected!")
    elif loss.item() < expected_loss * 0.5:
        print("WARNING: Loss is suspiciously low!")

# 5. Fix for data loader (line 285-299):
train_loader, val_loader, dataset_stats = create_imagenet_loaders(
    data_dir=config['data_dir'],
    batch_size=config['batch_size'],
    num_workers=config['num_workers'],
    subset_percent=config.get('subset_percent', None),
    tiny_imagenet=config.get('tiny_imagenet', False),  # Fix the parameter
    augment_train=True,
    augmentation_strength='strong'  # ADD THIS!
)

# 6. More conservative LR adjustment (after line 401):
print(f"\nLR Finder suggested: {suggested_lr:.2e}")
# Use 50% instead of 100% for safety with OneCycleLR
config['learning_rate'] = suggested_lr * 0.5
config['max_lr'] = suggested_lr * 0.5
print(f"Using LR: {config['learning_rate']:.2e}")