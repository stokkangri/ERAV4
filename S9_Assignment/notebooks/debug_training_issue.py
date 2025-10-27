# Debug Training Issues - Why isn't the model learning?

import torch
import torch.nn as nn
import numpy as np

print("="*60)
print("DEBUGGING TRAINING ISSUES")
print("="*60)

# 1. Test if model can overfit on a tiny batch
print("\n1. OVERFITTING TEST ON SINGLE BATCH:")
print("   Training on just 1 batch repeatedly - should overfit to 100%")

# Get one batch
single_batch = next(iter(train_loader))
single_images, single_labels = single_batch[0].to(device), single_batch[1].to(device)
print(f"   Batch size: {single_images.shape[0]}")

# Create fresh model (no pretrained weights)
test_model = resnet50(
    num_classes=1000,
    pretrained=False,
    replace_maxpool_with_conv=config['replace_maxpool_with_conv']
).to(device)

# Simple SGD optimizer
test_optimizer = torch.optim.SGD(test_model.parameters(), lr=0.01, momentum=0.9)
test_criterion = nn.CrossEntropyLoss()

# Train on single batch
print("\n   Training on single batch for 20 iterations:")
test_model.train()
for i in range(20):
    test_optimizer.zero_grad()
    outputs = test_model(single_images)
    loss = test_criterion(outputs, single_labels)
    loss.backward()
    test_optimizer.step()
    
    # Check accuracy
    _, preds = outputs.max(1)
    acc = 100. * preds.eq(single_labels).sum().item() / single_labels.size(0)
    
    if i % 5 == 0:
        print(f"   Iter {i}: Loss={loss.item():.3f}, Acc={acc:.1f}%")

if acc < 90:
    print("   ❌ Model failed to overfit on single batch - serious issue!")
else:
    print("   ✓ Model can overfit on single batch - optimizer works")

# 2. Check learning rate impact
print("\n2. LEARNING RATE TEST:")
print(f"   Current LR: {config['learning_rate']}")
print(f"   Suggested LR: {config.get('max_lr', 'Not set')}")

# Test different learning rates
test_lrs = [1e-4, 1e-3, 1e-2, 1e-1]
print("\n   Testing different learning rates on single batch:")

for lr in test_lrs:
    # Fresh model
    lr_model = resnet50(num_classes=1000, pretrained=False).to(device)
    lr_optimizer = torch.optim.SGD(lr_model.parameters(), lr=lr)
    
    # Train for 5 steps
    lr_model.train()
    losses = []
    for _ in range(5):
        lr_optimizer.zero_grad()
        outputs = lr_model(single_images)
        loss = test_criterion(outputs, single_labels)
        loss.backward()
        lr_optimizer.step()
        losses.append(loss.item())
    
    loss_change = losses[0] - losses[-1]
    print(f"   LR={lr:.0e}: Loss change={loss_change:.3f} "
          f"({'good' if loss_change > 0.1 else 'too small'})")

# 3. Check gradient flow
print("\n3. GRADIENT FLOW CHECK:")
test_model = resnet50(num_classes=1000, pretrained=False).to(device)
test_optimizer = torch.optim.SGD(test_model.parameters(), lr=0.01)

# Forward and backward pass
test_model.train()
outputs = test_model(single_images)
loss = test_criterion(outputs, single_labels)
loss.backward()

# Check gradients in different layers
print("   Gradient norms by layer type:")
layer_grads = {}
for name, param in test_model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.data.norm(2).item()
        layer_type = name.split('.')[0]
        if layer_type not in layer_grads:
            layer_grads[layer_type] = []
        layer_grads[layer_type].append(grad_norm)

for layer_type, grads in layer_grads.items():
    avg_grad = np.mean(grads)
    print(f"   {layer_type}: avg_grad_norm={avg_grad:.6f}")

# 4. Check if strong augmentations are preventing learning
print("\n4. AUGMENTATION IMPACT TEST:")
print(f"   Current augmentation: {dataset_stats.get('augmentation_strength', 'unknown')}")

# Compare with and without augmentation
print("\n   Try training with weaker augmentations:")
print("   Change to: augmentation_strength='basic' or 'medium'")

# 5. Common issues and solutions
print("\n5. DEBUGGING RECOMMENDATIONS:")
print("\n   Based on the tests above, try these fixes:")
print("\n   a) If single batch doesn't overfit:")
print("      • Model architecture issue")
print("      • Check final layer output dimension")
print("      • Try without replace_maxpool_with_conv")
print("\n   b) If gradients are too small:")
print("      • Increase learning rate 10x")
print("      • Check weight initialization")
print("\n   c) If loss doesn't decrease:")
print("      • Reduce learning rate 10x")
print("      • Try different optimizer (Adam)")
print("      • Disable OneCycleLR initially")
print("\n   d) Quick fixes to try:")
print("      config['learning_rate'] = 0.01  # Fixed LR")
print("      config['scheduler'] = None  # No scheduler")
print("      config['weight_decay'] = 0  # No weight decay")
print("      augmentation_strength='basic'  # Weaker augmentation")

# 6. Minimal working example
print("\n6. MINIMAL WORKING CONFIG:")
print("""
# Try this minimal config that should work:
config = {
    'dataset_type': 'small',  # 1% of data
    'batch_size': 64,
    'epochs': 10,
    'learning_rate': 0.01,  # Fixed, reasonable LR
    'momentum': 0.9,
    'weight_decay': 0,  # No regularization
    'scheduler': None,  # No LR scheduling
    'pretrained': False,
    'find_lr': False,  # Skip LR finder
}

# And use basic augmentation:
train_loader, val_loader, dataset_stats = create_imagenet_loaders(
    ...,
    augmentation_strength='basic'  # Not 'strong'
)
""")

print("\n" + "="*60)