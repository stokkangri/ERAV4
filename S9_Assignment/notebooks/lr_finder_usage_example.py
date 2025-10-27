"""
Example of how to use the improved LR finder with your ResNet50 training
"""

import torch
import torch.nn as nn
import sys
sys.path.append('..')

from models.resnet50_imagenet import resnet50
from dataset.imagenet_loader import create_imagenet_loaders
from improved_lr_finder import run_improved_lr_finder, test_lr_parameters
import yaml


def find_optimal_lr_for_training():
    """
    Complete example of finding optimal LR for your training setup
    """
    # Load configuration
    with open('../configs/config_small_subset.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, dataset_stats = create_imagenet_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        subset_percent=config.get('subset_percent', 0.1),  # Use small subset for LR finding
        tiny_imagenet=config.get('tiny_imagenet', False),
        augment_train=True,
        augmentation_strength='medium'  # Use medium augmentation for LR finding
    )
    
    print(f"Dataset: {dataset_stats['num_classes']} classes")
    print(f"Training samples: {dataset_stats['train_samples']}")
    
    # Create model
    print("\nCreating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50(num_classes=dataset_stats['num_classes'], pretrained=False)
    model = model.to(device)
    
    # Method 1: Run improved LR finder with better parameters
    print("\n" + "="*50)
    print("METHOD 1: Improved LR Finder")
    print("="*50)
    
    suggested_lr, lr_finder = run_improved_lr_finder(
        model=model,
        train_loader=train_loader,
        device=device,
        start_lr=1e-8,   # Very low start to see initial plateau
        end_lr=100,      # Very high end to see divergence
        num_iter=300,    # More iterations for complete curve
        smooth_f=0.98,   # Heavy smoothing for cleaner curve
        diverge_th=4     # Lower threshold to capture more curve
    )
    
    print(f"\nSuggested learning rate: {suggested_lr:.2e}")
    
    # Reset model for next test
    lr_finder.reset()
    
    # Method 2: Test multiple parameter configurations
    print("\n" + "="*50)
    print("METHOD 2: Testing Different Parameters")
    print("="*50)
    
    test_lr_parameters(model, train_loader, device)
    
    # Method 3: Custom parameter search for your specific case
    print("\n" + "="*50)
    print("METHOD 3: Custom Parameter Search")
    print("="*50)
    
    # If you're not seeing the loss drop, try these parameters
    custom_configs = [
        {
            'name': 'Ultra-wide range',
            'start_lr': 1e-10,
            'end_lr': 1000,
            'num_iter': 400,
            'smooth_f': 0.99,
            'diverge_th': 10
        },
        {
            'name': 'Focus on low LRs',
            'start_lr': 1e-9,
            'end_lr': 1e-2,
            'num_iter': 300,
            'smooth_f': 0.95,
            'diverge_th': 5
        },
        {
            'name': 'Standard range with more iterations',
            'start_lr': 1e-7,
            'end_lr': 10,
            'num_iter': 500,
            'smooth_f': 0.98,
            'diverge_th': 4
        }
    ]
    
    best_lr = None
    best_config = None
    
    for config_test in custom_configs:
        print(f"\nTesting: {config_test['name']}")
        print(f"Range: {config_test['start_lr']:.2e} to {config_test['end_lr']:.2e}")
        
        # Create fresh optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=config_test['start_lr'], 
                                  momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Create LR finder
        from utils.lr_finder import LRFinder
        lr_finder = LRFinder(model, optimizer, criterion, device)
        
        # Run range test
        lr_finder.range_test(
            train_loader,
            start_lr=config_test['start_lr'],
            end_lr=config_test['end_lr'],
            num_iter=config_test['num_iter'],
            step_mode='exp',
            smooth_f=config_test['smooth_f'],
            diverge_th=config_test['diverge_th']
        )
        
        # Analyze results
        lrs = lr_finder.history['lr']
        losses = lr_finder.history['loss']
        
        if len(losses) > 20:
            # Find if we have a proper curve
            min_loss = min(losses[10:])  # Skip initial instability
            max_loss = max(losses[10:])
            loss_range = max_loss - min_loss
            
            print(f"Loss range: {min_loss:.3f} to {max_loss:.3f} (range: {loss_range:.3f})")
            
            # Check if we see both decrease and increase
            min_idx = losses[10:].index(min_loss) + 10
            has_decrease = any(losses[i] > losses[i+1] for i in range(10, min_idx-1))
            has_increase = any(losses[i] < losses[i+1] for i in range(min_idx, len(losses)-1))
            
            print(f"Has decrease phase: {has_decrease}")
            print(f"Has increase phase: {has_increase}")
            
            if has_decrease and has_increase:
                print("✓ Good curve shape detected!")
                suggested_lr_test = lrs[min_idx] / 10  # Conservative: 10x before minimum
                print(f"Suggested LR: {suggested_lr_test:.2e}")
                
                if best_lr is None or suggested_lr_test > best_lr * 0.1:
                    best_lr = suggested_lr_test
                    best_config = config_test
        
        # Plot
        lr_finder.plot(skip_start=10, skip_end=5)
        
        # Reset
        lr_finder.reset()
    
    print("\n" + "="*50)
    print("FINAL RECOMMENDATIONS")
    print("="*50)
    
    if best_lr:
        print(f"Best LR found: {best_lr:.2e}")
        print(f"Best configuration: {best_config['name']}")
        print(f"\nFor OneCycleLR, use:")
        print(f"  max_lr = {best_lr:.2e}")
        print(f"  base_lr = {best_lr/10:.2e}")
    else:
        print("Could not find optimal LR automatically.")
        print("Please check the plots and choose manually.")
    
    return best_lr


def diagnose_lr_finder_issues(model, train_loader, device='cuda'):
    """
    Diagnose why LR finder might not be showing expected curve
    """
    print("Diagnosing LR Finder Issues...")
    
    # Test 1: Check if model outputs are reasonable
    print("\n1. Testing model outputs...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        images, labels = sample_batch
        images = images.to(device)
        outputs = model(images)
        
        print(f"Output shape: {outputs.shape}")
        print(f"Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
        
        # Check if outputs are too large/small
        if outputs.abs().max() > 100:
            print("⚠️  WARNING: Model outputs are very large. Consider adding normalization.")
        elif outputs.abs().max() < 0.01:
            print("⚠️  WARNING: Model outputs are very small. Check initialization.")
    
    # Test 2: Check loss at different LRs manually
    print("\n2. Testing loss at specific LRs...")
    test_lrs = [1e-8, 1e-6, 1e-4, 1e-2, 1, 10, 100]
    criterion = nn.CrossEntropyLoss()
    
    for lr in test_lrs:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        # Take one step
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        print(f"LR: {lr:.2e}, Loss: {loss.item():.4f}")
        
        # Don't actually update weights
        loss.backward()
        optimizer.zero_grad()
    
    # Test 3: Check gradient magnitudes
    print("\n3. Checking gradient magnitudes...")
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    
    total_norm = 0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** 0.5
    avg_norm = total_norm / param_count if param_count > 0 else 0
    
    print(f"Total gradient norm: {total_norm:.4f}")
    print(f"Average gradient norm: {avg_norm:.4f}")
    
    if total_norm > 1000:
        print("⚠️  WARNING: Gradients are very large. Consider gradient clipping.")
    elif total_norm < 0.001:
        print("⚠️  WARNING: Gradients are very small. Check loss function and data.")
    
    optimizer.zero_grad()


if __name__ == "__main__":
    # Run the complete LR finding process
    optimal_lr = find_optimal_lr_for_training()
    
    # If you're still having issues, run diagnostics
    print("\n" + "="*50)
    print("Running diagnostics...")
    print("="*50)
    
    # You would need to recreate model and loader here for diagnostics
    # diagnose_lr_finder_issues(model, train_loader)