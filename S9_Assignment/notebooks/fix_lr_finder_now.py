"""
Quick fix for LR finder not showing proper loss curve
Run this script to find the optimal learning rate with improved parameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys
sys.path.append('..')

from utils.lr_finder import LRFinder
from models.resnet50_imagenet import resnet50
from dataset.imagenet_loader import create_imagenet_loaders
import yaml


def run_fixed_lr_finder():
    """
    Run LR finder with parameters that will show the full curve
    """
    print("Loading configuration...")
    # Load your config
    with open('../configs/config_small_subset.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Creating data loaders...")
    # Create data loaders with small subset for faster LR finding
    train_loader, val_loader, dataset_stats = create_imagenet_loaders(
        data_dir=config['data_dir'],
        batch_size=64,  # Smaller batch size for LR finder
        num_workers=config['num_workers'],
        subset_percent=0.05,  # Use only 5% of data for LR finding
        tiny_imagenet=config.get('tiny_imagenet', False),
        augment_train=True,
        augmentation_strength='medium'
    )
    
    print(f"Dataset: {dataset_stats['num_classes']} classes")
    print(f"Training samples for LR finder: {dataset_stats['train_samples']}")
    
    # Create model
    print("\nCreating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50(num_classes=dataset_stats['num_classes'], pretrained=False)
    model = model.to(device)
    
    # Create optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-10, momentum=0.9, weight_decay=1e-4)
    
    # Create LR finder with fixed parameters
    print("\nRunning LR finder with improved parameters...")
    print("This will take a few minutes...")
    
    lr_finder = LRFinder(model, optimizer, criterion, device)
    
    # Run with parameters that WILL show the curve
    lr_finder.range_test(
        train_loader,
        start_lr=1e-10,  # Ultra low start
        end_lr=100,      # High end to see divergence
        num_iter=300,    # Enough iterations
        step_mode='exp',
        smooth_f=0.98,   # Heavy smoothing
        diverge_th=10,   # Higher threshold
        accumulation_steps=1
    )
    
    print("\nLR finder complete! Analyzing results...")
    
    # Analyze and plot results
    plot_and_analyze_results(lr_finder)
    
    # Find optimal LR
    optimal_lr = find_optimal_lr_from_curve(lr_finder)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print(f"Optimal Learning Rate: {optimal_lr:.2e}")
    print(f"\nFor OneCycleLR scheduler:")
    print(f"  max_lr: {optimal_lr:.2e}")
    print(f"  base_lr: {optimal_lr/10:.2e}")
    print(f"\nFor regular training:")
    print(f"  learning_rate: {optimal_lr/2:.2e} (conservative)")
    print(f"  learning_rate: {optimal_lr:.2e} (aggressive)")
    
    # Reset model
    lr_finder.reset()
    
    return optimal_lr, lr_finder


def plot_and_analyze_results(lr_finder):
    """
    Create comprehensive plots of LR finder results
    """
    # Get data
    lrs = np.array(lr_finder.history['lr'])
    losses = np.array(lr_finder.history['loss'])
    
    # Skip initial unstable part
    skip_start = min(20, len(lrs) // 10)
    skip_end = 5
    
    if len(lrs) > skip_start + skip_end:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Learning Rate Finder Analysis', fontsize=16)
    
    # 1. Raw loss curve
    ax1 = axes[0, 0]
    ax1.plot(lrs, losses, 'b-', alpha=0.8)
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate (log scale)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Raw Loss vs Learning Rate')
    ax1.grid(True, alpha=0.3)
    
    # 2. Smoothed loss curve
    ax2 = axes[0, 1]
    smoothed_losses = gaussian_filter1d(losses, sigma=5)
    ax2.plot(lrs, losses, 'b-', alpha=0.3, label='Raw')
    ax2.plot(lrs, smoothed_losses, 'r-', linewidth=2, label='Smoothed')
    ax2.set_xscale('log')
    ax2.set_xlabel('Learning Rate (log scale)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Smoothed Loss Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss gradient
    ax3 = axes[1, 0]
    gradients = np.gradient(smoothed_losses)
    ax3.plot(lrs, gradients, 'g-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xscale('log')
    ax3.set_xlabel('Learning Rate (log scale)')
    ax3.set_ylabel('Loss Gradient')
    ax3.set_title('Rate of Change of Loss')
    ax3.grid(True, alpha=0.3)
    
    # 4. Key points identification
    ax4 = axes[1, 1]
    
    # Find key points
    min_loss_idx = np.argmin(smoothed_losses)
    steepest_idx = np.argmin(gradients)
    
    # Find where loss starts increasing
    increase_idx = min_loss_idx
    for i in range(min_loss_idx, len(smoothed_losses)-1):
        if smoothed_losses[i+1] > smoothed_losses[i] * 1.02:
            increase_idx = i
            break
    
    # Plot with key points
    ax4.plot(lrs, smoothed_losses, 'b-', linewidth=2)
    ax4.scatter(lrs[steepest_idx], smoothed_losses[steepest_idx], 
               color='green', s=100, zorder=5, label=f'Steepest: {lrs[steepest_idx]:.2e}')
    ax4.scatter(lrs[min_loss_idx], smoothed_losses[min_loss_idx], 
               color='red', s=100, zorder=5, label=f'Minimum: {lrs[min_loss_idx]:.2e}')
    ax4.scatter(lrs[increase_idx], smoothed_losses[increase_idx], 
               color='orange', s=100, zorder=5, label=f'Divergence: {lrs[increase_idx]:.2e}')
    
    # Suggested LR (geometric mean of steepest and minimum)
    suggested_lr = np.sqrt(lrs[steepest_idx] * lrs[min_loss_idx])
    ax4.axvline(x=suggested_lr, color='purple', linestyle='--', 
               label=f'Suggested: {suggested_lr:.2e}')
    
    ax4.set_xscale('log')
    ax4.set_xlabel('Learning Rate (log scale)')
    ax4.set_ylabel('Loss')
    ax4.set_title('Key Learning Rates')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print("\n" + "="*60)
    print("ANALYSIS RESULTS:")
    print("="*60)
    print(f"Loss range: {losses.min():.4f} to {losses.max():.4f}")
    print(f"Steepest descent at LR: {lrs[steepest_idx]:.2e}")
    print(f"Minimum loss at LR: {lrs[min_loss_idx]:.2e}")
    print(f"Loss starts diverging at LR: {lrs[increase_idx]:.2e}")
    print(f"Suggested optimal LR: {suggested_lr:.2e}")
    
    # Check curve quality
    has_descent = steepest_idx < min_loss_idx
    has_increase = increase_idx > min_loss_idx
    
    if has_descent and has_increase:
        print("\n✅ Good curve shape detected! (descent → minimum → increase)")
    elif not has_descent:
        print("\n⚠️  No clear descent phase. Try lower start_lr.")
    elif not has_increase:
        print("\n⚠️  No clear increase phase. Try higher end_lr or more iterations.")


def find_optimal_lr_from_curve(lr_finder):
    """
    Find optimal LR from the curve using multiple methods
    """
    lrs = np.array(lr_finder.history['lr'])
    losses = np.array(lr_finder.history['loss'])
    
    # Skip initial part
    skip = min(20, len(lrs) // 10)
    lrs = lrs[skip:]
    losses = losses[skip:]
    
    # Smooth the losses
    smoothed_losses = gaussian_filter1d(losses, sigma=5)
    
    # Method 1: Steepest descent
    gradients = np.gradient(smoothed_losses)
    steepest_idx = np.argmin(gradients)
    
    # Method 2: Minimum loss
    min_loss_idx = np.argmin(smoothed_losses)
    
    # Method 3: Geometric mean (balanced)
    optimal_lr = np.sqrt(lrs[steepest_idx] * lrs[min_loss_idx])
    
    return optimal_lr


def quick_diagnostic():
    """
    Quick diagnostic if LR finder still doesn't work
    """
    print("\nRunning quick diagnostic...")
    
    # Test with simple model and data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Simple test model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10)
    ).to(device)
    
    # Test data
    from torch.utils.data import TensorDataset, DataLoader
    X = torch.randn(1000, 3, 32, 32)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Test specific LRs
    test_lrs = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 10, 100]
    losses = []
    
    criterion = nn.CrossEntropyLoss()
    
    print("\nTesting specific learning rates:")
    for lr in test_lrs:
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        # Get one batch
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        
        print(f"LR: {lr:.2e}, Loss: {loss.item():.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(test_lrs, losses, 'bo-')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Quick LR Test')
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    print("="*60)
    print("FIXED LR FINDER - This will show the proper loss curve")
    print("="*60)
    
    try:
        # Run the fixed LR finder
        optimal_lr, lr_finder = run_fixed_lr_finder()
        
        print("\n" + "="*60)
        print("SUCCESS! Check the plots above.")
        print("="*60)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("\nRunning diagnostic mode...")
        quick_diagnostic()
        
        print("\nTroubleshooting tips:")
        print("1. Check if your data is loading correctly")
        print("2. Ensure CUDA is available if using GPU")
        print("3. Try reducing batch size")
        print("4. Check the lr_finder_troubleshooting_guide.md for more help")