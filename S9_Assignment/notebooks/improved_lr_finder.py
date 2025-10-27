"""
Improved Learning Rate Finder with better parameter defaults
to show the full loss curve (decrease -> increase pattern)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import sys
sys.path.append('..')

from utils.lr_finder import LRFinder


def run_improved_lr_finder(
    model: nn.Module,
    train_loader: DataLoader,
    device: str = 'cuda',
    start_lr: float = 1e-8,  # Start even lower
    end_lr: float = 100,     # Go higher to see divergence
    num_iter: int = 300,     # More iterations for better curve
    smooth_f: float = 0.98,  # Higher smoothing for cleaner curve
    diverge_th: float = 4    # Lower threshold to capture more of the curve
) -> Tuple[float, LRFinder]:
    """
    Run LR finder with improved parameters to see full loss curve
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        device: Device to use
        start_lr: Starting learning rate (very low)
        end_lr: Ending learning rate (very high)
        num_iter: Number of iterations
        smooth_f: Smoothing factor (0.98 = heavy smoothing)
        diverge_th: Divergence threshold
    
    Returns:
        Tuple of (suggested_lr, lr_finder_object)
    """
    print("Running improved LR finder with better parameters...")
    print(f"LR range: {start_lr:.2e} to {end_lr:.2e}")
    print(f"Iterations: {num_iter}")
    print(f"Smoothing: {smooth_f}")
    
    # Create optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=0.9, weight_decay=1e-4)
    
    # Create LR finder
    lr_finder = LRFinder(model, optimizer, criterion, device)
    
    # Run range test with improved parameters
    lr_finder.range_test(
        train_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode='exp',
        smooth_f=smooth_f,
        diverge_th=diverge_th,
        accumulation_steps=1
    )
    
    # Plot the full curve
    plot_lr_curve_analysis(lr_finder)
    
    # Find optimal LR using different methods
    suggested_lr = find_optimal_lr_advanced(lr_finder)
    
    return suggested_lr, lr_finder


def plot_lr_curve_analysis(lr_finder: LRFinder, skip_start: int = 10, skip_end: int = 5):
    """
    Plot comprehensive LR curve analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Get data
    lrs = np.array(lr_finder.history['lr'])
    losses = np.array(lr_finder.history['loss'])
    
    # Skip initial unstable part
    if len(lrs) > skip_start + skip_end:
        lrs = lrs[skip_start:-skip_end] if skip_end > 0 else lrs[skip_start:]
        losses = losses[skip_start:-skip_end] if skip_end > 0 else losses[skip_start:]
    
    # 1. Original loss curve
    ax1 = axes[0, 0]
    ax1.plot(lrs, losses, 'b-', alpha=0.8)
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate (log scale)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Learning Rate')
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss with gradient
    ax2 = axes[0, 1]
    gradients = np.gradient(losses)
    ax2_twin = ax2.twinx()
    
    ax2.plot(lrs, losses, 'b-', label='Loss')
    ax2_twin.plot(lrs, gradients, 'r-', alpha=0.7, label='Gradient')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Learning Rate (log scale)')
    ax2.set_ylabel('Loss', color='b')
    ax2_twin.set_ylabel('Gradient', color='r')
    ax2.set_title('Loss and Gradient')
    ax2.grid(True, alpha=0.3)
    
    # 3. Smoothed loss with suggestions
    ax3 = axes[1, 0]
    from scipy.ndimage import gaussian_filter1d
    smoothed_losses = gaussian_filter1d(losses, sigma=5)
    
    ax3.plot(lrs, losses, 'b-', alpha=0.3, label='Original')
    ax3.plot(lrs, smoothed_losses, 'r-', linewidth=2, label='Smoothed')
    
    # Find key points
    min_loss_idx = np.argmin(smoothed_losses)
    gradients_smooth = np.gradient(smoothed_losses)
    steepest_idx = np.argmin(gradients_smooth)
    
    # Find where loss starts increasing significantly
    increasing_idx = min_loss_idx
    for i in range(min_loss_idx, len(smoothed_losses)-1):
        if smoothed_losses[i+1] > smoothed_losses[i] * 1.05:  # 5% increase
            increasing_idx = i
            break
    
    # Mark important points
    ax3.scatter(lrs[steepest_idx], smoothed_losses[steepest_idx], 
               color='green', s=100, zorder=5, label=f'Steepest: {lrs[steepest_idx]:.2e}')
    ax3.scatter(lrs[min_loss_idx], smoothed_losses[min_loss_idx], 
               color='blue', s=100, zorder=5, label=f'Min Loss: {lrs[min_loss_idx]:.2e}')
    ax3.scatter(lrs[increasing_idx], smoothed_losses[increasing_idx], 
               color='orange', s=100, zorder=5, label=f'Start Increase: {lrs[increasing_idx]:.2e}')
    
    ax3.set_xscale('log')
    ax3.set_xlabel('Learning Rate (log scale)')
    ax3.set_ylabel('Loss')
    ax3.set_title('Smoothed Loss with Key Points')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Learning rate suggestions
    ax4 = axes[1, 1]
    ax4.text(0.1, 0.9, 'Learning Rate Suggestions:', transform=ax4.transAxes, 
             fontsize=14, fontweight='bold')
    
    suggestions = {
        'Steepest Descent': lrs[steepest_idx],
        'Minimum Loss': lrs[min_loss_idx],
        'Before Divergence': lrs[increasing_idx],
        'Conservative (10x before min)': lrs[max(0, min_loss_idx - 20)],
        'Aggressive (at steepest)': lrs[steepest_idx],
        'Recommended (between steep and min)': np.sqrt(lrs[steepest_idx] * lrs[min_loss_idx])
    }
    
    y_pos = 0.7
    for name, lr in suggestions.items():
        ax4.text(0.1, y_pos, f'{name}: {lr:.2e}', transform=ax4.transAxes, fontsize=12)
        y_pos -= 0.1
    
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return suggestions


def find_optimal_lr_advanced(lr_finder: LRFinder, skip_start: int = 10, skip_end: int = 5) -> float:
    """
    Find optimal LR using advanced analysis
    """
    # Get data
    lrs = np.array(lr_finder.history['lr'])
    losses = np.array(lr_finder.history['loss'])
    
    # Skip initial unstable part
    if len(lrs) > skip_start + skip_end:
        lrs = lrs[skip_start:-skip_end] if skip_end > 0 else lrs[skip_start:]
        losses = losses[skip_start:-skip_end] if skip_end > 0 else losses[skip_start:]
    
    # Smooth the losses
    from scipy.ndimage import gaussian_filter1d
    smoothed_losses = gaussian_filter1d(losses, sigma=5)
    
    # Find key points
    min_loss_idx = np.argmin(smoothed_losses)
    gradients = np.gradient(smoothed_losses)
    steepest_idx = np.argmin(gradients)
    
    # Recommended LR: geometric mean between steepest and minimum
    recommended_lr = np.sqrt(lrs[steepest_idx] * lrs[min_loss_idx])
    
    print(f"\n=== LR Finder Analysis ===")
    print(f"Steepest descent at: {lrs[steepest_idx]:.2e}")
    print(f"Minimum loss at: {lrs[min_loss_idx]:.2e}")
    print(f"Recommended LR: {recommended_lr:.2e}")
    
    return recommended_lr


def test_lr_parameters(model, train_loader, device='cuda'):
    """
    Test different LR finder parameters to find the best settings
    """
    print("Testing different LR finder parameters...")
    
    # Test configurations
    configs = [
        {'start_lr': 1e-8, 'end_lr': 10, 'num_iter': 200, 'smooth_f': 0.95},
        {'start_lr': 1e-9, 'end_lr': 100, 'num_iter': 300, 'smooth_f': 0.98},
        {'start_lr': 1e-7, 'end_lr': 1, 'num_iter': 150, 'smooth_f': 0.9},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1} ---")
        print(f"Parameters: {config}")
        
        # Create fresh optimizer
        optimizer = optim.SGD(model.parameters(), lr=config['start_lr'], 
                            momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Run LR finder
        lr_finder = LRFinder(model, optimizer, criterion, device)
        lr_finder.range_test(
            train_loader,
            start_lr=config['start_lr'],
            end_lr=config['end_lr'],
            num_iter=config['num_iter'],
            step_mode='exp',
            smooth_f=config['smooth_f']
        )
        
        # Quick plot
        lr_finder.plot(skip_start=10, skip_end=5)
        
        # Reset model
        lr_finder.reset()


# Example usage
if __name__ == "__main__":
    print("Improved LR Finder Example")
    
    # Example with dummy model and data
    # Replace with your actual model and data loader
    
    # Create a simple model for testing
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Create dummy data
    from torch.utils.data import TensorDataset
    X = torch.randn(1000, 10)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Run improved LR finder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    suggested_lr, lr_finder = run_improved_lr_finder(
        model, 
        train_loader, 
        device,
        start_lr=1e-8,
        end_lr=100,
        num_iter=300
    )
    
    print(f"\nFinal recommended LR: {suggested_lr:.2e}")