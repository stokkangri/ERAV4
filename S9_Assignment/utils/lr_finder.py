"""
Learning Rate Finder utility for finding optimal learning rate
Based on the paper: https://arxiv.org/abs/1506.01186
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any
from scipy.ndimage import gaussian_filter1d
import copy
from tqdm import tqdm


class LRFinder:
    """
    Learning Rate Finder to find optimal learning rate for training
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cuda'
    ):
        """
        Initialize LR Finder
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            criterion: Loss function
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Store initial state
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())
        
        # History
        self.history = {'lr': [], 'loss': []}
        
    def range_test(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: Optional[int] = None,
        step_mode: str = 'exp',
        smooth_f: float = 0.05,
        diverge_th: float = 5,
        accumulation_steps: int = 1
    ) -> None:
        """
        Perform learning rate range test
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations (default: length of train_loader)
            step_mode: 'exp' for exponential or 'linear' for linear increase
            smooth_f: Smoothing factor for loss
            diverge_th: Divergence threshold to stop early
            accumulation_steps: Gradient accumulation steps
        """
        # Reset history
        self.history = {'lr': [], 'loss': []}
        
        # Set model to training mode
        self.model.train()
        
        # Number of iterations
        if num_iter is None:
            num_iter = len(train_loader)
        else:
            num_iter = min(num_iter, len(train_loader))
        
        # Learning rate schedule
        if step_mode == 'exp':
            lr_schedule = np.geomspace(start_lr, end_lr, num_iter)
        else:
            lr_schedule = np.linspace(start_lr, end_lr, num_iter)
        
        # Initialize
        iterator = iter(train_loader)
        best_loss = None
        avg_loss = 0
        smoothed_loss = 0
        
        # Progress bar
        pbar = tqdm(range(num_iter), desc='LR Finder')
        
        for iteration in pbar:
            # Get batch
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, targets = next(iterator)
            
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Update learning rate
            lr = lr_schedule[iteration]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (iteration + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update loss
            avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss
            smoothed_loss = avg_loss / (1 - smooth_f ** (iteration + 1))
            
            # Record
            self.history['lr'].append(lr)
            self.history['loss'].append(smoothed_loss)
            
            # Update progress bar
            pbar.set_postfix({'lr': f'{lr:.2e}', 'loss': f'{smoothed_loss:.4f}'})
            
            # Check for divergence (but ensure we get minimum iterations)
            if iteration >= 20:  # Ensure minimum iterations before checking divergence
                if best_loss is None:
                    best_loss = smoothed_loss
                else:
                    if smoothed_loss > diverge_th * best_loss:
                        print(f"Stopping early, loss diverged at lr={lr:.2e}")
                        break
                    if smoothed_loss < best_loss:
                        best_loss = smoothed_loss
            elif best_loss is None or smoothed_loss < best_loss:
                best_loss = smoothed_loss
        
        print(f"LR range test complete. Best loss: {best_loss:.4f}")
    
    def find_lr(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 1,
        num_iter: int = 100,
        smooth_window: int = 10
    ) -> float:
        """
        Find optimal learning rate automatically
        
        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations
            smooth_window: Window size for smoothing
        
        Returns:
            Suggested learning rate
        """
        # Run range test
        self.range_test(
            train_loader,
            start_lr=start_lr,
            end_lr=end_lr,
            num_iter=num_iter
        )
        
        # Get losses and learning rates
        lrs = np.array(self.history['lr'])
        losses = np.array(self.history['loss'])
        
        # Smooth the losses
        if len(losses) > smooth_window:
            losses = gaussian_filter1d(losses, sigma=smooth_window)
        
        # Compute gradient
        gradients = np.gradient(losses)
        
        # Find steepest gradient (most negative)
        min_gradient_idx = np.argmin(gradients)
        suggested_lr = lrs[min_gradient_idx]
        
        # Alternative: find minimum loss
        min_loss_idx = np.argmin(losses)
        min_loss_lr = lrs[min_loss_idx]
        
        # Use a bit before the minimum (more conservative)
        if min_gradient_idx > 10:
            suggested_lr = lrs[min_gradient_idx - 10]
        
        print(f"Suggested LR (steepest): {suggested_lr:.2e}")
        print(f"LR at minimum loss: {min_loss_lr:.2e}")
        
        return suggested_lr
    
    def plot(
        self,
        skip_start: int = 10,
        skip_end: int = 5,
        log_lr: bool = True,
        show_lr: Optional[float] = None,
        ax: Optional[plt.Axes] = None
    ) -> None:
        """
        Plot the learning rate vs loss
        
        Args:
            skip_start: Number of iterations to skip at start
            skip_end: Number of iterations to skip at end
            log_lr: Use log scale for learning rate axis
            show_lr: Mark a specific learning rate on the plot
            ax: Matplotlib axes to plot on
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        lrs = self.history['lr'][skip_start:-skip_end] if skip_end > 0 else self.history['lr'][skip_start:]
        losses = self.history['loss'][skip_start:-skip_end] if skip_end > 0 else self.history['loss'][skip_start:]
        
        ax.plot(lrs, losses, label='Loss')
        
        if log_lr:
            ax.set_xscale('log')
        
        # Mark suggested LR if provided
        if show_lr:
            ax.axvline(x=show_lr, color='red', linestyle='--', label=f'LR = {show_lr:.2e}')
        
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Rate Finder')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_with_suggestion(
        self,
        skip_start: int = 10,
        skip_end: int = 5,
        smooth_window: int = 10
    ) -> Tuple[float, float]:
        """
        Plot with suggested learning rates marked
        
        Args:
            skip_start: Number of iterations to skip at start
            skip_end: Number of iterations to skip at end
            smooth_window: Window size for smoothing
        
        Returns:
            Tuple of (suggested_lr, min_loss_lr)
        """
        # Check if we have enough data points
        if len(self.history['lr']) < 20:
            print(f"Warning: Only {len(self.history['lr'])} data points collected. Need more iterations for accurate LR finding.")
            # Return a reasonable default
            if len(self.history['lr']) > 0:
                # Use the middle LR as a conservative estimate
                mid_idx = len(self.history['lr']) // 2
                suggested_lr = self.history['lr'][mid_idx]
                return suggested_lr, suggested_lr
            else:
                return 1e-3, 1e-3  # Default fallback
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Adjust skip parameters if we don't have enough data
        if len(self.history['lr']) < skip_start + skip_end + 10:
            skip_start = min(skip_start, len(self.history['lr']) // 4)
            skip_end = min(skip_end, len(self.history['lr']) // 4)
        
        # Get data
        lrs = np.array(self.history['lr'][skip_start:-skip_end] if skip_end > 0 else self.history['lr'][skip_start:])
        losses = np.array(self.history['loss'][skip_start:-skip_end] if skip_end > 0 else self.history['loss'][skip_start:])
        
        # Check if we have enough data after skipping
        if len(lrs) < 5:
            print("Warning: Not enough data points after skipping. Using all available data.")
            lrs = np.array(self.history['lr'])
            losses = np.array(self.history['loss'])
        
        # Original plot
        ax1.plot(lrs, losses, label='Original Loss', alpha=0.7)
        ax1.set_xscale('log')
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('Loss')
        ax1.set_title('Learning Rate Finder - Original')
        ax1.grid(True, alpha=0.3)
        
        # Adjust smoothing window if needed
        smooth_window = min(smooth_window, len(losses) // 2)
        if smooth_window < 1:
            smooth_window = 1
        
        # Smoothed plot with suggestions
        if len(losses) > 3:
            smoothed_losses = gaussian_filter1d(losses, sigma=smooth_window)
        else:
            smoothed_losses = losses
            
        ax2.plot(lrs, losses, label='Original Loss', alpha=0.3)
        ax2.plot(lrs, smoothed_losses, label='Smoothed Loss', color='red', linewidth=2)
        
        # Find suggestions (with safety checks)
        if len(smoothed_losses) > 2:
            gradients = np.gradient(smoothed_losses)
            min_gradient_idx = np.argmin(gradients)
            min_loss_idx = np.argmin(smoothed_losses)
        else:
            min_gradient_idx = len(smoothed_losses) // 2
            min_loss_idx = np.argmin(smoothed_losses)
        
        suggested_lr = lrs[min_gradient_idx]
        min_loss_lr = lrs[min_loss_idx]
        
        # Mark suggestions
        ax2.scatter(suggested_lr, smoothed_losses[min_gradient_idx], 
                   color='blue', s=100, zorder=5, 
                   label=f'Steepest: {suggested_lr:.2e}')
        ax2.scatter(min_loss_lr, smoothed_losses[min_loss_idx], 
                   color='green', s=100, zorder=5, 
                   label=f'Min Loss: {min_loss_lr:.2e}')
        
        ax2.set_xscale('log')
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Loss')
        ax2.set_title('Learning Rate Finder - Smoothed with Suggestions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return suggested_lr, min_loss_lr
    
    def reset(self) -> None:
        """
        Reset model and optimizer to initial state
        """
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        print("Model and optimizer reset to initial state")


def find_optimal_lr(
    model: nn.Module,
    train_loader: DataLoader,
    device: str = 'cuda',
    start_lr: float = 1e-7,
    end_lr: float = 1,
    num_iter: int = 200
) -> Tuple[float, LRFinder]:
    """
    Quick function to find optimal learning rate
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        device: Device to use
        start_lr: Starting learning rate
        end_lr: Ending learning rate
        num_iter: Number of iterations
    
    Returns:
        Tuple of (suggested_lr, lr_finder_object)
    """
    # Create optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=0.9, weight_decay=1e-4)
    
    # Create LR finder
    lr_finder = LRFinder(model, optimizer, criterion, device)
    
    # Run range test
    lr_finder.range_test(
        train_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode='exp'
    )
    
    # Find optimal LR
    suggested_lr = lr_finder.find_lr(
        train_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter
    )
    
    # Plot results
    lr_finder.plot_with_suggestion()
    
    # Reset model
    lr_finder.reset()
    
    return suggested_lr, lr_finder


if __name__ == "__main__":
    # Test the LR finder
    print("Testing LR Finder...")
    
    # Import necessary modules
    import sys
    sys.path.append('..')
    from models.resnet50 import resnet50
    from dataset.cifar100_loader import CIFAR100DataLoader
    
    # Create model
    model = resnet50(num_classes=100)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create data loader
    data_loader = CIFAR100DataLoader(batch_size=128)
    train_loader, _ = data_loader.get_loaders()
    
    # Find optimal LR
    print("Finding optimal learning rate...")
    suggested_lr, lr_finder = find_optimal_lr(
        model,
        train_loader,
        device=device,
        num_iter=200
    )
    
    print(f"\nOptimal learning rate: {suggested_lr:.2e}")
    print("LR Finder test completed!")