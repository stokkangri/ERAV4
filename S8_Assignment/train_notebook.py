"""
Notebook-compatible training module for ResNet-50 on CIFAR-100
This module can be imported and used in Jupyter notebooks
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
import pandas as pd

# Import custom modules
from models.resnet50 import resnet50
from dataset.cifar100_loader import CIFAR100DataLoader
from utils.lr_finder import LRFinder
from utils.train_test import train_epoch, test_epoch


class NotebookTrainer:
    """
    Notebook-friendly trainer for ResNet-50 on CIFAR-100
    """
    
    def __init__(
        self,
        batch_size: int = 128,
        epochs: int = 100,
        learning_rate: float = 0.1,
        optimizer_type: str = 'sgd',
        scheduler_type: str = 'onecycle',
        data_dir: str = './data',
        checkpoint_dir: str = './checkpoints',
        device: Optional[str] = None
    ):
        """
        Initialize notebook trainer
        
        Args:
            batch_size: Batch size for training
            epochs: Number of epochs to train
            learning_rate: Initial learning rate
            optimizer_type: Type of optimizer ('sgd' or 'adam')
            scheduler_type: Type of scheduler ('onecycle', 'cosine', 'step', or None)
            data_dir: Directory for CIFAR-100 data
            checkpoint_dir: Directory to save checkpoints
            device: Device to use (None for auto-detect)
        """
        # Configuration
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.data_dir = data_dir
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'test_acc_top5': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_acc = 0
        self.best_model_state = None
    
    def setup(self):
        """Setup all components"""
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_scheduler()
        return self
    
    def setup_data(self):
        """Setup data loaders"""
        print("Loading CIFAR-100 dataset...")
        data_loader = CIFAR100DataLoader(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            num_workers=4,
            augment_train=True
        )
        
        self.train_loader, self.test_loader = data_loader.get_loaders()
        self.dataset_stats = data_loader.get_dataset_stats()
        
        print(f"✓ Dataset loaded")
        print(f"  Train samples: {self.dataset_stats['num_train_samples']:,}")
        print(f"  Test samples: {self.dataset_stats['num_test_samples']:,}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")
    
    def setup_model(self):
        """Setup ResNet-50 model"""
        print("\nInitializing ResNet-50 model...")
        self.model = resnet50(num_classes=100).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Model initialized")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def setup_optimizer(self):
        """Setup optimizer"""
        print(f"\nSetting up {self.optimizer_type.upper()} optimizer...")
        
        if self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=5e-4
            )
        elif self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=5e-4
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
        
        print(f"✓ Optimizer configured with LR={self.learning_rate:.2e}")
    
    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.scheduler_type is None:
            print("\nNo learning rate scheduler")
            return
        
        print(f"\nSetting up {self.scheduler_type} scheduler...")
        
        if self.scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                epochs=self.epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=10.0,
                final_div_factor=1000.0
            )
            print(f"✓ OneCycleLR scheduler configured")
        elif self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=1e-6
            )
            print(f"✓ CosineAnnealingLR scheduler configured")
        elif self.scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
            print(f"✓ StepLR scheduler configured")
    
    def find_lr(self, start_lr: float = 1e-7, end_lr: float = 1, num_iter: int = 200):
        """
        Find optimal learning rate using LR finder
        
        Args:
            start_lr: Starting learning rate
            end_lr: Ending learning rate  
            num_iter: Number of iterations
        
        Returns:
            Suggested learning rate
        """
        print("\n" + "="*50)
        print("Finding Optimal Learning Rate")
        print("="*50)
        
        # Ensure model and data are setup
        if self.model is None:
            self.setup()
        
        # Create temporary optimizer
        temp_optimizer = optim.SGD(
            self.model.parameters(),
            lr=start_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        # Create LR finder
        lr_finder = LRFinder(
            self.model,
            temp_optimizer,
            self.criterion,
            self.device
        )
        
        # Run range test
        lr_finder.range_test(
            self.train_loader,
            start_lr=start_lr,
            end_lr=end_lr,
            num_iter=num_iter,
            step_mode='exp'
        )
        
        # Plot and find optimal LR
        suggested_lr, min_loss_lr = lr_finder.plot_with_suggestion()
        
        # Reset model
        lr_finder.reset()
        
        print(f"\n✓ LR Finder complete")
        print(f"  Suggested LR: {suggested_lr:.2e}")
        print(f"  Min Loss LR: {min_loss_lr:.2e}")
        
        return suggested_lr
    
    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        return train_epoch(
            self.model,
            self.device,
            self.train_loader,
            self.optimizer,
            self.criterion,
            self.scheduler if self.scheduler_type == 'onecycle' else None,
            epoch,
            accumulation_steps=1,
            clip_grad_norm=None,
            verbose=True
        )
    
    def test_one_epoch(self, epoch: int) -> Tuple[float, float, float, List]:
        """Test for one epoch"""
        return test_epoch(
            self.model,
            self.device,
            self.test_loader,
            self.criterion,
            epoch,
            verbose=True,
            calc_top5=True
        )
    
    def train(self, plot_interval: int = 5, save_best: bool = True):
        """
        Train the model
        
        Args:
            plot_interval: Interval for plotting progress
            save_best: Save best model checkpoint
        """
        # Ensure everything is setup
        if self.model is None:
            self.setup()
        
        print("\n" + "="*50)
        print(f"Starting Training for {self.epochs} Epochs")
        print("="*50)
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # Test
            test_loss, test_acc, test_acc_top5, _ = self.test_one_epoch(epoch)
            
            # Update scheduler (non-OneCycle)
            if self.scheduler and self.scheduler_type != 'onecycle':
                self.scheduler.step()
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['test_acc_top5'].append(test_acc_top5)
            self.history['lr'].append(current_lr)
            
            # Save best model
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_model_state = self.model.state_dict().copy()
                if save_best:
                    self.save_checkpoint('best_model.pth')
            
            # Print summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.epochs} Summary:")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%, Top5={test_acc_top5:.2f}%")
            print(f"  LR={current_lr:.2e}, Time={epoch_time:.1f}s, Best={self.best_acc:.2f}%")
            
            # Plot progress
            if (epoch + 1) % plot_interval == 0:
                self.plot_progress(show=True)
        
        print("\n" + "="*50)
        print("Training Completed!")
        print(f"Best Test Accuracy: {self.best_acc:.2f}%")
        print("="*50)
        
        return self.history
    
    def plot_progress(self, show: bool = True, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot training progress
        
        Args:
            show: Show plot immediately
            figsize: Figure size
        """
        if len(self.history['train_loss']) == 0:
            print("No training history to plot")
            return
        
        # Clear output in notebook
        if show:
            clear_output(wait=True)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['test_loss'], 'r-', label='Test', linewidth=2)
        axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Top-1 Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['test_acc'], 'r-', label='Test', linewidth=2)
        axes[0, 1].set_title('Top-1 Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-5 Accuracy
        axes[0, 2].plot(epochs, self.history['test_acc_top5'], 'g-', linewidth=2)
        axes[0, 2].set_title('Top-5 Accuracy (Test)', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy (%)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(epochs, self.history['lr'], 'orange', linewidth=2)
        axes[1, 0].set_title('Learning Rate', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy Gap
        acc_gap = [train - test for train, test in 
                   zip(self.history['train_acc'], self.history['test_acc'])]
        axes[1, 1].plot(epochs, acc_gap, 'purple', linewidth=2)
        axes[1, 1].set_title('Train-Test Gap', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gap (%)')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Summary Statistics
        axes[1, 2].axis('off')
        
        # Create summary table
        summary_data = {
            'Metric': ['Best Train Acc', 'Best Test Acc', 'Best Top-5 Acc',
                      'Final Train Acc', 'Final Test Acc', 'Final Top-5 Acc'],
            'Value': [
                f"{max(self.history['train_acc']):.2f}%",
                f"{max(self.history['test_acc']):.2f}%",
                f"{max(self.history['test_acc_top5']):.2f}%",
                f"{self.history['train_acc'][-1]:.2f}%",
                f"{self.history['test_acc'][-1]:.2f}%",
                f"{self.history['test_acc_top5'][-1]:.2f}%"
            ]
        }
        
        table = plt.table(cellText=[[m, v] for m, v in zip(summary_data['Metric'], 
                                                           summary_data['Value'])],
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.6, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        plt.suptitle('ResNet-50 CIFAR-100 Training Progress', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        filepath = self.checkpoint_dir / filename
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'best_acc': self.best_acc,
            'config': {
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'optimizer_type': self.optimizer_type,
                'scheduler_type': self.scheduler_type
            }
        }
        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        filepath = self.checkpoint_dir / filename
        if not filepath.exists():
            print(f"Checkpoint {filepath} not found")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Ensure model is initialized
        if self.model is None:
            self.setup_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            if self.optimizer is None:
                self.setup_optimizer()
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            if self.scheduler is None:
                self.setup_scheduler()
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        if 'best_acc' in checkpoint:
            self.best_acc = checkpoint['best_acc']
        
        print(f"✓ Checkpoint loaded from {filepath}")
        if 'best_acc' in checkpoint:
            print(f"  Best accuracy: {self.best_acc:.2f}%")
        
        return True
    
    def get_summary_df(self) -> pd.DataFrame:
        """Get training history as pandas DataFrame"""
        return pd.DataFrame(self.history)
    
    def evaluate(self) -> Dict:
        """Evaluate the model on test set"""
        if self.model is None:
            print("Model not initialized")
            return {}
        
        print("Evaluating model on test set...")
        test_loss, test_acc, test_acc_top5, misclassified = test_epoch(
            self.model,
            self.device,
            self.test_loader,
            self.criterion,
            epoch=0,
            verbose=True,
            calc_top5=True
        )
        
        results = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_acc_top5': test_acc_top5,
            'num_misclassified': len(misclassified)
        }
        
        print(f"\nEvaluation Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Test Top-5 Accuracy: {test_acc_top5:.2f}%")
        
        return results


# Convenience functions for notebook usage
def quick_train(
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 0.1,
    find_lr_first: bool = False
) -> NotebookTrainer:
    """
    Quick training function for notebooks
    
    Args:
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        find_lr_first: Find optimal LR before training
    
    Returns:
        Trained NotebookTrainer instance
    """
    print("="*50)
    print("ResNet-50 CIFAR-100 Quick Training")
    print("="*50)
    
    # Create trainer
    trainer = NotebookTrainer(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        scheduler_type='onecycle'
    )
    
    # Setup
    trainer.setup()
    
    # Find LR if requested
    if find_lr_first:
        suggested_lr = trainer.find_lr()
        trainer.learning_rate = suggested_lr
        trainer.setup_optimizer()
        trainer.setup_scheduler()
    
    # Train
    trainer.train(plot_interval=5)
    
    return trainer


if __name__ == "__main__":
    # Example usage
    print("Example notebook training usage:")
    print("-" * 50)
    print("""
# Import the module
from train_notebook import NotebookTrainer, quick_train

# Option 1: Quick training with defaults
trainer = quick_train(epochs=20, find_lr_first=True)

# Option 2: Custom configuration
trainer = NotebookTrainer(
    batch_size=128,
    epochs=100,
    learning_rate=0.1,
    scheduler_type='onecycle'
)
trainer.setup()

# Find optimal learning rate
suggested_lr = trainer.find_lr()

# Train the model
history = trainer.train(plot_interval=5)

# Save the best model
trainer.save_checkpoint('best_model.pth')

# Get training summary
df = trainer.get_summary_df()
print(df.tail())

# Evaluate on test set
results = trainer.evaluate()
""")