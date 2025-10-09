"""
Main training script for ResNet-50 on CIFAR-100
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR
import matplotlib.pyplot as plt
import numpy as np

# Import custom modules
from models.resnet50 import resnet50
from dataset.cifar100_loader import CIFAR100DataLoader
from utils.lr_finder import LRFinder, find_optimal_lr
from utils.train_test import Trainer, train_epoch, test_epoch


class ResNet50Trainer:
    """
    Complete training pipeline for ResNet-50 on CIFAR-100
    """
    
    def __init__(self, config: dict):
        """
        Initialize trainer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        self.setup_directories()
        
        # Initialize components
        self.setup_data()
        self.setup_model()
        self.setup_training()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'test_acc_top5': [],
            'lr': []
        }
        
    def setup_directories(self):
        """Create necessary directories"""
        self.checkpoint_dir = Path(self.config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(self.config['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.plot_dir = Path(self.config['plot_dir'])
        self.plot_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_data(self):
        """Setup data loaders"""
        print("Setting up data loaders...")
        data_loader = CIFAR100DataLoader(
            data_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            augment_train=self.config['augment_train']
        )
        
        self.train_loader, self.test_loader = data_loader.get_loaders()
        self.dataset_stats = data_loader.get_dataset_stats()
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Test batches: {len(self.test_loader)}")
    
    def setup_model(self):
        """Setup model"""
        print("Setting up model...")
        self.model = resnet50(num_classes=100).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def setup_training(self):
        """Setup training components"""
        print("Setting up training components...")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=self.config['momentum'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
        
        # Learning rate scheduler
        if self.config['scheduler'] == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config['max_lr'],
                epochs=self.config['epochs'],
                steps_per_epoch=len(self.train_loader),
                pct_start=self.config.get('pct_start', 0.3),
                anneal_strategy=self.config.get('anneal_strategy', 'cos'),
                div_factor=self.config.get('div_factor', 10.0),
                final_div_factor=self.config.get('final_div_factor', 1000.0)
            )
        elif self.config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif self.config['scheduler'] == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
    
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
        print("\nFinding optimal learning rate...")
        
        # Create temporary optimizer
        temp_optimizer = optim.SGD(
            self.model.parameters(),
            lr=start_lr,
            momentum=0.9,
            weight_decay=self.config['weight_decay']
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
        
        # Save plot
        plt.savefig(self.plot_dir / 'lr_finder.png')
        
        # Reset model
        lr_finder.reset()
        
        print(f"Suggested LR: {suggested_lr:.2e}")
        print(f"Min Loss LR: {min_loss_lr:.2e}")
        
        return suggested_lr
    
    def train(self):
        """Train the model"""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        
        best_acc = 0
        best_model_state = None
        start_epoch = 0
        
        # Check for resume
        if self.config.get('resume', False):
            checkpoint_path = self.checkpoint_dir / 'last_checkpoint.pth'
            if checkpoint_path.exists():
                start_epoch = self.load_checkpoint(str(checkpoint_path))
        
        # Training loop
        for epoch in range(start_epoch, self.config['epochs']):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_acc = train_epoch(
                self.model,
                self.device,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.scheduler if self.config['scheduler'] == 'onecycle' else None,
                epoch,
                self.config.get('accumulation_steps', 1),
                self.config.get('clip_grad_norm', None),
                verbose=True
            )
            
            # Testing phase
            test_loss, test_acc, test_acc_top5, misclassified = test_epoch(
                self.model,
                self.device,
                self.test_loader,
                self.criterion,
                epoch,
                verbose=True,
                calc_top5=True
            )
            
            # Update scheduler (for non-OneCycle schedulers)
            if self.scheduler and self.config['scheduler'] != 'onecycle':
                self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['test_acc_top5'].append(test_acc_top5)
            self.history['lr'].append(current_lr)
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(
                    self.checkpoint_dir / 'best_model.pth',
                    epoch,
                    is_best=True
                )
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth',
                    epoch
                )
            
            # Always save last checkpoint
            self.save_checkpoint(
                self.checkpoint_dir / 'last_checkpoint.pth',
                epoch
            )
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.config['epochs']} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            print(f"  Test Top-5 Acc: {test_acc_top5:.2f}%")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Best Test Acc: {best_acc:.2f}%")
            print(f"{'='*50}\n")
            
            # Plot progress
            if (epoch + 1) % self.config.get('plot_interval', 5) == 0:
                self.plot_progress()
        
        # Final summary
        print("\n" + "="*50)
        print("Training Completed!")
        print(f"Best Test Accuracy: {best_acc:.2f}%")
        print(f"Best Test Top-5 Accuracy: {max(self.history['test_acc_top5']):.2f}%")
        print("="*50)
        
        # Save final plots
        self.plot_progress(save_path=self.plot_dir / 'final_training_curves.png')
        
        # Save training history
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, filepath: Path, epoch: int, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        if is_best:
            print(f"Best model saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> int:
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {filepath}")
        print(f"Resuming from epoch {epoch + 1}")
        return epoch + 1
    
    def save_history(self):
        """Save training history to JSON"""
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to {history_path}")
    
    def plot_progress(self, save_path: Path = None):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Training Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.history['test_loss'], 'r-', label='Test')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Top-1 Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train')
        axes[0, 1].plot(epochs, self.history['test_acc'], 'r-', label='Test')
        axes[0, 1].set_title('Top-1 Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-5 Accuracy
        axes[0, 2].plot(epochs, self.history['test_acc_top5'], 'g-')
        axes[0, 2].set_title('Top-5 Accuracy (Test)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy (%)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(epochs, self.history['lr'], 'orange')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Train vs Test Accuracy Gap
        acc_gap = [train - test for train, test in 
                   zip(self.history['train_acc'], self.history['test_acc'])]
        axes[1, 1].plot(epochs, acc_gap, 'purple')
        axes[1, 1].set_title('Train-Test Accuracy Gap')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gap (%)')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Best Accuracies Text
        axes[1, 2].axis('off')
        best_train_acc = max(self.history['train_acc'])
        best_test_acc = max(self.history['test_acc'])
        best_test_acc_top5 = max(self.history['test_acc_top5'])
        
        summary_text = f"""Training Summary:
        
Best Train Acc: {best_train_acc:.2f}%
Best Test Acc: {best_test_acc:.2f}%
Best Test Top-5 Acc: {best_test_acc_top5:.2f}%

Final Train Acc: {self.history['train_acc'][-1]:.2f}%
Final Test Acc: {self.history['test_acc'][-1]:.2f}%
Final Test Top-5 Acc: {self.history['test_acc_top5'][-1]:.2f}%

Total Epochs: {len(epochs)}
Final LR: {self.history['lr'][-1]:.2e}"""
        
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, 
                       verticalalignment='center', fontfamily='monospace')
        
        plt.suptitle('ResNet-50 CIFAR-100 Training Progress', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


def get_default_config():
    """Get default configuration"""
    return {
        # Data
        'data_dir': './data',
        'batch_size': 128,
        'num_workers': 4,
        'augment_train': True,
        
        # Model
        'model': 'resnet50',
        'num_classes': 100,
        
        # Training
        'epochs': 100,
        'optimizer': 'sgd',
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        
        # Scheduler
        'scheduler': 'onecycle',
        'max_lr': 0.1,
        'pct_start': 0.3,
        'div_factor': 10.0,
        'final_div_factor': 1000.0,
        
        # Other
        'accumulation_steps': 1,
        'clip_grad_norm': None,
        'save_interval': 10,
        'plot_interval': 5,
        
        # Directories
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
        'plot_dir': './plots',
        
        # Resume
        'resume': False,
        
        # LR Finder
        'find_lr': False,
        'lr_finder_iterations': 200
    }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ResNet-50 on CIFAR-100')
    
    # Basic arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='onecycle', 
                       choices=['onecycle', 'cosine', 'step', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--max-lr', type=float, default=0.1, help='Max LR for OneCycle')
    
    # Other arguments
    parser.add_argument('--find-lr', action='store_true', help='Find optimal learning rate')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--no-augment', action='store_true', help='Disable data augmentation')
    
    # Directory arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Log directory')
    parser.add_argument('--plot-dir', type=str, default='./plots',
                       help='Plot directory')
    
    args = parser.parse_args()
    
    # Create configuration
    config = get_default_config()
    
    # Update config with arguments
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['momentum'] = args.momentum
    config['weight_decay'] = args.weight_decay
    config['scheduler'] = args.scheduler if args.scheduler != 'none' else None
    config['max_lr'] = args.max_lr
    config['find_lr'] = args.find_lr
    config['resume'] = args.resume
    config['num_workers'] = args.num_workers
    config['augment_train'] = not args.no_augment
    config['checkpoint_dir'] = args.checkpoint_dir
    config['log_dir'] = args.log_dir
    config['plot_dir'] = args.plot_dir
    
    # Print configuration
    print("\nConfiguration:")
    print("-" * 50)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    print("-" * 50)
    
    # Create trainer
    trainer = ResNet50Trainer(config)
    
    # Find optimal LR if requested
    if config['find_lr']:
        suggested_lr = trainer.find_lr(num_iter=config['lr_finder_iterations'])
        config['learning_rate'] = suggested_lr
        config['max_lr'] = suggested_lr
        trainer.setup_training()  # Reinitialize with new LR
    
    # Train model
    history = trainer.train()
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()