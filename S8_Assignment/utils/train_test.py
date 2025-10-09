"""
Training and testing functions for model training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict, Optional, Any
from tqdm import tqdm
import time
import numpy as np


def train_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    accumulation_steps: int = 1,
    clip_grad_norm: Optional[float] = None,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Train model for one epoch
    
    Args:
        model: PyTorch model
        device: Device to train on
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch number
        accumulation_steps: Gradient accumulation steps
        clip_grad_norm: Gradient clipping value (optional)
        verbose: Print progress
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    
    train_loss = 0
    correct = 0
    total = 0
    batch_losses = []
    
    # Progress bar
    if verbose:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    else:
        pbar = train_loader
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Scheduler step (if using OneCycleLR or similar)
            if scheduler and hasattr(scheduler, 'step') and \
               scheduler.__class__.__name__ in ['OneCycleLR', 'CosineAnnealingWarmRestarts']:
                scheduler.step()
        
        # Statistics
        train_loss += loss.item() * accumulation_steps
        batch_losses.append(loss.item() * accumulation_steps)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'acc': f'{100. * correct / total:.2f}%',
                'lr': f'{current_lr:.2e}'
            })
    
    # Calculate averages
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def test_epoch(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterion: nn.Module,
    epoch: int = 0,
    verbose: bool = True,
    calc_top5: bool = False
) -> Tuple[float, float, float, List]:
    """
    Test model for one epoch
    
    Args:
        model: PyTorch model
        device: Device to test on
        test_loader: Test data loader
        criterion: Loss function
        epoch: Current epoch number
        verbose: Print progress
        calc_top5: Calculate top-5 accuracy
    
    Returns:
        Tuple of (average_loss, top1_accuracy, top5_accuracy, misclassified_data)
    """
    model.eval()
    
    test_loss = 0
    correct = 0
    correct_top5 = 0
    total = 0
    misclassified = []
    
    # Progress bar
    if verbose:
        pbar = tqdm(test_loader, desc=f'Epoch {epoch} [Test]')
    else:
        pbar = test_loader
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Top-5 accuracy
            if calc_top5:
                _, pred_top5 = outputs.topk(5, 1, largest=True, sorted=True)
                correct_top5 += pred_top5.eq(targets.view(-1, 1).expand_as(pred_top5)).sum().item()
            
            # Store misclassified samples (limit to first 10)
            if len(misclassified) < 10:
                wrong_idx = (predicted != targets).nonzero(as_tuple=True)[0]
                for idx in wrong_idx:
                    if len(misclassified) < 10:
                        misclassified.append({
                            'image': inputs[idx].cpu(),
                            'true_label': targets[idx].cpu().item(),
                            'pred_label': predicted[idx].cpu().item(),
                            'confidence': torch.softmax(outputs[idx], dim=0).max().cpu().item()
                        })
            
            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%',
                    'top5': f'{100. * correct_top5 / total:.2f}%' if calc_top5 else 'N/A'
                })
    
    # Calculate averages
    avg_loss = test_loss / len(test_loader)
    top1_accuracy = 100. * correct / total
    top5_accuracy = 100. * correct_top5 / total if calc_top5 else 0
    
    return avg_loss, top1_accuracy, top5_accuracy, misclassified


class Trainer:
    """
    Trainer class to handle the complete training process
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        epochs: int = 100,
        accumulation_steps: int = 1,
        clip_grad_norm: Optional[float] = None,
        calc_top5: bool = False,
        verbose: bool = True
    ):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            device: Device to train on
            train_loader: Training data loader
            test_loader: Test data loader
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler
            epochs: Number of epochs to train
            accumulation_steps: Gradient accumulation steps
            clip_grad_norm: Gradient clipping value
            calc_top5: Calculate top-5 accuracy
            verbose: Print progress
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.clip_grad_norm = clip_grad_norm
        self.calc_top5 = calc_top5
        self.verbose = verbose
        
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
        
    def train(self, start_epoch: int = 0) -> Dict:
        """
        Train the model
        
        Args:
            start_epoch: Starting epoch (for resuming training)
        
        Returns:
            Training history dictionary
        """
        print(f"Starting training from epoch {start_epoch}")
        print(f"Total epochs: {self.epochs}")
        print(f"Device: {self.device}")
        print("-" * 50)
        
        for epoch in range(start_epoch, self.epochs):
            # Training
            start_time = time.time()
            train_loss, train_acc = train_epoch(
                self.model,
                self.device,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.scheduler,
                epoch,
                self.accumulation_steps,
                self.clip_grad_norm,
                self.verbose
            )
            
            # Testing
            test_loss, test_acc, test_acc_top5, misclassified = test_epoch(
                self.model,
                self.device,
                self.test_loader,
                self.criterion,
                epoch,
                self.verbose,
                self.calc_top5
            )
            
            # Learning rate scheduling (epoch-based schedulers)
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler and hasattr(self.scheduler, 'step') and \
               self.scheduler.__class__.__name__ not in ['OneCycleLR', 'CosineAnnealingWarmRestarts']:
                self.scheduler.step()
            
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
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            if self.calc_top5:
                print(f"  Test Top-5 Acc: {test_acc_top5:.2f}%")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Best Test Acc: {self.best_acc:.2f}%")
            print("-" * 50)
        
        print(f"\nTraining completed!")
        print(f"Best Test Accuracy: {self.best_acc:.2f}%")
        
        return self.history
    
    def save_checkpoint(self, filepath: str, epoch: int) -> None:
        """
        Save training checkpoint
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_acc': self.best_acc,
            'history': self.history
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> int:
        """
        Load training checkpoint
        
        Args:
            filepath: Path to checkpoint file
        
        Returns:
            Epoch to resume from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_acc = checkpoint['best_acc']
        self.history = checkpoint['history']
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {filepath}")
        print(f"Resuming from epoch {epoch + 1}")
        return epoch + 1


if __name__ == "__main__":
    # Test the training functions
    print("Testing training functions...")
    
    # Import necessary modules
    import sys
    sys.path.append('..')
    from models.resnet50 import resnet50
    from dataset.cifar100_loader import CIFAR100DataLoader
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = resnet50(num_classes=100).to(device)
    
    # Create data loaders
    data_loader = CIFAR100DataLoader(batch_size=128)
    train_loader, test_loader = data_loader.get_loaders()
    
    # Create optimizer and criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Test single epoch
    print("\nTesting single epoch training...")
    train_loss, train_acc = train_epoch(
        model, device, train_loader, optimizer, criterion, scheduler, epoch=0
    )
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    
    print("\nTesting single epoch evaluation...")
    test_loss, test_acc, test_acc_top5, _ = test_epoch(
        model, device, test_loader, criterion, epoch=0, calc_top5=True
    )
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Top-5 Acc: {test_acc_top5:.2f}%")
    
    print("\nTraining test completed successfully!")