#!/usr/bin/env python3
"""
ResNet50 Training Script for ImageNet
Converted from notebook for command-line execution with logging
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import time
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
import signal
import gc

# Import PyTorch libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_lr_finder import LRFinder
from scipy.ndimage import gaussian_filter1d

# Import ResNet50 model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.model_resnet50 import resnet50


class DualLogger:
    """Logger that writes to both file and console"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


def setup_logging(log_dir, experiment_name):
    """Setup logging to both file and console"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    # Create dual logger
    dual_logger = DualLogger(log_file)
    
    # Redirect stdout and stderr
    sys.stdout = dual_logger
    sys.stderr = dual_logger
    
    print(f"Logging to: {log_file}")
    print(f"Started at: {datetime.now()}")
    print("="*80)
    
    return dual_logger


class Params:
    """Training parameters"""
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.name = args.name
        self.workers = args.workers
        self.lr = args.lr
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_step_size = 30
        self.lr_gamma = 0.1
        self.epochs = args.epochs
        self.resume = args.resume
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir
        self.data_path = args.data_path
        self.val_path = args.val_path
        
    def __repr__(self):
        return str(self.__dict__)
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def train(dataloader, model, loss_fn, optimizer, epoch, writer, device):
    """Training function"""
    size = len(dataloader.dataset)
    model.train()
    start0 = time.time()
    start = time.time()
    
    # Add tracking variables
    total_loss = 0
    correct = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Track metrics
        total_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        batch_size = len(X)
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}], {(current/size * 100):>4f}%")
            step = epoch * size + current
            writer.add_scalar('training loss', loss, step)
            new_start = time.time()
            delta = new_start - start
            start = new_start
            if batch != 0:
                print("Done in ", delta, " seconds")
                remaining_steps = size - current
                speed = 100 * batch_size / delta
                remaining_time = remaining_steps / speed
                print("Remaining time (seconds): ", remaining_time)
        optimizer.zero_grad()
    print("Entire epoch done in ", time.time() - start0, " seconds")
    # Return metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / size
    return avg_loss, accuracy


def test(dataloader, model, loss_fn, epoch, writer, train_dataloader, device, calc_acc5=False):
    """Validation function"""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, correct_top5 = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if calc_acc5:
                _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
                correct_top5 += pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()
    test_loss /= num_batches
    step = epoch * len(train_dataloader.dataset)
    if writer != None:
        writer.add_scalar('test loss', test_loss, step)
    correct /= size
    correct_top5 /= size
    if writer != None:
        writer.add_scalar('test accuracy', 100*correct, step)
        if calc_acc5:
            writer.add_scalar('test accuracy5', 100*correct_top5, step)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if calc_acc5:
        print(f"Test Error: \n Accuracy-5: {(100*correct_top5):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss, correct * 100, correct_top5 * 100  # Return as percentages


def find_lr(model, train_loader, device, end_lr=0.02, num_iter=5000):
    """Learning rate finder"""
    print("\nRunning Learning Rate Finder...")
    
    # Create a copy of the model to avoid affecting the original
    model_copy = resnet50(num_classes=1000).to(device)
    model_copy.load_state_dict(model.state_dict())
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_copy.parameters(), lr=1e-7, weight_decay=1e-4)
    lr_finder = LRFinder(model_copy, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=num_iter, smooth_f=0.1)
    
    # Extract and analyze
    lrs = lr_finder.history["lr"]
    losses = lr_finder.history["loss"]
    smoothed_losses = gaussian_filter1d(losses, sigma=50)
    gradients = np.gradient(smoothed_losses, np.log(lrs))
    min_grad_idx = np.argmin(gradients)
    optimal_lr = lrs[min_grad_idx]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses, label="Original Loss", alpha=0.5)
    plt.plot(lrs, smoothed_losses, label="Smoothed Loss", color="red")
    plt.scatter(optimal_lr, smoothed_losses[min_grad_idx], color="blue", 
                label=f"Steepest Drop LR: {optimal_lr:.2e}", zorder=5)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder with Steepest Drop Marked")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    Path("plots").mkdir(exist_ok=True)
    plt.savefig("plots/lr_finder_curve.png", dpi=100)
    plt.close()
    
    print(f"Learning rate with steepest drop in loss: {optimal_lr:.2e}")
    lr_finder.reset()
    
    # Clean up
    del model_copy
    del lr_finder
    torch.cuda.empty_cache()
    gc.collect()
    
    return optimal_lr


def save_training_curves(history, params, best_val_acc, best_epoch):
    """Save training curves plot"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Top-1')
    axes[0, 1].plot(epochs, history['val_acc_top5'], 'g-', label='Val Top-5')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate plot
    axes[1, 0].plot(epochs, history['lr'], 'orange')
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('LR')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    # Summary text
    axes[1, 1].axis('off')
    summary_text = f"""Training Summary:

Model: ResNet50
Batch Size: {params.batch_size}
Epochs Trained: {len(history['train_loss'])}

Best Val Acc: {best_val_acc:.2f}%
Best Epoch: {best_epoch + 1}

Final Train Acc: {history['train_acc'][-1]:.2f}%
Final Val Acc: {history['val_acc'][-1]:.2f}%
Final Val Top-5: {history['val_acc_top5'][-1]:.2f}%
"""
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                    verticalalignment='center', fontfamily='monospace')
    
    plt.suptitle('ResNet50 Training Progress', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    Path("plots").mkdir(exist_ok=True)
    plt.savefig("plots/resnet50_training_curves.png", dpi=100)
    plt.close()


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\n\nReceived interrupt signal. Saving checkpoint before exit...")
    # The checkpoint will be saved in the main training loop
    global interrupted
    interrupted = True


# Global flag for interrupt handling
interrupted = False


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ResNet50 on ImageNet')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=None, help='learning rate (if None, use LR finder)')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--name', type=str, default='resnet_50_sgd1', help='experiment name')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='log directory')
    parser.add_argument('--data-path', type=str, 
                        default='/home/xpz1/Downloads/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train',
                        help='path to training data')
    parser.add_argument('--val-path', type=str,
                        default='/home/xpz1/Downloads/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val',
                        help='path to validation data')
    parser.add_argument('--skip-lr-finder', action='store_true', help='skip LR finder and use provided LR')
    
    args = parser.parse_args()
    
    # Setup parameters
    params = Params(args)
    
    # Setup logging
    logger = setup_logging(params.log_dir, params.name)
    print("Training Parameters:")
    print(params)
    print("="*80)
    
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Device setup
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    # Data loading
    print("\nSetting up data loaders...")
    
    # Training data loader
    train_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(
        root=params.data_path,
        transform=train_transformation
    )
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        sampler=train_sampler,
        num_workers=params.workers,
        pin_memory=True,
    )
    
    # Validation data loader
    val_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=256, antialias=True),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = torchvision.datasets.ImageFolder(
        root=params.val_path,
        transform=val_transformation
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        num_workers=params.workers,
        shuffle=False,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating ResNet50 model...")
    model = resnet50(num_classes=1000).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Learning rate finder
    if params.lr is None and not args.skip_lr_finder:
        optimal_lr = find_lr(model, train_loader, device)
        params.lr = optimal_lr
    elif params.lr is None:
        params.lr = 0.002  # Default LR
        print(f"Using default learning rate: {params.lr}")
    
    # Setup training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
    
    # OneCycleLR scheduler
    steps_per_epoch = len(train_loader)
    print(f"\nSteps per epoch: {steps_per_epoch}")
    
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.002, total_steps=None, epochs=params.epochs, 
        steps_per_epoch=steps_per_epoch, pct_start=0.3, anneal_strategy='cos', 
        cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, 
        div_factor=10.0, final_div_factor=1000.0, 
        three_phase=False, last_epoch=-1, verbose='deprecated'
    )
    
    # Resume from checkpoint
    start_epoch = 0
    checkpoint_path = os.path.join(params.checkpoint_dir, params.name, "checkpoint.pth")
    
    if params.resume and os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        print(f"Resumed from epoch {start_epoch}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_acc_top5': [],
        'lr': []
    }
    
    # Best model tracking
    best_val_acc = 0
    best_epoch = 0
    
    # Setup tensorboard
    Path(os.path.join(params.checkpoint_dir, params.name)).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter('runs/' + params.name)
    
    # Initial validation
    print("\nRunning initial validation...")
    test(val_loader, model, loss_fn, epoch=0, writer=writer, train_dataloader=train_loader, 
         device=device, calc_acc5=True)
    
    # Training loop
    print("\nStarting training...")
    print("="*80)
    
    try:
        for epoch in range(start_epoch, params.epochs):
            if interrupted:
                print("Training interrupted by user")
                break
                
            print(f"\nEpoch {epoch+1}/{params.epochs}")
            print("-"*40)
            
            # Training
            train_loss, train_acc = train(train_loader, model, loss_fn, optimizer, 
                                          epoch=epoch, writer=writer, device=device)
            
            # Save checkpoint
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "params": params.__dict__
            }
            torch.save(checkpoint, os.path.join(params.checkpoint_dir, params.name, f"model_{epoch}.pth"))
            torch.save(checkpoint, os.path.join(params.checkpoint_dir, params.name, "checkpoint.pth"))
            
            lr_scheduler.step()
            
            # Validation
            val_loss, val_acc, val_acc_top5 = test(val_loader, model, loss_fn, epoch + 1, writer, 
                                                   train_dataloader=train_loader, device=device, calc_acc5=True)
            
            # Update history
            current_lr = optimizer.param_groups[0]['lr']
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_acc_top5'].append(val_acc_top5)
            history['lr'].append(current_lr)
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(checkpoint, os.path.join(params.checkpoint_dir, params.name, "best_model.pth"))
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{params.epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Val Top-5 Acc: {val_acc_top5:.2f}%")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"  Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch+1})")
            print("="*80)
            
            # Save history periodically
            if (epoch + 1) % 5 == 0:
                history_path = os.path.join(params.log_dir, f"{params.name}_history.json")
                with open(history_path, 'w') as f:
                    json.dump(history, f, indent=4)
                save_training_curves(history, params, best_val_acc, best_epoch)
                
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save final results
        print("\nSaving final results...")
        
        # Save history
        history_path = os.path.join(params.log_dir, f"{params.name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to {history_path}")
        
        # Save training configuration
        config_dict = {
            'model': 'ResNet50',
            'batch_size': params.batch_size,
            'initial_lr': params.lr,
            'momentum': params.momentum,
            'weight_decay': params.weight_decay,
            'epochs_trained': len(history['train_loss']),
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch + 1
        }
        
        config_path = os.path.join(params.log_dir, f"{params.name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"Configuration saved to {config_path}")
        
        # Save final training curves
        if len(history['train_loss']) > 0:
            save_training_curves(history, params, best_val_acc, best_epoch)
            print("Training curves saved to plots/resnet50_training_curves.png")
        
        # Close logger
        print(f"\nTraining completed at: {datetime.now()}")
        logger.close()
        
        # Close writer
        writer.close()


if __name__ == "__main__":
    main()