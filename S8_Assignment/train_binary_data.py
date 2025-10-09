"""
Training script for ResNet-50 on CIFAR-100 using binary data files
This script is designed to work with downloaded CIFAR-100 binary files
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime
import json

# Import custom modules
from models.resnet50 import resnet50
from utils.lr_finder import LRFinder
from utils.train_test import train_epoch, test_epoch


class CIFAR100BinaryDataset(Dataset):
    """
    PyTorch Dataset for CIFAR-100 binary files
    """
    
    def __init__(self, data_file, transform=None):
        """
        Args:
            data_file: Path to binary file (train or test)
            transform: Transformations to apply
        """
        self.transform = transform
        self.data, self.labels = self._load_binary_data(data_file)
    
    def _unpickle(self, file):
        """Unpickle a binary file"""
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data
    
    def _load_binary_data(self, file):
        """Load data from binary file"""
        print(f"Loading data from {file}...")
        data_dict = self._unpickle(file)
        
        # Extract images and labels
        images = data_dict[b'data']
        
        # Check for CIFAR-100 specific labels
        if b'fine_labels' in data_dict:
            labels = np.array(data_dict[b'fine_labels'])
            print("Detected CIFAR-100 fine labels")
        else:
            labels = np.array(data_dict[b'labels'])
            print("Using standard labels")
        
        # Reshape images from flat array to 32x32x3
        # CIFAR format: [num_samples, 3072] where 3072 = 32*32*3
        # First 1024 = R, next 1024 = G, last 1024 = B
        num_samples = images.shape[0]
        images = images.reshape(num_samples, 3, 32, 32)
        images = images.transpose(0, 2, 3, 1)  # Convert to HWC format for transforms
        
        print(f"Loaded {num_samples} images")
        return images, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image for transforms
        from PIL import Image
        image = Image.fromarray(image.astype('uint8'))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_label_names(meta_file):
    """Load label names from meta file"""
    with open(meta_file, 'rb') as fo:
        meta_data = pickle.load(fo, encoding='bytes')
    
    if b'fine_label_names' in meta_data:
        fine_labels = [label.decode('utf-8') for label in meta_data[b'fine_label_names']]
        print(f"Loaded {len(fine_labels)} fine label names")
        return fine_labels
    elif b'label_names' in meta_data:
        labels = [label.decode('utf-8') for label in meta_data[b'label_names']]
        print(f"Loaded {len(labels)} label names")
        return labels
    
    return None


def get_transforms(augment=True):
    """Get data transformations"""
    # CIFAR-100 statistics
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    
    if augment:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    return transform


class DualLogger:
    """Logger that writes to both file and console"""
    
    def __init__(self, log_file):
        """Initialize dual logger"""
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')
        
    def write(self, message):
        """Write to both terminal and file"""
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        """Flush both outputs"""
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        """Close the log file"""
        self.log.close()


def setup_logging(log_dir='logs'):
    """Setup logging configuration"""
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(log_dir) / f'training_{timestamp}.log'
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Also redirect stdout to capture all print statements
    dual_logger = DualLogger(str(log_file))
    
    return logging.getLogger(__name__), dual_logger, log_file


def train_with_binary_data(
    train_file,
    test_file,
    meta_file=None,
    epochs=100,
    batch_size=128,
    learning_rate=0.1,
    find_lr_first=False,
    device='cuda',
    resume_from=None,
    checkpoint_interval=10,
    log_dir='logs'
):
    """
    Main training function using binary data files
    
    Args:
        train_file: Path to training binary file
        test_file: Path to test binary file
        meta_file: Path to meta binary file (optional)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        find_lr_first: Whether to find optimal LR first
        device: Device to train on
        resume_from: Path to checkpoint file to resume from
        checkpoint_interval: Save checkpoint every N epochs
        log_dir: Directory to save logs
    """
    # Setup logging
    logger, dual_logger, log_file = setup_logging(log_dir)
    sys.stdout = dual_logger  # Redirect stdout to dual logger
    
    print("="*60)
    print("ResNet-50 CIFAR-100 Training with Binary Data")
    print("="*60)
    print(f"Log file: {log_file}")
    print("="*60)
    
    # Log configuration
    config = {
        'train_file': str(train_file),
        'test_file': str(test_file),
        'meta_file': str(meta_file) if meta_file else None,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'find_lr_first': find_lr_first,
        'device': device,
        'resume_from': str(resume_from) if resume_from else None,
        'checkpoint_interval': checkpoint_interval,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save configuration to log directory
    config_file = Path(log_dir) / f'config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to: {config_file}")
    
    # Load label names if available
    label_names = None
    if meta_file and Path(meta_file).exists():
        label_names = load_label_names(meta_file)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = CIFAR100BinaryDataset(
        train_file,
        transform=get_transforms(augment=True)
    )
    
    test_dataset = CIFAR100BinaryDataset(
        test_file,
        transform=get_transforms(augment=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nInitializing ResNet-50 model...")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = resnet50(num_classes=100).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # Find optimal LR if requested
    if find_lr_first:
        print("\nFinding optimal learning rate...")
        lr_finder = LRFinder(model, optimizer, criterion, device)
        
        try:
            lr_finder.range_test(
                train_loader,
                start_lr=1e-7,
                end_lr=1,
                num_iter=200,
                step_mode='exp',
                diverge_th=5  # Divergence threshold
            )
            
            suggested_lr, min_loss_lr = lr_finder.plot_with_suggestion()
            lr_finder.reset()
            
            print(f"Suggested LR: {suggested_lr:.2e}")
            
            # Only use suggested LR if it's reasonable
            if 1e-5 < suggested_lr < 1:
                learning_rate = suggested_lr
            else:
                print(f"Suggested LR seems unreasonable, using default: {learning_rate:.2e}")
                
        except Exception as e:
            print(f"LR Finder encountered an error: {e}")
            print(f"Using default learning rate: {learning_rate:.2e}")
            # Reset model if LR finder failed
            model = resnet50(num_classes=100).to(device)
        
        # Recreate optimizer with final LR
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
    
    # Learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=1000.0
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'test_acc_top5': [],
        'lr': []
    }
    
    best_acc = 0
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if resume_from and Path(resume_from).exists():
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training history
        if 'history' in checkpoint:
            history = checkpoint['history']
        
        # Load best accuracy
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
        
        # Get starting epoch
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        
        # Adjust scheduler steps
        if scheduler and hasattr(scheduler, '_step_count'):
            # For OneCycleLR, we need to step it to the right position
            for _ in range(start_epoch * len(train_loader)):
                scheduler.step()
        
        print(f"Loaded checkpoint with best accuracy: {best_acc:.2f}%")
    elif resume_from:
        print(f"Warning: Checkpoint file {resume_from} not found, starting fresh training")
    
    # Training loop
    print("\n" + "="*60)
    print(f"Training from epoch {start_epoch} to {epochs}")
    print("="*60)
    
    for epoch in range(start_epoch, epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion,
            scheduler, epoch, verbose=True
        )
        
        # Test
        test_loss, test_acc, test_acc_top5, _ = test_epoch(
            model, device, test_loader, criterion,
            epoch, verbose=True, calc_top5=True
        )
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_acc_top5'].append(test_acc_top5)
        history['lr'].append(current_lr)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_acc': best_acc,
            'history': history,
            'config': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'epochs': epochs
            }
        }
        
        # Always save latest checkpoint
        torch.save(checkpoint, 'latest_checkpoint_binary.pth')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint['best_acc'] = best_acc
            torch.save(checkpoint, 'best_model_binary.pth')
            print(f"  New best model saved!")
        
        # Save periodic checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_name = f'checkpoint_epoch_{epoch+1}_binary.pth'
            torch.save(checkpoint, checkpoint_name)
            print(f"  Checkpoint saved: {checkpoint_name}")
        
        # Print summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%, Top5={test_acc_top5:.2f}%")
        print(f"  LR={current_lr:.2e}, Best={best_acc:.2f}%")
        print("-"*60)
    
    print("\n" + "="*60)
    print("Training Completed!")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print("="*60)
    
    # Save final training history to log directory
    history_file = Path(log_dir) / f'history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(history_file, 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        json_history = {
            key: [float(v) for v in values]
            for key, values in history.items()
        }
        json_history['best_accuracy'] = float(best_acc)
        json_history['total_epochs'] = epochs
        json_history['final_epoch'] = epochs
        json.dump(json_history, f, indent=4)
    print(f"Training history saved to: {history_file}")
    
    # Plot final results
    plot_file = Path(log_dir) / f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plot_training_history(history, save_path=plot_file)
    print(f"Training curves saved to: {plot_file}")
    
    # Create summary report
    summary_file = Path(log_dir) / f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate:.2e}\n")
        f.write(f"Device: {device}\n\n")
        f.write("RESULTS:\n")
        f.write(f"Best Test Accuracy: {best_acc:.2f}%\n")
        f.write(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%\n")
        f.write(f"Final Test Top-5 Accuracy: {history['test_acc_top5'][-1]:.2f}%\n")
        f.write(f"Final Train Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Test Loss: {history['test_loss'][-1]:.4f}\n")
        f.write("\n" + "="*60 + "\n")
    print(f"Summary report saved to: {summary_file}")
    
    # Close the dual logger
    dual_logger.close()
    sys.stdout = dual_logger.terminal  # Restore original stdout
    
    return history, model


def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['test_loss'], 'r-', label='Test')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['test_acc'], 'r-', label='Test')
    axes[0, 1].set_title('Top-1 Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top-5 Accuracy
    axes[1, 0].plot(epochs, history['test_acc_top5'], 'g-')
    axes[1, 0].set_title('Top-5 Accuracy (Test)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(epochs, history['lr'], 'orange')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('ResNet-50 CIFAR-100 Training (Binary Data)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100)
    else:
        plt.savefig('training_history_binary.png', dpi=100)
    
    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Train ResNet-50 on CIFAR-100 using binary data files'
    )
    
    # Required arguments
    parser.add_argument('train_file', type=str,
                       help='Path to training binary file')
    parser.add_argument('test_file', type=str,
                       help='Path to test binary file')
    
    # Optional arguments
    parser.add_argument('--meta', type=str, default=None,
                       help='Path to meta binary file with label names')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')
    parser.add_argument('--find-lr', action='store_true',
                       help='Find optimal learning rate before training')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory to save logs (default: logs)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.train_file).exists():
        print(f"Error: Training file '{args.train_file}' not found")
        sys.exit(1)
    
    if not Path(args.test_file).exists():
        print(f"Error: Test file '{args.test_file}' not found")
        sys.exit(1)
    
    if args.meta and not Path(args.meta).exists():
        print(f"Warning: Meta file '{args.meta}' not found, continuing without label names")
        args.meta = None
    
    # Train model
    history, model = train_with_binary_data(
        train_file=args.train_file,
        test_file=args.test_file,
        meta_file=args.meta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        find_lr_first=args.find_lr,
        device=args.device,
        resume_from=args.resume,
        checkpoint_interval=args.checkpoint_interval,
        log_dir=args.log_dir
    )
    
    print("\nTraining completed successfully!")
    print("Files saved:")
    print("  - best_model_binary.pth (best model)")
    print("  - latest_checkpoint_binary.pth (latest checkpoint)")
    print(f"  - {args.log_dir}/ (logs, history, plots, and summary)")
    print("\nLog files include:")
    print(f"  - {args.log_dir}/training_*.log (complete training log)")
    print(f"  - {args.log_dir}/config_*.json (training configuration)")
    print(f"  - {args.log_dir}/history_*.json (training metrics)")
    print(f"  - {args.log_dir}/summary_*.txt (final summary)")
    print(f"  - {args.log_dir}/training_curves_*.png (loss/accuracy plots)")
    print("\nTo continue training, use:")
    print(f"  python {sys.argv[0]} {args.train_file} {args.test_file} --resume latest_checkpoint_binary.pth --epochs <new_total_epochs>")


if __name__ == "__main__":
    main()