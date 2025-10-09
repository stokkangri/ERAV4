from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torchsummary import summary
from model import Net
from dataset import DatasetLoader
from tqdm import tqdm

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, device, train_loader, optimizer, criterion, scheduler, epoch):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    correct = 0
    processed = 0
    pbar = tqdm(train_loader)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        loss = criterion(output, target)
        train_loss.update(loss.item(), data.size(0))
        
        # Backward pass
        loss.backward()
        optimizer.step()

        scheduler.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        train_acc.update(100. * correct / data.size(0), data.size(0))
        
        
        # Display current learning rate # Update progress bar
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_description(f'Epoch={epoch} LR={current_lr:.5f} Avg Training loss={train_loss.avg:.4f} Batch Loss={loss.item():.4f} Batch_id={batch_idx}, Acc={train_acc.avg:.2f}')
    
    return train_loss.avg, train_acc.avg

def test(model, device, test_loader, criterion, debug=False):
    model.eval()
    test_loss = AverageMeter()
    test_acc = AverageMeter()
    
    # Lists to store misclassified examples
    misclassified_images = []
    misclassified_preds = []
    misclassified_targets = []
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            test_loss.update(loss.item(), data.size(0))
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            test_acc.update(100. * correct / data.size(0), data.size(0))
            
            # Collect misclassified examples if in debug mode
            if debug:
                misclassified_mask = ~pred.eq(target.view_as(pred)).squeeze()
                if misclassified_mask.any():
                    misclassified_images.append(data[misclassified_mask])
                    misclassified_preds.append(pred[misclassified_mask].squeeze())
                    misclassified_targets.append(target[misclassified_mask])
    
    print(f'\nTest set: Average loss: {test_loss.avg:.4f}, Accuracy: {test_acc.avg:.2f}%\n')
    
    # Return misclassified examples if any were found
    if debug and misclassified_images:
        misclassified_data = (
            torch.cat(misclassified_images),
            torch.cat(misclassified_preds),
            torch.cat(misclassified_targets)
        )
        return test_loss.avg, test_acc.avg, misclassified_data
    
    return test_loss.avg, test_acc.avg, None
