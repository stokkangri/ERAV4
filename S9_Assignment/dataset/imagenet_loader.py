"""
Improved ImageNet Dataset Loader with stronger augmentations to prevent overfitting
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import random
from PIL import Image
import json


class ImageNetDataLoader:
    """
    ImageNet dataset loader with enhanced augmentation and subset support
    """
    
    def __init__(
        self,
        data_dir: str = './data/imagenet',
        batch_size: int = 256,
        num_workers: int = 8,
        pin_memory: bool = True,
        augment_train: bool = True,
        normalize: bool = True,
        subset_size: Optional[int] = None,
        subset_classes: Optional[int] = None,
        distributed: bool = False,
        image_size: int = 224,
        crop_pct: float = 0.875,
        augmentation_strength: str = 'strong'  # 'basic', 'medium', 'strong'
    ):
        """
        Initialize ImageNet data loader with enhanced augmentation options
        
        Args:
            data_dir: Directory containing ImageNet data (train/val folders)
            batch_size: Batch size for training and testing
            num_workers: Number of workers for data loading
            pin_memory: Pin memory for faster GPU transfer
            augment_train: Apply data augmentation to training set
            normalize: Apply normalization
            subset_size: If specified, use only this many samples per class
            subset_classes: If specified, use only this many classes
            distributed: Whether to use distributed training
            image_size: Target image size
            crop_pct: Center crop percentage for validation
            augmentation_strength: Level of augmentation ('basic', 'medium', 'strong')
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment_train = augment_train
        self.normalize = normalize
        self.subset_size = subset_size
        self.subset_classes = subset_classes
        self.distributed = distributed
        self.image_size = image_size
        self.crop_pct = crop_pct
        self.augmentation_strength = augmentation_strength
        
        # ImageNet statistics
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Paths
        self.train_dir = self.data_dir / 'train'
        self.val_dir = self.data_dir / 'val'
        
        # Check if data exists
        if not self.train_dir.exists() or not self.val_dir.exists():
            raise ValueError(f"ImageNet data not found in {self.data_dir}. "
                           f"Please download and extract ImageNet dataset.")
        
        # Create data loaders
        self.train_loader = self._create_train_loader()
        self.val_loader = self._create_val_loader()
        
        # Load class names
        self.classes = self._load_class_names()
    
    def _load_class_names(self) -> List[str]:
        """Load ImageNet class names from synset mapping"""
        # Try to load from imagenet_class_index.json if available
        class_index_path = self.data_dir / 'imagenet_class_index.json'
        if class_index_path.exists():
            with open(class_index_path, 'r') as f:
                class_idx = json.load(f)
                return [class_idx[str(i)][1] for i in range(1000)]
        else:
            # Use folder names as class names
            if self.train_dir.exists():
                return sorted([d.name for d in self.train_dir.iterdir() if d.is_dir()])
            return [f"class_{i}" for i in range(1000)]
    
    def _get_train_transforms(self) -> transforms.Compose:
        """Get training data transformations with different augmentation levels"""
        transform_list = []
        
        if self.augment_train:
            if self.augmentation_strength == 'basic':
                # Basic augmentations
                transform_list.extend([
                    transforms.RandomResizedCrop(self.image_size),
                    transforms.RandomHorizontalFlip(),
                ])
            
            elif self.augmentation_strength == 'medium':
                # Medium augmentations
                transform_list.extend([
                    transforms.RandomResizedCrop(self.image_size, scale=(0.08, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                ])
            
            elif self.augmentation_strength == 'strong':
                # Strong augmentations to prevent overfitting
                transform_list.extend([
                    transforms.RandomResizedCrop(self.image_size, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
                    ], p=0.5),
                ])
                
                # Add RandAugment for even stronger augmentation
                transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
        else:
            transform_list.extend([
                transforms.Resize(int(self.image_size / self.crop_pct)),
                transforms.CenterCrop(self.image_size),
            ])
        
        # Add RandomErasing after ToTensor for additional regularization
        transform_list.append(transforms.ToTensor())
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        # Add RandomErasing for strong augmentation
        if self.augment_train and self.augmentation_strength == 'strong':
            transform_list.append(
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            )
        
        return transforms.Compose(transform_list)
    
    def _get_val_transforms(self) -> transforms.Compose:
        """Get validation data transformations"""
        transform_list = [
            transforms.Resize(int(self.image_size / self.crop_pct)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        return transforms.Compose(transform_list)
    
    def _create_subset_dataset(self, dataset: Dataset, is_train: bool = True) -> Dataset:
        """Create a subset of the dataset"""
        if self.subset_size is None and self.subset_classes is None:
            return dataset
        
        # Get all targets
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        else:
            # For ImageFolder
            targets = [s[1] for s in dataset.samples]
        
        # Get unique classes
        unique_classes = list(set(targets))
        
        # Limit number of classes if specified
        if self.subset_classes is not None:
            unique_classes = sorted(unique_classes)[:self.subset_classes]
        
        # Create subset indices
        subset_indices = []
        for class_idx in unique_classes:
            # Get all indices for this class
            class_indices = [i for i, t in enumerate(targets) if t == class_idx]
            
            # Limit samples per class if specified
            if self.subset_size is not None:
                if is_train:
                    # Random sampling for training
                    class_indices = random.sample(class_indices, 
                                                min(self.subset_size, len(class_indices)))
                else:
                    # Take first N samples for validation (consistent)
                    class_indices = class_indices[:self.subset_size]
            
            subset_indices.extend(class_indices)
        
        return Subset(dataset, subset_indices)
    
    def _create_train_loader(self) -> DataLoader:
        """Create training data loader"""
        train_dataset = torchvision.datasets.ImageFolder(
            root=self.train_dir,
            transform=self._get_train_transforms()
        )
        
        # Create subset if needed
        if self.subset_size is not None or self.subset_classes is not None:
            train_dataset = self._create_subset_dataset(train_dataset, is_train=True)
            print(f"Using subset of training data: {len(train_dataset)} samples")
        
        # Create sampler for distributed training
        train_sampler = None
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def _create_val_loader(self) -> DataLoader:
        """Create validation data loader"""
        val_dataset = torchvision.datasets.ImageFolder(
            root=self.val_dir,
            transform=self._get_val_transforms()
        )
        
        # Create subset if needed
        if self.subset_size is not None or self.subset_classes is not None:
            val_dataset = self._create_subset_dataset(val_dataset, is_train=False)
            print(f"Using subset of validation data: {len(val_dataset)} samples")
        
        # Create sampler for distributed training
        val_sampler = None
        if self.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False
            )
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation data loaders"""
        return self.train_loader, self.val_loader
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        num_classes = self.subset_classes if self.subset_classes else 1000
        
        # Calculate actual dataset sizes
        train_size = len(self.train_loader.dataset)
        val_size = len(self.val_loader.dataset)
        
        return {
            'num_classes': num_classes,
            'num_train_samples': train_size,
            'num_val_samples': val_size,
            'num_train_batches': len(self.train_loader),
            'num_val_batches': len(self.val_loader),
            'batch_size': self.batch_size,
            'image_size': (3, self.image_size, self.image_size),
            'mean': self.mean,
            'std': self.std,
            'augmentation_strength': self.augmentation_strength if self.augment_train else 'none'
        }

class TinyImageNetDataLoader(ImageNetDataLoader):
    """
    Tiny ImageNet dataset loader (200 classes, 64x64 images)
    Useful for quick experiments
    """
    
    def __init__(
        self,
        data_dir: str = './data/tiny-imagenet-200',
        batch_size: int = 256,
        num_workers: int = 8,
        **kwargs
    ):
        """
        Initialize Tiny ImageNet data loader
        
        Args:
            data_dir: Directory containing Tiny ImageNet data
            batch_size: Batch size
            num_workers: Number of workers
            **kwargs: Additional arguments passed to parent class
        """
        # Override image size for Tiny ImageNet
        kwargs['image_size'] = 64
        
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )
        
        # Update statistics for Tiny ImageNet
        self.mean = [0.4802, 0.4481, 0.3975]
        self.std = [0.2302, 0.2265, 0.2262]

def create_imagenet_loaders(
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 8,
    subset_percent: Optional[float] = None,
    tiny_imagenet: bool = False,
    augmentation_strength: str = 'strong',
    **kwargs
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Convenience function to create ImageNet data loaders with enhanced augmentation
    
    Args:
        data_dir: Data directory
        batch_size: Batch size
        num_workers: Number of workers
        subset_percent: If specified, use this percentage of data (0-1)
        tiny_imagenet: Use Tiny ImageNet instead of full ImageNet
        augmentation_strength: Level of augmentation ('basic', 'medium', 'strong')
        **kwargs: Additional arguments
    
    Returns:
        Tuple of (train_loader, val_loader, dataset_stats)
    """
    if subset_percent is not None:
        # Calculate subset sizes
        if tiny_imagenet:
            total_train = 100000  # Tiny ImageNet has 100k training images
            total_classes = 200
        else:
            total_train = 1281167  # ImageNet has ~1.28M training images
            total_classes = 1000
        
        samples_per_class = int((total_train / total_classes) * subset_percent)
        kwargs['subset_size'] = max(1, samples_per_class)
        
        # Optionally reduce number of classes for very small subsets
        if subset_percent < 0.1:
            kwargs['subset_classes'] = int(total_classes * subset_percent * 10)
    
    # Create loader with specified augmentation strength
    loader = ImageNetDataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation_strength=augmentation_strength,
        **kwargs
    )
    
    train_loader, val_loader = loader.get_loaders()
    stats = loader.get_dataset_stats()
    
    return train_loader, val_loader, stats

if __name__ == "__main__":
    # Test the data loader
    print("Testing ImageNet DataLoader...")
    
    # Test with a small subset
    try:
        train_loader, val_loader, stats = create_imagenet_loaders(
            data_dir='./data/imagenet',
            batch_size=32,
            num_workers=2,
            subset_percent=0.01  # Use 1% of data for testing
        )
        
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test loading a batch
        print("\nTesting batch loading...")
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        print(f"Train batch - Images: {train_batch[0].shape}, Labels: {train_batch[1].shape}")
        print(f"Val batch - Images: {val_batch[0].shape}, Labels: {val_batch[1].shape}")
        
        print("\nDataLoader test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure ImageNet dataset is downloaded and extracted properly.")

