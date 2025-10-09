"""
CIFAR-100 Dataset Loader with data augmentation and transformations
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Optional, Dict
import pickle
from pathlib import Path


class CIFAR100DataLoader:
    """
    CIFAR-100 dataset loader with augmentation
    """
    
    def __init__(
        self,
        data_dir: str = './data',
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        augment_train: bool = True,
        normalize: bool = True
    ):
        """
        Initialize CIFAR-100 data loader
        
        Args:
            data_dir: Directory to download/load CIFAR-100 data
            batch_size: Batch size for training and testing
            num_workers: Number of workers for data loading
            pin_memory: Pin memory for faster GPU transfer
            augment_train: Apply data augmentation to training set
            normalize: Apply normalization
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment_train = augment_train
        self.normalize = normalize
        
        # CIFAR-100 statistics
        self.mean = [0.5071, 0.4867, 0.4408]
        self.std = [0.2675, 0.2565, 0.2761]
        
        # Class names
        self.classes = self._get_classes()
        
        # Create data loaders
        self.train_loader = self._create_train_loader()
        self.test_loader = self._create_test_loader()
    
    def _get_classes(self) -> list:
        """Get CIFAR-100 class names"""
        return [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
            'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
            'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
            'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]
    
    def _get_train_transforms(self) -> transforms.Compose:
        """Get training data transformations"""
        transform_list = []
        
        if self.augment_train:
            transform_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        
        transform_list.append(transforms.ToTensor())
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        # Add Cutout augmentation
        if self.augment_train:
            transform_list.append(Cutout(n_holes=1, length=16))
        
        return transforms.Compose(transform_list)
    
    def _get_test_transforms(self) -> transforms.Compose:
        """Get test data transformations"""
        transform_list = [transforms.ToTensor()]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        return transforms.Compose(transform_list)
    
    def _create_train_loader(self) -> DataLoader:
        """Create training data loader"""
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self._get_train_transforms()
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def _create_test_loader(self) -> DataLoader:
        """Create test data loader"""
        test_dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self._get_test_transforms()
        )
        
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get train and test data loaders"""
        return self.train_loader, self.test_loader
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            'num_classes': 100,
            'num_train_samples': len(self.train_loader.dataset),
            'num_test_samples': len(self.test_loader.dataset),
            'num_train_batches': len(self.train_loader),
            'num_test_batches': len(self.test_loader),
            'batch_size': self.batch_size,
            'image_size': (3, 32, 32),
            'mean': self.mean,
            'std': self.std
        }


class Cutout:
    """
    Cutout augmentation
    Reference: https://arxiv.org/abs/1708.04552
    """
    
    def __init__(self, n_holes: int = 1, length: int = 16):
        """
        Args:
            n_holes: Number of patches to cut out
            length: Length (in pixels) of each square patch
        """
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply cutout augmentation
        
        Args:
            img: Tensor image of size (C, H, W)
        
        Returns:
            Image with cutout applied
        """
        h = img.size(1)
        w = img.size(2)
        
        mask = np.ones((h, w), np.float32)
        
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img


class CIFAR100BinaryLoader:
    """
    Load CIFAR-100 from binary files (for custom downloaded files)
    """
    
    def __init__(
        self,
        train_file: str,
        test_file: str,
        meta_file: Optional[str] = None,
        batch_size: int = 128,
        num_workers: int = 4,
        augment_train: bool = True
    ):
        """
        Initialize binary file loader
        
        Args:
            train_file: Path to training binary file
            test_file: Path to test binary file
            meta_file: Path to meta binary file (optional)
            batch_size: Batch size
            num_workers: Number of workers
            augment_train: Apply augmentation to training data
        """
        self.train_file = train_file
        self.test_file = test_file
        self.meta_file = meta_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Load data from binary files
        self.train_data, self.train_labels = self._load_binary_data(train_file)
        self.test_data, self.test_labels = self._load_binary_data(test_file)
        
        # Load label names if meta file provided
        self.label_names = None
        if meta_file and Path(meta_file).exists():
            self.label_names = self._load_meta(meta_file)
        
        # Create data loaders
        loader = CIFAR100DataLoader(
            batch_size=batch_size,
            num_workers=num_workers,
            augment_train=augment_train
        )
        
        # Use the transform pipelines
        self.train_transform = loader._get_train_transforms()
        self.test_transform = loader._get_test_transforms()
    
    def _unpickle(self, file: str) -> dict:
        """Unpickle a binary file"""
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data
    
    def _load_binary_data(self, file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from binary file"""
        data_dict = self._unpickle(file)
        
        # Extract images and labels
        images = data_dict[b'data']
        if b'fine_labels' in data_dict:
            labels = np.array(data_dict[b'fine_labels'])
        else:
            labels = np.array(data_dict[b'labels'])
        
        # Reshape images from flat array to 3x32x32
        images = images.reshape(-1, 3, 32, 32)
        images = images.transpose(0, 2, 3, 1)  # Convert to HWC format
        
        return images, labels
    
    def _load_meta(self, meta_file: str) -> list:
        """Load label names from meta file"""
        meta_data = self._unpickle(meta_file)
        
        if b'fine_label_names' in meta_data:
            return [name.decode('utf-8') for name in meta_data[b'fine_label_names']]
        elif b'label_names' in meta_data:
            return [name.decode('utf-8') for name in meta_data[b'label_names']]
        
        return None


if __name__ == "__main__":
    # Test the data loader
    print("Testing CIFAR-100 DataLoader...")
    
    # Create data loader
    data_loader = CIFAR100DataLoader(
        batch_size=64,
        num_workers=2,
        augment_train=True
    )
    
    # Get loaders
    train_loader, test_loader = data_loader.get_loaders()
    
    # Print statistics
    stats = data_loader.get_dataset_stats()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test loading a batch
    print("\nTesting batch loading...")
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))
    
    print(f"Train batch - Images: {train_batch[0].shape}, Labels: {train_batch[1].shape}")
    print(f"Test batch - Images: {test_batch[0].shape}, Labels: {test_batch[1].shape}")
    
    print("\nDataLoader test completed successfully!")