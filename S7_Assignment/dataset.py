import torch
from torchvision import datasets, transforms

from transforms import Transforms

from torch.utils.data import DataLoader
import numpy as np

class DatasetLoader:
    def __init__(self, batch_size=128, num_workers=4):
        self.kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
        }
        self.transforms = Transforms()

    def train_loader(self):
        mean, std = self.compute_mean_std()
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=self.transforms.train_transforms(mean, std)
        )
        return DataLoader(train_dataset, **self.kwargs)

        

    def test_loader(self):
        """
        Test loader for the dataset
        """
        mean, std = self.compute_mean_std()
        test_dataset = datasets.CIFAR10('./data', train=False, download=True,
                                         transform=self.transforms.test_transforms(mean, std))
        
        return DataLoader(test_dataset, **self.kwargs)

    def compute_mean_std(self):
        """
        Compute dataset mean and std (per channel) for CIFAR-10 train split.
        """
        # Use only ToTensor here (no normalization)
        temp_transform = transforms.ToTensor()

        dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=temp_transform
        )
        loader = DataLoader(dataset, batch_size=self.kwargs['batch_size'],
                            shuffle=False, num_workers=self.kwargs['num_workers'])

        mean = 0.0
        std = 0.0
        total_images = 0

        for images, _ in loader:
            # images: [B, C, H, W]
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)  # [B, C, H*W]
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images += batch_samples

        mean /= total_images
        std /= total_images

        return mean.tolist(), std.tolist()