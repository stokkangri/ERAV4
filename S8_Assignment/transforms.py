import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import ToTensor

import numpy as np
from PIL import Image

class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img)  # Convert PIL to NumPy
        augmented = self.transform(image=img)
        img = augmented['image']
        #re_img = torch.tensor(img).permute(2, 0, 1)
        #print ("Image shape", img.shape, re_img.shape)
        return  img # re_img # Convert to Tensor and rearrange dimensions
    
class Transforms():
    """
    This class defines the train/test transforms for our CNN model for MNIST dataset
    """
    def __init__(self) :
        pass

    def __call__(self, img):
        # Convert PIL Image to NumPy array
        img = np.array(img)
        # Apply Albumentations transformations
        augmented = self.transform(image=img)
        # Return the transformed image
        return augmented['image']

    def albumentations_transforms(self, mean, std, test=False):
        if test:
            transform = A.Compose([
                #A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),  # Normalize to mean 0, std 1
                #transforms.ToTensor(),
                #A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                A.Normalize(mean=mean, std=std),
                #transforms.ToTensor(),
                ToTensorV2()
            ])
            return transform
        else:
            transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_height=4, min_width=4, fill_value=0, mask_fill_value=None, p=0.5),
                #A.RandomCrop(width=32, height=32),  # Match CIFAR-10 dimensions
                A.HorizontalFlip(p=0.5),
                #A.RandomBrightnessContrast(p=0.2),
                #A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),  # Normalize to mean 0, std 1
                A.Normalize(mean=mean, std=std),
                #transforms.ToTensor(),
                ToTensorV2(),  # Convert NumPy array to PyTorch tensor
            ])
            return transform
        


    def train_transforms(self, mean, std):
        return AlbumentationsTransform(self.albumentations_transforms(mean, std, test=False))

        #train_transforms = transforms.Compose(
        #    [transforms.RandomHorizontalFlip(),
        #        transforms.RandomRotation((-5, 5)),
        #        transforms.ColorJitter(hue=.02, saturation=.05, brightness=0.005),
        #        transforms.ToTensor(),
        #        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #    ])    

    def test_transforms(self, mean, std):
        test_transforms = AlbumentationsTransform(self.albumentations_transforms(mean, std, test=True))
        
        #test_transforms = A.Compose([
        #        #A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),  # Normalize to mean 0, std 1
        #        #transforms.ToTensor(),
        #        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        #        #transforms.ToTensor(),
        #        ToTensorV2()
        #    ])

        return test_transforms
    