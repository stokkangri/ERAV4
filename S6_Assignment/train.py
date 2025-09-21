'''
Assignment:

Your new target is:
99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
Less than or equal to 15 Epochs
Less than 8000 Parameters
Do this using your modular code. Every model that you make must be there in the model.py file as Model_1, Model_2, etc.
Once done, train on SageMaker or EC2
Do this in exactly 3 steps (Exclusind SageMaker/EC2 part)
Each File must have a "target, result, analysis" TEXT block (either at the start or the end)
You must convince why have you decided that your target should be what you have decided it to be, and your analysis MUST be correct. 
Evaluation is highly subjective, and if you target anything out of the order, marks will be deducted. 
Explain your 3 steps using these targets, results, and analysis with links to your GitHub files (Colab files moved to GitHub). 
Keep Receptive field calculations handy for each of your models. 
If your GitHub folder structure or file_names are messy, -100. 
When ready, attempt SESSION 7 -Assignment Solution (Includes a screenshot of logs on SageMaker/EC2, without this step, whole assignment is marked Zero).


'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

#Enhanced data augmentation pipeline for better generalization

train_transform = transforms.Compose([

                        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # Apply Gaussian Blur
                        transforms.RandomRotation(degrees=10), 
                        transforms.RandomAffine(degrees=5, translate=(0.1,0.1), scale=(0.9, 1.1), shear=5),
                        #transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                        #transforms.ColorJitter(brightness=0.2, contrast=0.2),
                        transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
                        transforms.ToTensor(), 
                        transforms.Normalize((0.1307,), (0.3081,))
])


# Clean validation transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# Load the full training dataset (60K samples)
full_train_dataset = datasets.MNIST('../data', train=True, download=True, transform=None)

# Split 60K into 50K train + 10K validation
train_size = 50000
val_size = 10000
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create data loaders
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Step 4: Apply specific transforms to the train and test datasets
# Wrap the datasets with the appropriate transform
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                          shuffle=True, **kwargs)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                        shuffle=False, **kwargs)

print(f"Training set size: {len(train_dataset)}")      # 50,000
print(f"Validation set size: {len(val_dataset)}")   
dataiter = iter(train_loader)
images, labels = next(dataiter)

print(images.shape)
print(labels.shape)

# Let's visualize some of the images
#%matplotlib inline
import matplotlib.pyplot as plt

#plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

#figure = plt.figure()
num_of_images = 60
#for index in range(1, num_of_images + 1):
#    plt.subplot(6, 10, index)
#    plt.axis('off')
#    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')


#print(' - var:', torch.var(exp_data))

from torchsummary import summary
from model import Model_3

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = Model_3().to(device)
summary(model, input_size=(1, 28, 28))

from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, scheduler, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    scheduler.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))

#model =  Net().to(device)
from torch.optim.lr_scheduler import StepLR
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

# Use OneCycleLR scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=15,
    steps_per_epoch=len(train_loader)
)

EPOCHS = 15
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(model, device, train_loader, optimizer, scheduler, epoch)
    test(model, device, val_loader)

t = [t_items.item() for t_items in train_losses]

fig, axs = plt.subplots(2,2,figsize=(15,10))
axs[0, 0].plot(t)
axs[0, 0].set_title("Training Loss")
axs[1, 0].plot(train_acc)
axs[1, 0].set_title("Training Accuracy")
axs[0, 1].plot(test_losses)
axs[0, 1].set_title("Test Loss")
axs[1, 1].plot(test_acc)
axs[1, 1].set_title("Test Accuracy")

