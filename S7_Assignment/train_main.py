import torch
import torch.nn as nn
from torchsummary import summary
from model import Net
from dataset import DatasetLoader
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
from func_train_test import train, test
from rf_calculator import summary_with_rf

import multiprocessing


#get_data = DatasetLoader(batch_size=256)

def main():
    get_data = DatasetLoader(batch_size=128)
    train_loader = get_data.train_loader()
    test_loader = get_data.test_loader()

    # Analyze the first batch of 512 images
    #images, labels = next(iter(train_loader))
    #print(f'Shape of the batch - {images.shape}')
    #print(f'Total images in the batch - {len(labels)}')

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    use_cuda = torch.cuda.is_available()
    print(f"Cuda available ###### {use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    model = Net().to(device)
    summary(model, input_size=(3, 32, 32))
    summary_with_rf(model, input_size=(3, 32, 32))


    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

    # Use OneCycleLR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=50,
        steps_per_epoch=len(train_loader)
    )
    criterion = F.nll_loss
    #criterion = nn.CrossEntropyLoss()

    EPOCHS = 35
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, scheduler, epoch)
        test_loss, test_acc, misclass_data = test(model, device, test_loader, criterion)
    
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()