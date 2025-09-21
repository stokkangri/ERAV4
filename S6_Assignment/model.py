from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # 28>28 | 3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 28 > 28 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 14> 14 | 12
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) #14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(256, 512, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(512, 1024, 3) # 5 > 3 | 32 | 3*3*1024 | 3x3x1024x10 |
        self.conv7 = nn.Conv2d(1024, 10, 3) # 3 > 1 | 34 | > 1x1x10

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)
    
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) # 28x28x1 > 28x28x16 | 3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # 28x28x16 > 28x28x32 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 28x28x32 > 14x14x32 | 10  
        self.conv3 = nn.Conv2d(32, 32, 3) # 14x14x32 > 12x12x32 | 12
        self.conv4 = nn.Conv2d(32, 32, 3) # 12x12x32 > 10x105x32 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 10x10x32 > 5x5x32 | 28
        self.conv5 = nn.Conv2d(32, 16, 3) # 5x5x32 > 3x3x16 | 30
        self.conv6 = nn.Conv2d(16, 10, 3) # 3x3x16 > 1x1x10 | 30
        

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)
    
class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()
        dropout = 0.1

        self.conv1 = nn.Conv2d(1, 8, 3, bias=False)  # 28x28x1 > 26x26x8
        self.bn1 = nn.BatchNorm2d(8)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(8, 16, 3, bias=False)  # 26x26x8 > 24x24x16
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(16, 8, 1, bias=False)  # 24x24x16 > 24x24x8
        self.bn3 = nn.BatchNorm2d(8)
        self.dropout3 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool2d(2, 2)  # 24x24x16 > 12x12x8
        self.conv4 = nn.Conv2d(8, 18, 3, bias=False)  # 12x12x8 > 10x10x16
        self.bn4 = nn.BatchNorm2d(18)
        self.dropout4 = nn.Dropout(dropout)
        self.conv5 = nn.Conv2d(18, 8, 3, bias=False)  # 10x10x16 > 8x8x8
        self.bn5 = nn.BatchNorm2d(8)
        self.dropout5 = nn.Dropout(dropout)
        self.conv6 = nn.Conv2d(8, 16, 3, bias=False)  # 8x8x8 > 6x6x16
        self.bn6 = nn.BatchNorm2d(16)
        self.dropout6 = nn.Dropout(dropout)
        self.conv7 = nn.Conv2d(16, 16, 3, bias=False)  # 6x6x16 > 4x4x16
        self.bn7 = nn.BatchNorm2d(16)
        self.dropout7 = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool2d(1)  # 4x4x16 > 1x1x16
        self.conv8 = nn.Conv2d(16, 10, 1, bias=False)  # 1x1x16 > 1x1x10
        self.bn8 = nn.BatchNorm2d(10)
        self.dropout8 = nn.Dropout(dropout)
        

    def forward(self, x):
        x = self.dropout2(F.relu(self.bn2(self.conv2(self.dropout1(F.relu(self.bn1(self.conv1(x))))))))
        x = self.pool1(self.dropout3(self.bn3(F.relu(self.conv3(x)))))
        x = self.dropout4(self.bn4(F.relu(self.conv4(x))))
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.dropout5(x)
        x = self.bn6(F.relu(self.conv6(x)))
        #x = self.dropout6(x)
        x = self.bn7(F.relu(self.conv7(x)))  # 1x1 convolution to map 14 channels to 10
        #x = self.dropout7(x)
        x = self.gap(x)  # Global average pooling to 1x1x14
        x = self.bn8(F.relu(self.conv8(x)))
        #x = self.dropout8(x)
        #x = self.conv_out(x)  # 1x1 convolution to map 14 channels to 10
        x = x.view(-1, 10)  # Flatten to (batch_size, 10)
        return F.log_softmax(x, dim=-1)