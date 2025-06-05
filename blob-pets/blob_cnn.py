import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Use 3x3 kernels instead of 5x5 for tiny images
        self.conv1 = nn.Conv2d(3, 6, 3)  # 4x4 -> 2x2
        self.pool = nn.MaxPool2d(2, 2)   # 2x2 -> 1x1
        # Remove second conv layer since we're down to 1x1 after pooling
        self.fc1 = nn.Linear(6 * 1 * 1, 32)  # Smaller hidden layer
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)  # 3 output classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 4x4 -> 2x2 -> 1x1
        x = torch.flatten(x, 1)  # Flatten for FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x