import torch
import torch.nn as nn
import torch.nn.functional as F


class limited(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4*4*3, 3)

    def forward(self, x):
        x = x.view(-1, 4 * 4 * 3)  # Flatten
        x = F.relu(self.fc1(x))
        return x