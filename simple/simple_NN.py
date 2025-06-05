import torch
import torch.nn as nn
import torch.nn.functional as F

from load_simple import *

class Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4*4*3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 4 * 4 * 3)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


#questions for meeting:
#schedule -> I run away at 3pm today and then come back
#general ML questions -> hyperparameters? Validation, train, test? Which loss do we care about?
#why freeze? Methodology behInd freezing? 
#I read online that it is optimal to use a convolutional NN for CIFAR, I am not, is this something that should be explored?
#test loss as batches every 2000ish


#do 4x4 and 3 classes in tandem