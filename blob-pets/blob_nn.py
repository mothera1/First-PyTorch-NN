import torch
import torch.nn as nn
import torch.nn.functional as F

class Blob(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4*4*3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(0.2)
    

    
    def forward(self, x):
        x = x.view(-1, 4 * 4 * 3)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


#take sample from test set, see if the RGB values for a given item appear multiple times
#do this to see if we have created ambiguity
#look for the frequency of the mistake based on the ambiguous examples