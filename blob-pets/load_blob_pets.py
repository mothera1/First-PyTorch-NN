import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from collections import defaultdict, Counter

batch_size = 1

transform = transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((4, 4), antialias=True),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
full_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

# CIFAR-10 class indices
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Classes we want to keep
desired_classes = ['dog', 'cat', 'bird']
desired_indices = [cifar10_classes.index(cls) for cls in desired_classes]  # [5, 3, 2]

# Filter training set
train_indices = [i for i, (_, label) in enumerate(full_trainset) if label in desired_indices]
trainset = Subset(full_trainset, train_indices)

# Filter test set
test_indices = [i for i, (_, label) in enumerate(full_testset) if label in desired_indices]
testset = Subset(full_testset, test_indices)

# Create data loaders
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Updated classes list (only the ones we're using)
classes = ('bird', 'cat', 'dog')  # Ordered by original CIFAR-10 indices: 2, 3, 5

# Create mapping from original labels to new labels (0, 1, 2)
label_mapping = {2: 0, 3: 1, 5: 2}  # bird->0, cat->1, dog->2


def map_labels(batch):
    """Helper function to remap labels when iterating through data"""
    inputs, labels = batch
    mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])
    return inputs, mapped_labels

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

