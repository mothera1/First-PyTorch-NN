import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

batch_size = 4

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

"""
for batch_idx, (images, labels) in enumerate(testloader):
    print(f"\nBatch {batch_idx}:")
    
    # Look at first image in this batch
    first_image = images[0].permute(1, 2, 0).numpy()
    
    if first_image.max() <= 1.0:
        first_image = (first_image * 255).astype(np.uint8)
    
    print("RGB values:")
    for row in range(4):
        for col in range(4):
            r, g, b = first_image[row, col]
            print(f"({r:3d},{g:3d},{b:3d})", end=" ")
        print()

"""

def find_similar_images_batch(testloader, method='euclidean', threshold=50, max_images=500):
    all_images = []
    all_labels = []
    
    # Collect images
    for images, labels in testloader:
        batch_size = images.shape[0]
        # Flatten each image: (batch_size, 3, 4, 4) -> (batch_size, 48)
        images_flat = images.view(batch_size, -1)
        
        all_images.append(images_flat)
        all_labels.extend(labels.tolist())
        
        if len(all_labels) >= max_images:
            break
    
    # Concatenate all images
    all_images = torch.cat(all_images, dim=0)[:max_images]
    all_labels = all_labels[:max_images]
    
    print(f"Comparing {len(all_images)} images...")
    
    if method == 'euclidean':
        # Compute pairwise distances efficiently
        distances = torch.cdist(all_images, all_images, p=2)
        
        # Find pairs below threshold (excluding diagonal)
        mask = (distances < threshold) & (distances > 0)
        indices = torch.nonzero(mask)
        
        similar_pairs = []
        for idx in indices:
            i, j = idx[0].item(), idx[1].item()
            if i < j:  # Avoid duplicates
                similar_pairs.append({
                    'indices': (i, j),
                    'distance': distances[i, j].item(),
                    'labels': (all_labels[i], all_labels[j])
                })
    
    return similar_pairs

# Usage
similar_pairs = find_similar_images_batch(testloader, threshold=30)
print(f"Found {len(similar_pairs)} similar pairs")

# Show some examples
for pair in similar_pairs[:10]:
    i, j = pair['indices']
    print(f"Images {i} and {j}: distance={pair['distance']:.2f}, labels=({pair['labels'][0]}, {pair['labels'][1]})")