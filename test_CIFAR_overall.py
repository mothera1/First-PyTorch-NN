import torch

from load_CIFAR import *
from CNN_CIFAR import Net
from Hnn_CIFAR import H

#model = Net()
model = H()
#PATH = './cifar_net_cross_entropy.pth'
#PATH = './cifar_net_hinge.pth'
#PATH = './cifar_h_cross_entropy.pth'
PATH = './cifar_h_hinge.pth'

model.load_state_dict(torch.load(PATH, weights_only=True))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')