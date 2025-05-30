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

dataiter = iter(trainloader)
images, labels = next(dataiter)


imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

outputs = model(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))