import torch
import torch.nn as nn

from load_simple import *
from simple_NN import Simple


model = Simple()


PATH = './simple_cifar_CE_frozen_fc1.pth'
model.load_state_dict(torch.load(PATH, weights_only=True))
criterion = nn.CrossEntropyLoss()

correct = 0
total = 0
loss_total = []
# since we're not training, we don't need to calculate the gradients for our outputs
model.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
            # calculate outputs by running images through the network
        outputs = model(images)
        loss = criterion(outputs, labels)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss_total.append(loss.item())
        """
        count += 1
        if count == 166:
            count = 0
            tally /= len(testloader)
            loss_total.append(tally)
        """
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    print(f'Average Test Loss: {sum(loss_total)/len(loss_total)}')
