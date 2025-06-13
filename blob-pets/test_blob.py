import torch
import torch.nn as nn

from load_blob_pets import *
from blob_nn import Blob

model = Blob()

PATH = './blob_CE.pth'

model.load_state_dict(torch.load(PATH, weights_only=True))

correct = 0
total = 0
predictions = []

with torch.no_grad():
    for data in testloader:
        images, labels = map_labels(data)

        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        predictions.append(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

print(len(predictions))

"""

Ideas for putting predictions on graph

1. Assign each aggregate point on the simplex a ID
2. Do predictions as normal
3. For each image have the dict/list situation where instead of the true label we have the predicted label
4. plot + compare with cross entropy and hinge
5. put class boundaries on the graph

"""