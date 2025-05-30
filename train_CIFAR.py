#create class that trains nn, have the loss function be an input. Save the trained nn.

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


from load_CIFAR import *
from CNN_CIFAR import Net
from Hnn_CIFAR import H

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(criterion, optimizer, model):
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

model = H()

model.to(device)



criterion = nn.CrossEntropyLoss()
#criterion = nn.MultiMarginLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

loss_history = train(criterion, optimizer, model)

#PATH = './cifar_net_cross_entropy.pth'
#PATH = './cifar_net_hinge.pth'\
#PATH = './cifar_h_cross_entropy.pth'
#PATH = './cifar_h_hinge.pth'


#torch.save(model.state_dict(), PATH)

