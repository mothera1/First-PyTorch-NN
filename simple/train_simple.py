import torch
import torch.nn as nn
import torch.optim as optim

from load_simple import *
from simple_NN import Simple


def train(criterion, optimizer, model, epochs = 10):
        
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
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
                print(f'[{epochs + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Simple().to(device)

criterion = nn.CrossEntropyLoss()
#criterion = nn.MultiMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(criterion, optimizer, model)

#PATH = './simple_cifar_CE.pth'
PATH = './simple_cifar_hinge.pth'
torch.save(model.state_dict(), PATH)