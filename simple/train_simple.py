import torch
import torch.nn as nn
import torch.optim as optim

from load_simple import *
from simple_NN import Simple


def train(criterion, optimizer, model, epochs = 14):
        
    train_loss = []

    for epoch in range(epochs):
        running_loss = 0.0
        tracking_loss = []
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            tracking_loss.append(loss.item())
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
            
        train_loss.append(sum(tracking_loss)/len(tracking_loss))

    print('Finished Training')
    return train_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Simple().to(device)
model.train()

"""
for i in model.fc1.parameters():
    i.requires_grad = False
"""

criterion = nn.CrossEntropyLoss()
#criterion = nn.MultiMarginLoss()
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

history = train(criterion, optimizer, model)

PATH = './simple_cifar_CE_robust.pth'
#PATH = './simple_cifar_hinge.pth'
torch.save(model.state_dict(), PATH)

plt.plot(history, color='blue', marker='o')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('CE Loss no Freeze')
plt.show()