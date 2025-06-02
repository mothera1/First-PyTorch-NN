import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from load_simple import *
from simple_NN import Simple

def train(criterion, optimizer, model, trainloader, testloader, device, epochs=10):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # Set up the plot
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, data in enumerate(trainloader):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Print statistics every 200 mini-batches (adjusted for smaller dataset)
            if batch_idx % 200 == 199:
                print(f'[Epoch {epoch + 1}, Batch {batch_idx + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
        
        # Calculate training metrics for the epoch
        avg_train_loss = running_loss / len(trainloader) if running_loss > 0 else sum(history['train_loss'][-1:]) or 0
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(testloader)
        val_acc = 100. * val_correct / val_total
        
        # Store metrics
        # For train_loss, we need to recalculate properly
        model.eval()
        epoch_train_loss = 0.0
        with torch.no_grad():
            for data in trainloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                epoch_train_loss += criterion(outputs, labels).item()
        avg_train_loss = epoch_train_loss / len(trainloader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Update plots in real-time
        ax1.clear()
        ax1.plot(history['train_loss'], label='Training Loss', color='blue', marker='o')
        ax1.plot(history['val_loss'], label='Validation Loss', color='red', marker='s')
        ax1.set_title('Loss Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.clear()
        ax2.plot(history['train_acc'], label='Training Accuracy', color='blue', marker='o')
        ax2.plot(history['val_acc'], label='Validation Accuracy', color='red', marker='s')
        ax2.set_title('Accuracy Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.pause(0.01)  # Brief pause to update plot
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)

    plt.ioff()  # Turn off interactive mode
    plt.show()
    
    print('Finished Training')
    return history

# Usage with your existing code:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Simple().to(device)

# You can use either loss function
#criterion = nn.CrossEntropyLoss()
criterion = nn.MultiMarginLoss()  # Uncomment to use hinge loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train with plotting (make sure trainloader and testloader are defined)
history = train(criterion, optimizer, model, trainloader, testloader, device, epochs=20)

# Save the model
PATH = './simple_cifar_with_plots_hinge.pth'
torch.save(model.state_dict(), PATH)

# Final evaluation (your existing code)
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Final Accuracy: {100 * correct / total:.2f}%')

# Optional: Save the training history for later analysis
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history, f)

# Optional: Create a final summary plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Training Loss', marker='o')
plt.plot(history['val_loss'], label='Validation Loss', marker='s')
plt.title('Final Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Training Accuracy', marker='o')
plt.plot(history['val_acc'], label='Validation Accuracy', marker='s')
plt.title('Final Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")