# Author: Dr. Roi Yehoshua
# March 2023

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from torch import nn
import time

torch.manual_seed(0)

# Hyperparameters
batch_size = 32
learning_rate = 1e-3
validation_split = 0.1
n_epochs = 10

# Load the CIFAR-10 training and test sets
training_set = datasets.CIFAR10(root='data', train=True, download=True, transform=ToTensor())
test_set = datasets.CIFAR10(root='data', train=False, download=True, transform=ToTensor())

# Create a validation set
training_size = int((1 - validation_split) * len(training_set))
validation_size = int(validation_split * len(training_set))
training_set, validation_set = random_split(training_set, [training_size, validation_size])

# Define the data loaders
train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Define the model
model = nn.Sequential(   
    # A convolution layer with 32 filters of size 3x3
    nn.Conv2d(3, 32, 3),             
    nn.ReLU(),
    nn.MaxPool2d(2),                 
    nn.Dropout(0.25),

    # A convolutional layer with 64 filters of size 3x3
    nn.Conv2d(32, 64, 3),            
    nn.ReLU(),
    nn.MaxPool2d(2),                 
    nn.Dropout(0.25),

    # A fully-connected layer with 512 neurons
    nn.Flatten(),
    nn.Linear(64 * 6 * 6, 512),      
    nn.ReLU(),
    nn.Dropout(0.5),

    # The final output layer with 10 neurons
    nn.Linear(512, 10)
)

print(model)

# The training loop
def train_loop(model, data_loader, loss_fn, optimizer):  
    size = len(data_loader.dataset)  

    for batch, (X, y) in enumerate(data_loader):       
        # Compute prediction and loss
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
        
        # Print the loss every 100 mini-batches
        if (batch + 1) % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
            
def evaluate_model(model, data_loader):
    size = len(data_loader.dataset)
    correct = 0

    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            y_pred = output.argmax(1)  
            correct += (y_pred == y).sum().item()
    return 100 * correct / size

def train_model(model, loss_fn, optimizer):
    train_start_time = time.time()

    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}\n-------------------------------')
        
        epoch_start_time = time.time()
        model.train() # Ensure the dropout layers are in training mode
        train_loop(model, train_loader, loss_fn, optimizer)        
        model.eval() # Set dropout layers to evaluation mode
        val_accuracy = evaluate_model(model, validation_loader)
        epoch_elapsed_time = time.time() - epoch_start_time      
        
        print(f'Epoch {epoch + 1} completed in {epoch_elapsed_time:.3f}s, ' 
              f'val_accuracy: {val_accuracy:.3f}%\n')
    
    train_elapsed_time = time.time() - train_start_time
    print(f'Training completed in {train_elapsed_time:.3f}s')

    model.eval()
    train_accuracy = evaluate_model(model, train_loader)
    print(f'Accuracy on training set: {train_accuracy:.3f}%')
    test_accuracy = evaluate_model(model, test_loader)
    print(f'Accuracy on test set: {test_accuracy:.3f}%')

# Define a loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    train_model(model, loss_fn, optimizer)