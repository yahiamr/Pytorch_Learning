
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from GenericNet import GenericNet
from train_utils import train_one_epoch, evaluate_model, print_epoch_summary
from iris_dataset import IrisDataset

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create Dataset objects
train_dataset = IrisDataset(X_train, y_train)
test_dataset = IrisDataset(X_test, y_test)

# Model instantiation
model = GenericNet(input_size=4, hidden_layers=[10, 10], output_size=3)

# Data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize lists to track metrics
train_losses = []
test_losses = []
test_accuracies = []

# Training loop
epochs = 100
for epoch in range(epochs):
    avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    avg_test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)

    # Store metrics
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    test_accuracies.append(test_accuracy)

    print_epoch_summary(epoch, epochs, avg_train_loss, avg_test_loss, test_accuracy)

# Plotting
plot_losses(train_losses, test_losses)
plot_accuracy(test_accuracies)