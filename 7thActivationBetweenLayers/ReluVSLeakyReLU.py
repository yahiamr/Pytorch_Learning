import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MSELoss,CrossEntropyLoss
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from GENSynthetic_BinaryData import generate_synthetic_data

features, labels = generate_synthetic_data(100,10)

# Generate data and split into training and testing sets
features, labels = generate_synthetic_data(100, 10)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train.float(), y_train.float())
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

test_dataset = TensorDataset(X_test.float(), y_test.float())
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Define models with ReLU and LeakyReLU
model_relu = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
    nn.ReLU(),
    # Add more layers as needed
    nn.Linear(5, 1)
)

model_leakyrelu = nn.Sequential(
    nn.Linear(10, 20),
    nn.LeakyReLU(0.01),
    nn.Linear(20, 5),
    nn.LeakyReLU(0.01),
    # Add more layers as needed
    nn.Linear(5, 1)
)

#loss and optimizer 
criterion = MSELoss()
optimizer_relu = optim.SGD(model_relu.parameters(), lr=0.01)
optimizer_leakyrelu = optim.SGD(model_leakyrelu.parameters(), lr=0.01)

# Training loop (repeat for both models)
num_epochs = 10
for model, optimizer, name in [(model_relu, optimizer_relu, 'ReLU'), (model_leakyrelu, optimizer_leakyrelu, 'LeakyReLU')]:
    # Training phase
    for epoch in range(num_epochs):
        model.train()
        for feature, target in train_loader:
            optimizer.zero_grad()
            pred = model(feature).squeeze()
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

    # Evaluation phase
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for feature, target in test_loader:
            pred = model(feature).squeeze()
            loss = criterion(pred, target)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f'Final Test Loss for {name} Model: {test_loss:.4f}')
