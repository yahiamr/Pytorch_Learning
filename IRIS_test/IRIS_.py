import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from GenericNet import GenericNet

def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()  # Set the model to training mode
    total_train_loss = 0

    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    return avg_train_loss

def evaluate_model(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_test_loss = 0
    correct_predictions = 0

    with torch.no_grad():  # No need to track gradients
        for features, labels in test_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = correct_predictions / len(test_loader.dataset)
    return avg_test_loss, test_accuracy

def print_epoch_summary(epoch, epochs, train_loss, test_loss, test_accuracy):
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

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

class IrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create Dataset objects
train_dataset = IrisDataset(X_train, y_train)
test_dataset = IrisDataset(X_test, y_test)


model = GenericNet(input_size=4, hidden_layers=[10, 10], output_size=3)

train_loader = DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    avg_test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    print_epoch_summary(epoch, epochs, avg_train_loss, avg_test_loss, test_accuracy)