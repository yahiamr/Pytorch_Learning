import torch
import torch.nn as nn

from SyntheticDataGenerator import generate_synthetic_data

X_tensor, Y_tensor = generate_synthetic_data(num_samples= 100, num_features=10)


model = nn.Sequential(
    nn.Linear(10,5),
    nn.Linear(5,1),
    nn.Sigmoid()
)

output = model(X_tensor[0])
print("Model output for the first sample:", output.item())


# Forward pass for the entire dataset
outputs = model(X_tensor).squeeze()  # Remove unnecessary dimensions

# Binarize outputs (using 0.5 as the threshold)
predicted = (outputs > 0.5).float()

# Calculate accuracy
accuracy = (predicted == Y_tensor).float().mean()
print("Accuracy on the synthetic dataset:", accuracy.item())