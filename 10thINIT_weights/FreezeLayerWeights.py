# PyTorch and its neural network module are imported for model construction and training.
import torch
import torch.nn as nn

# Define a sequential model with two linear layers.
model = nn.Sequential(
    nn.Linear(64, 128),  # First linear layer with 64 inputs and 128 outputs.
    nn.Linear(128, 256)  # Second linear layer with 128 inputs and 256 outputs.
)

# Iterate over the model's parameters to freeze the first layer's parameters.
for name, param in model.named_parameters():
    if name == '0.weight' or name == '0.bias':  # Checks if the parameter belongs to the first layer.
        param.requires_grad = False  # Freezes the parameter by not calculating gradients.

# Saves the initial state of the model's parameters for later comparison.
initial_params = {name: param.clone() for name, param in model.named_parameters()}

# Generates a dummy input vector and a target vector for demonstration purposes.
input = torch.randn(1, 64)  # Random input vector of size 64.
target = torch.randn(1, 256)  # Random target vector of size 256.
criterion = nn.MSELoss()  # Defines the loss function as Mean Squared Error (MSE).

# Defines the optimizer, only optimizing parameters that require gradients.
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

# Training step begins:
optimizer.zero_grad()  # Resets gradients of all optimized parameters to zero.
output = model(input)  # Passes the input through the model to get an output.
loss = criterion(output, target)  # Calculates loss between the model output and the target.
loss.backward()  # Performs backpropagation to calculate gradients.
optimizer.step()  # Updates the parameters based on calculated gradients.

# Checks if the parameters were updated during the training step.
for name, param in model.named_parameters():
    if param.requires_grad:
        # Checks if the current parameter has changed from its initial value.
        changed = not torch.equal(initial_params[name], param)
        print(f'Parameter {name} changed: {changed}')  # Prints if the parameter has changed.
    else:
        # If the parameter was frozen, it reports that the parameter is frozen.
        print(f'Parameter {name} is frozen.')