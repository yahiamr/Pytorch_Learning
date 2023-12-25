---

# PyTorch Sequential Model Example

This README file explains a Python script that demonstrates the use of PyTorch, a popular deep learning library. The script covers creating a simple neural network model, modifying its parameters, and performing a basic training step.

### Model Definition

The script begins with importing the necessary modules from PyTorch:

```python
import torch
import torch.nn as nn
```

A neural network model is defined using `nn.Sequential`. This model is a simple feedforward network with two linear layers:

```python
model = nn.Sequential(
    nn.Linear(64, 128),  # First linear layer with 64 inputs and 128 outputs
    nn.Linear(128, 256)  # Second linear layer with 128 inputs and 256 outputs
)
```

### Parameter Modification

The script then iterates through the model's parameters. It selectively freezes the parameters (weights and biases) of the first layer (`nn.Linear(64, 128)`). Freezing parameters means they will not be updated during training.

```python
for name, param in model.named_parameters():
    if name == '0.weight' or name == '0.bias':
        param.requires_grad = False
```

### Saving Initial Parameters

The initial state of all parameters is saved for later comparison:

```python
initial_params = {name: param.clone() for name, param in model.named_parameters()}
```

### Training Preparation

Dummy input and target tensors are created for the demonstration:

```python
input = torch.randn(1, 64)   # Random input tensor
target = torch.randn(1, 256) # Random target tensor
criterion = nn.MSELoss()     # Mean Squared Error Loss function
```

An optimizer is defined, but it only includes parameters that require gradient computation (i.e., the unfrozen parameters):

```python
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
```

### Training Step

A single training step is performed:

```python
optimizer.zero_grad()   # Clearing previous gradients
output = model(input)   # Forward pass
loss = criterion(output, target) # Calculating loss
loss.backward()         # Backpropagation
optimizer.step()        # Updating parameters
```

### Checking Parameter Updates

Finally, the script checks and prints whether each parameter has changed after the training step. It differentiates between the frozen and unfrozen parameters:

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        changed = not torch.equal(initial_params[name], param)
        print(f'Parameter {name} changed: {changed}')
    else:
        print(f'Parameter {name} is frozen.')
```

---