
---

# GenericNet Module Documentation

## Overview

`GenericNet.py` is a Python module that defines a generic neural network model using PyTorch. It is designed to be flexible and adaptable to various neural network architectures for different types of machine learning problems. The module makes it easy to define a neural network with customizable layers and activation functions.

## Class: GenericNet

### Description

`GenericNet` is a class that extends `nn.Module` from PyTorch. It allows for the creation of a feedforward neural network with a variable number of hidden layers and neurons, along with a choice of activation functions.

### Constructor Parameters

- `input_size` (int): The number of input features.
- `hidden_layers` (list of int): A list containing the number of units in each hidden layer.
- `output_size` (int): The number of output units.
- `activation_fn` (torch.nn.Module, optional): The activation function to be used in hidden layers. The default is `nn.ReLU`.

### Methods

- `__init__(self, input_size, hidden_layers, output_size, activation_fn=nn.ReLU)`: Initializes the neural network.
- `forward(self, x)`: Forward pass through the network. 

### Usage Example

```python
from GenericNet import GenericNet

# Example: Creating a model with 20 input features, two hidden layers with 50 and 30 units, and 3 output units.
model = GenericNet(input_size=20, hidden_layers=[50, 30], output_size=3)
```

## Requirements

- PyTorch: This module is dependent on the PyTorch library. Make sure you have PyTorch installed in your environment. Installation instructions can be found at [PyTorch's official website](https://pytorch.org/get-started/locally/).

## Note

`GenericNet` is designed to be a versatile and reusable component for various machine learning projects. It simplifies the process of neural network creation and can be adapted to different datasets and problem types.

---
