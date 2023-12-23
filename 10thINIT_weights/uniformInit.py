import torch 
import torch.nn as nn

layer0 = nn.Linear(16,32)
layer1 = nn.Linear(32,64)

print("Before uniform init")
print(layer0.weight.min(),layer0.weight.max())
print(layer1.weight.min(),layer1.weight.max())

nn.init.uniform_(layer0.weight)
nn.init.uniform_(layer1.weight)

print("After Uniform init")
print(layer0.weight.min(),layer0.weight.max())
print(layer1.weight.min(),layer1.weight.max())

