import torch
import torch.nn as nn 




input_tensor = torch.tensor([[0.3471,0.4547,-0.2356]])

linear_layer = nn.Linear(in_features= 3 , out_features= 2)

output = linear_layer(input_tensor)

print(output)

model = nn.Sequential(
    nn.Linear(3,10),
    nn.Linear(10,25),
    nn.Linear(25,10),
    nn.Linear(10,3)
)
print(model(input_tensor))