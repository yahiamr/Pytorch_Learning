import torch.nn as nn
import torch

model = nn.Sequential(nn.Linear(8,6),
                      nn.ReLU(),
                      nn.Dropout(p=0.5))

features = torch.randn((1,8))

output = model(features)
print(output)

# Pass the input through the model multiple times with different seeds
for i in range(5):
    torch.manual_seed(torch.initial_seed() + i)  # Change the seed
    output = model(features)
    print(f"Iteration {i+1}: {output}")