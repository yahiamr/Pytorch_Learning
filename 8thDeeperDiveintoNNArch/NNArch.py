import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(10,20),
                      nn.Linear(20,20),
                      nn.Linear(20,40),
                      nn.Linear(40,10))

total = 0

for parameter in model.parameters():
    total+= parameter.numel()

print(total)