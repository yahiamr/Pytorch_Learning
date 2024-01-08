import torch.nn as nn
import torch

model = nn.Sequential(nn.Linear(8,6),
                      nn.ReLU(),
                      nn.Dropout(p=0.5))

features = torch.randn((1,8))

output = model(features)
print(output)