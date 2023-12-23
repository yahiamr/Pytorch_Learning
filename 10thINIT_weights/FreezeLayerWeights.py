import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(64,128),
    nn.Linear(128,256)
)

for name , param in model.named_parameters():
    if name == '0.weight' or name == '0.bias':
        param.requires_grad = False



