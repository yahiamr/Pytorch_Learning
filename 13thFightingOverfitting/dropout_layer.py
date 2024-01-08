import torch.nn as nn
import torch

model = nn.Sequential(nn.Linear(8,4),
                      nn.ReLU(),
                      nn.Dropout(p=0.5))

features = torch.randn((1,8))