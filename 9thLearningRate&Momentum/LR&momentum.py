import torch 
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(nn.Linear(10,20),
                      nn.Linear(20,40),
                      nn.Linear(40,5))

optimizer = optim.SGD(model.parameters(), lr= 0.1, momentum=0.9)