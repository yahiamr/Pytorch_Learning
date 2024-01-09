import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
model = nn.Sequential(nn.Linear(8, 4),
                      nn.ReLU(),
                      nn.Dropout(p=0.5))

# Define the optimizer with weight decay
# The weight_decay value is a hyperparameter and can be adjusted based on your needs.
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
