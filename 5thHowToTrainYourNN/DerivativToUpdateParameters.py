import torch.optim as optim
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from GENSynthetic_MultiClassData import generate_multiclass_synthetic_data

X_data, y_data = generate_multiclass_synthetic_data(100,10,0)

model = nn.Sequential(
    nn.Linear(10,30),
    nn.Linear(30,20),
    nn.Linear(20,10),
    nn.Softmax(dim = -1)
)

results = model(X_data)

criterion = CrossEntropyLoss()

loss = criterion (results , y_data)
print(loss)

loss.backward()

optimizer = optim.SGD(model.parameters(), lr=0.01)

optimizer.step()