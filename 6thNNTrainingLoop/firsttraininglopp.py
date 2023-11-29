import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import TensorDataset,DataLoader
from GENSynthetic_RegressionData import generate_synthetic_regression_data


features, labels = generate_synthetic_regression_data(100,10)

dataset = TensorDataset(features.float(),labels.float())
dataloader = DataLoader(dataset,batch_size=10,shuffle=True)

model = nn.Sequential(
    nn.Linear(10,5),
    nn.Linear(5,3),
    nn.Linear(3,1)
)

criterion = MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

num_epochs = 10
count =0
for epoch in range(num_epochs):
    for data_batch in dataloader:
        optimizer.zero_grad()

        feature, target =data_batch

        pred = model(feature).squeeze()

        loss = criterion(pred,target)
        print(loss)
        count = count +1
        print(count)
        loss.backward()

        optimizer.step()