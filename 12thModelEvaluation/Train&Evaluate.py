import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_one_epoch(model,dataloader,optimizer,criterion):
    count = 0
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

