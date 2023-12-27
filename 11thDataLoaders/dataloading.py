import pandas as pd 
import numpy as np

import torch 
from torch.utils.data import TensorDataset , DataLoader
df = pd.read_csv('./11thDataLoaders/animals.csv')


features = df[df.columns[1:-1]]
X = np.array(features)
print(X)

target = df[df.columns[-1]]
y = np.array(target)
print(y)

dataset = TensorDataset(torch.tensor(X).float(),torch.tensor(y).float())

sample = dataset[0]
X_sample, y_sample = sample
print("X Sample",X_sample)
print("y sample",y_sample)

batch_size = 2
shuffle = True

dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=shuffle)

for batch_X, batch_y in dataloader:
    print("X Batch", batch_X)
    print("y batch", batch_y)