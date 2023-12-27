import pandas as pd 
import numpy as np

import torch 
from torch.utils.data import TensorDataset
df = pd.read_csv('./11thDataLoaders/animals.csv')


features = df[df.columns[1:-1]]
X = np.array(features)
print(X)

target = df[df.columns[-1]]
y = np.array(target)
print(y)

dataset = TensorDataset(torch.tensor(X).float(),torch.tensor(y).float())