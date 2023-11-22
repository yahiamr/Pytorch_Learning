import torch
import numpy as np

# tensor from data
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

# from numpy array 
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# tensors retaining other tensors shapes
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data , dtype=torch.float) # overrides the data type of x_data
print(f"Random tensor: \n {x_rand} \n")
