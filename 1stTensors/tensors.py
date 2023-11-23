import torch
import numpy as np

# Creating a tensor from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# Creating a tensor from a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Creating tensors retaining other tensors' shapes
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the data type of x_data
print(f"Random tensor: \n {x_rand} \n")

# Tensor Attributes
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Tensor Operations - Addition
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(f"Tensor after manipulation: \n{tensor}")
print(f"Tensor addition: \n{tensor + tensor}")

# Moving Tensors to the GPU
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# Reshaping Tensors
tensor = torch.rand(4, 4)
print(f"Original tensor: \n{tensor}")
reshaped_tensor = tensor.view(16)
print(f"Reshaped tensor: \n{reshaped_tensor}")

# Tensor Slicing and Indexing
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(f"Sliced tensor: \n{tensor[:, 1]}")

# Concatenating Tensors
tensor1 = torch.rand(2, 3)
tensor2 = torch.rand(2, 3)
concatenated = torch.cat([tensor1, tensor2], dim=0)
print(f"Concatenated tensor: \n{concatenated}")

# Computing gradients (for neural networks)
x = torch.rand(3, requires_grad=True)
y = x + 2
z = y * y * 2
z = z.mean()
z.backward()  # computes the gradient
print(f"Gradients: \n{x.grad}")

# Tensor to NumPy and Vice Versa
tensor = torch.ones(5)
numpy_array = tensor.numpy()
print(f"Tensor to NumPy: \n{numpy_array}")

numpy_array = np.array([2, 3])
tensor = torch.from_numpy(numpy_array)
print(f"NumPy to tensor: \n{tensor}")