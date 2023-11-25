import torch
import torch.nn as nn
import numpy as np
from SyntheticDataGenerator import generate_synthetic_data

#X_tensor, Y_tensor = generate_synthetic_data(num_samples= 100, num_features=10,seed=100)

# Generating a random input sample to test the models
random_input_sample = torch.randint(100,1000,(1, 10))  # 1 sample with 10 features
random_input_sample = random_input_sample.float()

model = nn.Sequential(
    nn.Linear(10,20),
    nn.Linear(20,40),
    nn.Linear(40,10),
    nn.Linear(10,1),
    nn.Sigmoid()
)

output = model(random_input_sample)
print("Model output with the sigmoid:", output.item())

model_nosigmoid = nn.Sequential(
    nn.Linear(10,20),
    nn.Linear(20,40),
    nn.Linear(40,10),
    nn.Linear(10,1)
)

output_nosigmoid = model_nosigmoid(random_input_sample)
print("Model output without the sigmoid:", output_nosigmoid.item())
