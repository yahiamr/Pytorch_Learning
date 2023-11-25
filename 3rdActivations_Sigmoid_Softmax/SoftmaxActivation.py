import torch
import torch.nn as nn

# Generating a random input sample to test the models
random_input_sample = torch.randint(100,1000,(1, 10))  # 1 sample with 10 features
random_input_sample = random_input_sample.float()

model = nn.Sequential(
    nn.Linear(10,20),
    nn.Linear(20,40),
    nn.Linear(40,10),
    nn.Linear(10,5),
    nn.Softmax(dim= -1)
)

output = model(random_input_sample)
print("Model output with the Softmax:", output)

model_noSoftmax = nn.Sequential(
    nn.Linear(10,20),
    nn.Linear(20,40),
    nn.Linear(40,10),
    nn.Linear(10,5)
)

output_noSoftmax = model_noSoftmax(random_input_sample)
print("Model output without the sigmoid:", output_noSoftmax)