import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define a convolution layer
        # 1 input channel (e.g., grayscale image), 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        # Define a pooling layer
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Apply the convolution layer
        x = self.conv1(x)
        # Apply activation function (ReLU)
        x = F.relu(x)
        # Apply the pooling layer
        x = self.pool(x)
        return x
        
# Create an instance of the CNN
net = SimpleCNN()

# Create a dummy input tensor (1 image, 1 channel, 32 height, 32 width)
input = torch.randn(1, 1, 32, 32)

# Pass the input through the network
output = net(input)
print(output)