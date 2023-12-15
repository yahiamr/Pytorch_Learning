
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # First convolutional layer: 1 input channel, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        # Second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected layers
        self.fc1 = nn.Linear(16*4*4, 120)  # 16*4*4 is the dimension of the flattened feature map
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Apply first convolution, followed by ReLU activation and max pooling
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv1(x)), 2)
        # Apply second convolution, followed by ReLU activation and max pooling
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv2(x)), 2)
        # Flatten the feature map
        x = torch.flatten(x, 1)
        # Apply fully connected layers with ReLU activations
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = LeNet5()

