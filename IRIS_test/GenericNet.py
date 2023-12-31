import torch
import torch.nn as nn

class GenericNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation_fn=nn.ReLU):
        """
        Generic neural network model.

        :param input_size: Number of input features.
        :param hidden_layers: List containing the number of units in each hidden layer.
        :param output_size: Number of output units.
        :param activation_fn: Activation function to be used in hidden layers.
        """
        super(GenericNet, self).__init__()
        layers = []
        last_size = input_size

        # Create hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(activation_fn())
            last_size = hidden_size

        # Add output layer
        layers.append(nn.Linear(last_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


#### Example USAGE #####
#-----   
#model = GenericNet(input_size=20, hidden_layers=[50, 30], output_size=3)
#-----