import torch 
import numpy as np


def generate_synthetic_data(num_samples=100, num_features=10, seed=0):
    """
    Generates a synthetic dataset.

    Parameters:
    - num_samples: Number of samples in the dataset.
    - num_features: Number of features per sample.
    - seed: Random seed for reproducibility.

    Returns:
    - X_tensor: A PyTorch tensor of shape (num_samples, num_features) representing the features.
    - y_tensor: A PyTorch tensor of shape (num_samples,) representing the labels.
    """

    # Set the random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate random input data
    X = np.random.randn(num_samples, num_features)

    # Generate labels based on a simple rule
    y = (np.sum(X, axis=1) > 0).astype(int)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor