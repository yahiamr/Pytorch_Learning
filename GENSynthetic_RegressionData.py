import torch
import numpy as np

def generate_synthetic_regression_data(num_samples=100, num_features=10, noise_level=0.1, seed=0):
    """
    Generates a synthetic dataset for regression.

    Parameters:
    - num_samples: Number of samples in the dataset.
    - num_features: Number of features per sample.
    - noise_level: Standard deviation of Gaussian noise added to the targets.
    - seed: Random seed for reproducibility.

    Returns:
    - X_tensor: A PyTorch tensor of shape (num_samples, num_features) representing the features.
    - y_tensor: A PyTorch tensor of shape (num_samples,) representing the target values.
    """

    # Set the random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate random input data
    X = np.random.randn(num_samples, num_features)

    # Generate target values: here we use a simple linear combination of the input features
    weights = np.random.randn(num_features)
    y = np.dot(X, weights)

    # Add Gaussian noise to the target values
    y += np.random.normal(0, noise_level, num_samples)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor
