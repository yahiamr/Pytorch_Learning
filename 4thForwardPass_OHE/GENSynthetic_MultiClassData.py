import torch
import numpy as np

def generate_multiclass_synthetic_data(num_samples=100, num_features=10, num_classes=3, seed=0):
    """
    Generates a synthetic dataset for multi-class classification.

    Parameters:
    - num_samples: Number of samples in the dataset.
    - num_features: Number of features per sample.
    - num_classes: Number of classes for classification.
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

    # Calculate the sum for each sample
    sums = np.sum(X, axis=1)

    # Determine the range for each class based on the sum
    class_ranges = np.linspace(sums.min(), sums.max(), num_classes + 1)

    # Assign labels based on the range in which the sum of each sample falls
    y = np.digitize(sums, class_ranges) - 1

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)

    return X_tensor, y_tensor
