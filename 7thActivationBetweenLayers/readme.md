
---

# Neural Network Comparison: ReLU vs LeakyReLU

This Python script demonstrates the use and comparison of two popular activation functions in neural networks: the Rectified Linear Unit (ReLU) and Leaky Rectified Linear Unit (LeakyReLU). The script uses synthetic binary data for this purpose, showcasing how each activation function influences the model's performance in terms of loss.

## Script Overview

The script is organized into several key sections:

1. **Import Statements**: Imports necessary Python libraries and modules, including PyTorch, scikit-learn, and the custom data generation module.

2. **Data Generation and Preparation**:
   - Utilizes `GENSynthetic_BinaryData.generate_synthetic_data` to create synthetic binary data.
   - Splits the data into training and testing sets using `train_test_split` from scikit-learn for model validation.

3. **Model Definition**:
   - Defines two neural network models using PyTorch's `nn.Sequential` class.
   - The first model (`model_relu`) uses ReLU activation functions, while the second (`model_leakyrelu`) uses LeakyReLU.

4. **Loss Function and Optimizers**:
   - Specifies the Mean Squared Error Loss (MSELoss) as the criterion for evaluating model performance.
   - Defines separate optimizers for each model using Stochastic Gradient Descent (SGD).

5. **Training Loop**:
   - Iteratively trains both models on the training dataset over a predefined number of epochs.
   - Each model's parameters are updated based on the calculated loss.

6. **Evaluation and Output**:
   - Evaluates both models on the test dataset.
   - Prints out the final test loss for each model, providing a direct comparison of their performance.

## Requirements

- Python 3.x
- PyTorch
- scikit-learn
- GENSynthetic_BinaryData (a custom module for data generation)

## Usage

Run the script in a Python environment with the required libraries installed. Ensure `GENSynthetic_BinaryData.py` is available in the same directory or a directory in the Python path.

## Conclusion

This script serves as an educational tool to understand and compare the effects of ReLU and LeakyReLU activation functions in neural networks, particularly in the context of binary classification tasks.

---
