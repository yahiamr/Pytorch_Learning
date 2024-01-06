
---

# Train Plotter Module Documentation

## Overview

`train_plotter.py` is a Python module designed to assist in visualizing the training process of machine learning models. It provides functionalities to plot training and validation losses, as well as validation accuracy over epochs. This module uses `matplotlib` for generating plots.

## Functions

### 1. `plot_losses(train_losses, val_losses, title='Training and Validation Loss')`

This function plots the training and validation loss per epoch.

**Arguments:**
- `train_losses` (list of float): A list of training loss values, one for each epoch.
- `val_losses` (list of float): A list of validation loss values, one for each epoch.
- `title` (str, optional): The title of the plot. Default is 'Training and Validation Loss'.

### 2. `plot_accuracy(accuracies, title='Validation Accuracy')`

This function plots the validation accuracy per epoch.

**Arguments:**
- `accuracies` (list of float): A list of accuracy values, one for each epoch.
- `title` (str, optional): The title of the plot. Default is 'Validation Accuracy'.

## Usage

1. **Import the module in your training script:**
   ```python
   from train_plotter import plot_losses, plot_accuracy
   ```

2. **Track the training and validation metrics during your training loop.** For example:
   ```python
   train_losses = []
   val_losses = []
   accuracies = []
   for epoch in range(epochs):
       # ... training and validation logic ...
       train_losses.append(avg_train_loss)
       val_losses.append(avg_val_loss)
       accuracies.append(accuracy)
   ```

3. **Plot the metrics after the training loop:**
   ```python
   plot_losses(train_losses, val_losses)
   plot_accuracy(accuracies)
   ```

## Requirements

- matplotlib: This module uses `matplotlib` for plotting. Ensure you have `matplotlib` installed in your environment:
  ```bash
  pip install matplotlib
  ```

## Note

This module is intended for use with machine learning models, particularly those using the PyTorch framework. It is part of a larger suite of utilities designed to streamline and visualize the model training and evaluation process.

---
