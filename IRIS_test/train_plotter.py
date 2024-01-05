# for plotting during training process
# train_plotter.py
import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, title='Training and Validation Loss'):
    """
    Plots the training and validation loss over epochs.

    Args:
        train_losses (list of float): A list of training losses per epoch.
        val_losses (list of float): A list of validation losses per epoch.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(accuracies, title='Validation Accuracy'):
    """
    Plots the validation accuracy over epochs.

    Args:
        accuracies (list of float): A list of accuracies per epoch.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies, label='Validation Accuracy', color='green')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()