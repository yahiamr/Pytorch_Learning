---

# Iris Classification with Neural Networks

## Introduction

This project presents a neural network-based approach for classifying the Iris dataset using PyTorch. The implementation demonstrates building a custom neural network (`GenericNet`), training the model, and evaluating its performance with visual insights. Key features include data preprocessing, model training, performance evaluation, and visualizations of the training process.

## Installation

Before running the project, ensure you have the following dependencies installed:

1. PyTorch - For building and training the neural network model.
2. Scikit-Learn - For data loading and preprocessing.
3. Matplotlib - For generating training and validation metric plots.

To install these dependencies, run:

```bash
pip install torch
pip install scikit-learn
pip install matplotlib
```

## Usage

To run the classification model on the Iris dataset, execute the script from the command line:

```bash
python iris_classification.py
```

## Dataset

The Iris dataset is a classic dataset in the machine learning community, consisting of 150 samples of iris flowers with 4 features: sepal length, sepal width, petal length, and petal width. It includes three species of Iris (setosa, virginica, and versicolor). Our model aims to classify these species based on the given features.

## Model Architecture

The `GenericNet` class allows for creating a neural network with customizable layers and activation functions. In this project, the network is configured with:

- Input size matching the feature count of the Iris dataset.
- Two hidden layers.
- An output layer with units corresponding to the number of Iris species.

## Training Process

The model is trained on the Iris dataset over multiple epochs, and we track metrics like training loss, validation loss, and validation accuracy. These metrics are visualized to understand the model's learning progress.

## Results

### Training and Validation Loss

![Training and Validation Loss](IRIS_test/train_val_loss_plot.png)

*This plot illustrates the training and validation loss over each epoch, showcasing the model's learning curve.*

### Validation Accuracy

![Validation Accuracy](IRIS_test/val_accuracy_plot.png)

*This plot displays the validation accuracy per epoch, highlighting the model's performance in classifying the Iris species.*

---

**Note:** Ensure that the paths to the images in the README (`IRIS_test/train_val_loss_plot.png` and `IRIS_test/val_accuracy_plot.png`) correctly point to where your images are stored. If you're hosting this project on a platform like GitHub, you might want to upload these images to your repository and use relative paths to link them in the README.