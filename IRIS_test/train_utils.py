import torch

def train_one_epoch(model, train_loader, optimizer, criterion):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model to be trained.
        train_loader: DataLoader for the training data.
        optimizer: The optimizer used for training.
        criterion: The loss function.

    Returns:
        float: The average training loss for this epoch.
    """
    model.train()  # Set the model to training mode
    total_train_loss = 0

    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    return avg_train_loss

def evaluate_model(model, test_loader, criterion):
    """
    Evaluates the model on the test dataset.

    Args:
        model: The neural network model to be evaluated.
        test_loader: DataLoader for the test data.
        criterion: The loss function.

    Returns:
        tuple: A tuple containing the average test loss and test accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    total_test_loss = 0
    correct_predictions = 0

    with torch.no_grad():  # No need to track gradients
        for features, labels in test_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = correct_predictions / len(test_loader.dataset)
    return avg_test_loss, test_accuracy

def print_epoch_summary(epoch, epochs, train_loss, test_loss, test_accuracy):
    """
    Prints a summary of the training process at the end of an epoch.

    Args:
        epoch: The current epoch number.
        epochs: The total number of epochs.
        train_loss: The training loss for this epoch.
        test_loss: The testing loss for this epoch.
        test_accuracy: The testing accuracy for this epoch.
    """
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")