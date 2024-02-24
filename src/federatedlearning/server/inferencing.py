import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


def inference(
    cfg: DictConfig, model: nn.Module, test_dataset: Dataset
) -> tuple[float, float]:
    """
    Evaluate a neural network model on a given dataset to calculate the test accuracy and loss.

    Args:
        cfg (DictConfig): The configuration object containing settings such as device and batch size.
        model (nn.Module): Trained PyTorch model that will be evaluated.
        test_dataset (Dataset): The dataset on which the model should be evaluated.

    Returns:
        A tuple containing two elements:
            - test accuracy (float)
            - test loss (float)
    """

    # Prepare the model for evaluation mode (i.e., disable dropout, batch-norm, etc.)
    model.eval()

    # Initialize the loss and statistics
    loss, total, correct = 0.0, 0.0, 0.0

    # Set device for computation based on configuration
    # GPU if specified and available, otherwise CPU
    device: torch.device = (
        torch.device(f"cuda:{cfg.train.gpu}")
        if cfg.train.gpu is not None and cfg.train.gpu >= 0
        else torch.device("cpu")
    )

    # Define the loss criterion and move it to the specified device
    criterion = nn.NLLLoss().to(device)

    # Create DataLoader for the testing set
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Loop through the dataset using DataLoader
    images: torch.Tensor
    labels: torch.Tensor
    for _, (images, labels) in enumerate(testloader):
        # Move images and labels to the device (GPU or CPU)
        images, labels = images.to(device), labels.to(device)

        # Execute the model to get predictions
        outputs = model(images)

        # Calculate the loss between predicted and true values
        batch_loss = criterion(outputs, labels)

        # Accumulate the loss
        loss += batch_loss.item()

        # Get the predicted labels by finding the max value in predictions
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)

        # Count correct predictions and total predictions
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    # Calculate accuracy as the ratio of correct predictions to total predictions
    accuracy: float = correct / total

    # Return the accuracy and the accumulated loss
    return accuracy, loss
