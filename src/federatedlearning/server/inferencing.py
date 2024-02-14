import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


def inference(
    cfg: DictConfig, model: nn.Module, test_dataset: Dataset
) -> tuple[float, float]:
    """Returns the test accuracy and loss."""

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device: torch.device = (
        torch.device(f"cuda:{cfg.train.gpu}")
        if cfg.train.gpu is not None and cfg.train.gpu >= 0
        else torch.device("cpu")
    )
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for _, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss
