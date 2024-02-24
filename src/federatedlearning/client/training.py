import torch
import torch.nn as nn
from federatedlearning.datasets.common import DatasetSplit
from omegaconf import DictConfig
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset


class LocalUpdate(object):
    def __init__(
        self,
        cfg: DictConfig,
        dataset: Dataset,
        idxs: list,
        logger: SummaryWriter,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        # Create data loaders for training, validation, and testing
        (
            self.trainloader,
            self.validloader,
            self.testloader,
        ) = self.train_val_test(dataset, list(idxs))
        # Determine the computing device (GPU or CPU)
        self.device: torch.device = (
            torch.device(f"cuda:{cfg.train.gpu}")
            if cfg.train.gpu is not None and cfg.train.gpu >= 0
            else torch.device("cpu")
        )
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    # Method to create data loaders
    def train_val_test(
        self, dataset: Dataset, idxs: list
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Splits indices into train, validation, and test sets and returns the corresponding
        DataLoaders.
        """
        # Split indexes for training, validation, and testing (80%, 10%, 10% split)
        idxs_train: list = idxs[: int(0.8 * len(idxs))]
        idxs_val: list = idxs[int(0.8 * len(idxs)) : int(0.9 * len(idxs))]
        idxs_test: list = idxs[int(0.9 * len(idxs)) :]

        # Initialize DataLoaders with batch sizes defined in config
        trainloader: DataLoader = DataLoader(
            DatasetSplit(dataset, idxs_train),
            batch_size=self.cfg.train.local_bs,
            shuffle=True,
        )
        validloader: DataLoader = DataLoader(
            DatasetSplit(dataset, idxs_val),
            batch_size=int(len(idxs_val) / 10),
            shuffle=False,
        )
        testloader: DataLoader = DataLoader(
            DatasetSplit(dataset, idxs_test),
            batch_size=int(len(idxs_test) / 10),
            shuffle=False,
        )
        return trainloader, validloader, testloader

    # Perform local training and update weights
    def update_weights(
        self, model: nn.Module, global_round: int
    ) -> tuple[dict[str, torch.Tensor], float]:
        # Set the model to training mode
        model.train()
        epoch_loss: list[float] = []

        # Initialize an optimizer based on the selected configuration
        optimizer: torch.optim.Optimizer
        if self.cfg.train.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.cfg.train.lr, momentum=0.5
            )
        elif self.cfg.train.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.cfg.train.lr, weight_decay=1e-4
            )

        # Iterate over the local epochs
        for iter in range(self.cfg.train.local_ep):
            batch_loss: list[float] = []
            # Loop over the training data batches
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                # Move batch data to the computing device
                images, labels = images.to(self.device), labels.to(self.device)

                # Reset gradients to zero
                model.zero_grad()
                # Forward pass
                log_probs = model(images)
                # Calculate loss
                loss = self.criterion(log_probs, labels)
                # Backward pass
                loss.backward()
                # Update weights
                optimizer.step()

                # Log the information if verbose mode is on
                if self.cfg.train.verbose and (batch_idx % 10 == 0):
                    print(
                        f"| Global Round : {global_round} | Local Epoch : {iter} | "
                        f"[{batch_idx * len(images)}/{len(self.trainloader.dataset)} "  # type: ignore
                        f"({100.0 * batch_idx / len(self.trainloader):.0f}%)]\t"
                        f"Loss: {loss.item():.6f}"
                    )
                # Add loss to the logger
                self.logger.add_scalar("loss", loss.item())
                # Add the current loss to the batch losses list
                batch_loss.append(loss.item())
            # Compute average loss for the epoch
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # Return the updated state dictionary of the model and the average loss for this round
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # Evaluate the model on the test data
    def inference(self, model: nn.Module) -> tuple[float, float]:
        """Evaluate the model on the test dataset and return accuracy and loss."""
        # Set the model to evaluation mode
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        # Iterate over the test data batches
        for _, (images, labels) in enumerate(self.testloader):
            # Move the batch data to the computing device
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass for inference
            outputs = model(images)
            # Calculate the batch loss
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Get the predictions from the model outputs
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            # Calculate how many predictions were correct
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        # Calculate accuracy
        accuracy = correct / total
        return accuracy, loss
