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
        (
            self.trainloader,
            self.validloader,
            self.testloader,
        ) = self.train_val_test(dataset, list(idxs))
        self.device: torch.device = (
            torch.device(f"cuda:{cfg.train.gpu}")
            if cfg.train.gpu is not None and cfg.train.gpu >= 0
            else torch.device("cpu")
        )
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(
        self, dataset: Dataset, idxs: list
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns train, validation and test dataloaders for a given dataset
        and client indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train: list = idxs[: int(0.8 * len(idxs))]
        idxs_val: list = idxs[int(0.8 * len(idxs)) : int(0.9 * len(idxs))]
        idxs_test: list = idxs[int(0.9 * len(idxs)) :]

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

    def update_weights(
        self, model: nn.Module, global_round: int
    ) -> tuple[dict[str, torch.Tensor], float]:
        # Set mode to train model
        model.train()
        epoch_loss: list[float] = []

        # Set optimizer for the local updates
        optimizer: torch.optim.Optimizer
        if self.cfg.train.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.cfg.train.lr, momentum=0.5
            )
        elif self.cfg.train.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.cfg.train.lr, weight_decay=1e-4
            )

        for iter in range(self.cfg.train.local_ep):
            batch_loss: list[float] = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.cfg.train.verbose and (batch_idx % 10 == 0):
                    print(
                        "| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(  # NOQA
                            global_round,
                            iter,
                            batch_idx * len(images),
                            len(self.trainloader.dataset),  # type: ignore
                            100.0 * batch_idx / len(self.trainloader),
                            loss.item(),
                        )
                    )
                self.logger.add_scalar("loss", loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model: nn.Module) -> tuple[float, float]:
        """Returns the inference accuracy and loss."""

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for _, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss
