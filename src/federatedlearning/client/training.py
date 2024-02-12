from argparse import Namespace
from typing import Any, Union

import torch
import torch.nn as nn
from federatedlearning.datasets.common import DatasetSplit
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


class LocalUpdate(object):
    def __init__(
        self, args: Namespace, dataset: Any, idxs: list, logger: SummaryWriter
    ) -> None:
        self.args = args
        self.logger = logger
        (
            self.trainloader,
            self.validloader,
            self.testloader,
        ) = self.train_val_test(dataset, list(idxs))
        self.device = "cuda" if args.gpu else "cpu"
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset: Any, idxs: list) -> tuple[Any, Any, Any]:
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train: list = idxs[: int(0.8 * len(idxs))]
        idxs_val: list = idxs[int(0.8 * len(idxs)) : int(0.9 * len(idxs))]
        idxs_test: list = idxs[int(0.9 * len(idxs)) :]

        trainloader: Any = DataLoader(
            DatasetSplit(dataset, idxs_train),
            batch_size=self.args.local_bs,
            shuffle=True,
        )
        validloader: Any = DataLoader(
            DatasetSplit(dataset, idxs_val),
            batch_size=int(len(idxs_val) / 10),
            shuffle=False,
        )
        testloader: Any = DataLoader(
            DatasetSplit(dataset, idxs_test),
            batch_size=int(len(idxs_test) / 10),
            shuffle=False,
        )
        return trainloader, validloader, testloader

    def update_weights(
        self, model: Any, global_round: int
    ) -> tuple[Any, float]:
        # Set mode to train model
        model.train()
        epoch_loss: list[float] = []

        # Set optimizer for the local updates
        optimizer: Any
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.args.lr, momentum=0.5
            )
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.args.lr, weight_decay=1e-4
            )

        for iter in range(self.args.local_ep):
            batch_loss: list = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs: Any = model(images)
                loss: Any = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        "| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(  # NOQA
                            global_round,
                            iter,
                            batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100.0 * batch_idx / len(self.trainloader),
                            loss.item(),
                        )
                    )
                self.logger.add_scalar("loss", loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model: Any) -> tuple[float, Union[float, Any]]:
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
