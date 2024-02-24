from typing import Any

import torch
from federatedlearning.datasets.sampling import (
    cifar_iid,
    cifar_noniid,
    mnist_iid,
    mnist_noniid,
    mnist_noniid_unequal,
)
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class DatasetSplit(Dataset):
    def __init__(self, dataset: Dataset, idxs: list) -> None:
        self.dataset = dataset
        self.idxs: list[int] = [int(i) for i in idxs]

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, item: Any) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image).clone().detach(), torch.tensor(
            label
        ).clone().detach()


def get_dataset(cfg: DictConfig) -> tuple[Any, Any, dict]:
    """Returns train and test datasets and a client group which is a dict where
    the keys are the client index and the values are the corresponding data for
    each of those clients.
    """

    if cfg.train.dataset == "cifar":
        data_dir: str = "/workspace/data/cifar/"
        apply_transform: transforms.Compose = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_dataset: Any = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=apply_transform
        )

        test_dataset: Any = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=apply_transform
        )

        # sample training data amongst clients
        if cfg.federatedlearning.iid:
            # Sample IID client data from Mnist
            client_groups: dict = cifar_iid(
                train_dataset, cfg.federatedlearning.num_clients
            )
        else:
            # Sample Non-IID client data from Mnist
            if cfg.federatedlearning.unequal:
                # Chose uneuqal splits for every client
                raise NotImplementedError()
            else:
                # Chose euqal splits for every client
                client_groups = cifar_noniid(
                    train_dataset, cfg.federatedlearning.num_clients
                )

    elif cfg.train.dataset == "mnist" or "fmnist":
        if cfg.train.dataset == "mnist":
            data_dir = "/workspace/data/mnist/"
        else:
            data_dir = "/workspace/data/fmnist/"

        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=apply_transform
        )

        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=apply_transform
        )

        # sample training data amongst clients
        if cfg.federatedlearning.iid:
            # Sample IID client data from Mnist
            client_groups = mnist_iid(
                train_dataset, cfg.federatedlearning.num_clients
            )
        else:
            # Sample Non-IID client data from Mnist
            if cfg.federatedlearning.unequal:
                # Chose uneuqal splits for every client
                client_groups = mnist_noniid_unequal(
                    train_dataset, cfg.federatedlearning.num_clients
                )
            else:
                # Chose euqal splits for every client
                client_groups = mnist_noniid(
                    train_dataset, cfg.federatedlearning.num_clients
                )

    return train_dataset, test_dataset, client_groups
