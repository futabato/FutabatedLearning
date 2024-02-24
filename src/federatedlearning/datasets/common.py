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
        """
        Initialize a subset of a dataset at the provided indices.

        Args:
            dataset (Dataset): The original dataset.
            idxs (list): A list of indices specifying which subset to take.
        """
        self.dataset = dataset
        self.idxs: list[int] = [
            int(i) for i in idxs
        ]  # Ensure indices are integers

    def __len__(self) -> int:
        """Return the length of the subset."""
        return len(self.idxs)

    def __getitem__(self, item: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve an item and its label at the provided index from the subset.

        Args:
            item (Any): The index of the data item.

        Returns:
            A tuple where the first element is the data and the second is the label.
        """
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_dataset(cfg: DictConfig) -> tuple[Any, Any, dict]:
    """
    Prepare the datasets and client groups based on the given configuration for federated learning.

    Args:
        cfg (DictConfig): Configuration object that includes settings for dataset selection and sampling.

    Returns:
        A tuple containing:
            - train_dataset: Dataset object for training.
            - test_dataset: Dataset object for testing.
            - client_groups: A dictionary with client indices as keys and corresponding data indices as values.
    """

    # Initialize transformations and datasets depending on the chosen dataset
    if cfg.train.dataset == "cifar":
        data_dir: str = "/workspace/data/cifar/"
        apply_transform: transforms.Compose = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Download and load CIFAR10 dataset
        train_dataset: Any = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=apply_transform
        )
        test_dataset: Any = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=apply_transform
        )

        # Sample training data amongst clients based on IID or Non-IID
        if cfg.federatedlearning.iid:
            # For IID data distribution across clients
            client_groups: dict = cifar_iid(
                train_dataset, cfg.federatedlearning.num_clients
            )
        else:
            # For Non-IID data distribution
            if cfg.federatedlearning.unequal:
                # If unequal partition requested, raise error (not implemented)
                raise NotImplementedError()
            else:
                # For equal partitions amongst clients
                client_groups = cifar_noniid(
                    train_dataset, cfg.federatedlearning.num_clients
                )

    elif cfg.train.dataset in ["mnist", "fmnist"]:
        # Set the correct directory based on the dataset
        data_dir = f"/workspace/data/{cfg.train.dataset}/"

        # Define transformations for MNIST/Fashion-MNIST
        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # Load MNIST or Fashion-MNIST dataset
        if cfg.train.dataset == "mnist":
            DatasetClass = datasets.MNIST
        else:
            DatasetClass = datasets.FashionMNIST

        train_dataset = DatasetClass(
            data_dir, train=True, download=True, transform=apply_transform
        )
        test_dataset = DatasetClass(
            data_dir, train=False, download=True, transform=apply_transform
        )

        # Sample training data amongst clients based on IID or Non-IID
        if cfg.federatedlearning.iid:
            # For IID data distribution across clients
            client_groups = mnist_iid(
                train_dataset, cfg.federatedlearning.num_clients
            )
        else:
            # For Non-IID data distribution
            if cfg.federatedlearning.unequal:
                # If unequal partition requested, use specific function
                client_groups = mnist_noniid_unequal(
                    train_dataset, cfg.federatedlearning.num_clients
                )
            else:
                # For equal partitions amongst clients
                client_groups = mnist_noniid(
                    train_dataset, cfg.federatedlearning.num_clients
                )

    # Return the training dataset, testing dataset, and the dictionary of client groups
    return train_dataset, test_dataset, client_groups
