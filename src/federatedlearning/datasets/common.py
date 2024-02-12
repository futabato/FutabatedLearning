from argparse import Namespace
from typing import Any

import torch
from federatedlearning.datasets.sampling import (
    cifar_iid,
    cifar_noniid,
    mnist_iid,
    mnist_noniid,
    mnist_noniid_unequal,
)
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
        return torch.tensor(image), torch.tensor(label)


def get_dataset(args: Namespace) -> tuple[Any, Any, dict]:
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == "cifar":
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

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups: dict = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == "mnist" or "fmnist":
        if args.dataset == "mnist":
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

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(
                    train_dataset, args.num_users
                )
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups
