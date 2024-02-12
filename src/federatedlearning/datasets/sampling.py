#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any

import numpy as np
from nptyping import NDArray
from torchvision import datasets, transforms


def mnist_iid(dataset: Any, num_users: int) -> dict:
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items: int = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(
            np.random.choice(all_idxs, num_items, replace=False)
        )
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset: Any, num_users: int) -> dict[int, NDArray[Any, Any]]:
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard: list[int] = list(range(num_shards))
    dict_users: dict[int, NDArray[Any, Any]] = {
        i: np.array([]) for i in range(num_users)
    }
    idxs: NDArray[Any, Any] = np.arange(num_shards * num_imgs)
    labels: NDArray[Any, Any] = dataset.train_labels.numpy()

    # sort labels
    idxs_labels: NDArray[Any, Any] = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set: set[int] = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                axis=0,
            )
    return dict_users


def mnist_noniid_unequal(
    dataset: Any, num_users: int
) -> dict[int, NDArray[Any, Any]]:
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard: list[int] = list(range(num_shards))
    dict_users: dict[int, NDArray[Any, Any]] = {
        i: np.array([]) for i in range(num_users)
    }
    idxs: NDArray[Any, Any] = np.arange(num_shards * num_imgs)
    labels: NDArray[Any, Any] = dataset.train_labels.numpy()

    # sort labels
    idxs_labels: NDArray[Any, Any] = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard: int = 1
    max_shard: int = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(
        min_shard, max_shard + 1, size=num_users
    )
    random_shard_size = np.around(
        random_shard_size / sum(random_shard_size) * num_shards
    )
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:
        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set: set[int] = set(
                np.random.choice(idx_shard, 1, replace=False)
            )
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (
                        dict_users[i],
                        idxs[rand * num_imgs : (rand + 1) * num_imgs],
                    ),
                    axis=0,
                )

        # NOTE: random_shard_size is not int
        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(
                np.random.choice(idx_shard, shard_size, replace=False)
            )
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (
                        dict_users[i],
                        idxs[rand * num_imgs : (rand + 1) * num_imgs],
                    ),
                    axis=0,
                )
    else:
        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(
                np.random.choice(idx_shard, shard_size, replace=False)
            )
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (
                        dict_users[i],
                        idxs[rand * num_imgs : (rand + 1) * num_imgs],
                    ),
                    axis=0,
                )

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(
                np.random.choice(idx_shard, shard_size, replace=False)
            )
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (
                        dict_users[k],
                        idxs[rand * num_imgs : (rand + 1) * num_imgs],
                    ),
                    axis=0,
                )

    return dict_users


def cifar_iid(dataset: Any, num_users: int) -> dict:
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items: int = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(
            np.random.choice(all_idxs, num_items, replace=False)
        )
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset: Any, num_users: int) -> dict[int, NDArray[Any, Any]]:
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard: list[int] = list(range(num_shards))
    dict_users: dict[int, NDArray[Any, Any]] = {
        i: np.array([]) for i in range(num_users)
    }
    idxs: NDArray[Any, Any] = np.arange(num_shards * num_imgs)
    labels: NDArray[Any, Any] = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set: set[int] = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                axis=0,
            )
    return dict_users


if __name__ == "__main__":
    dataset_train: datasets.MNIST = datasets.MNIST(
        "/workspace/data/mnist/",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    num: int = 100
    d: dict[int, NDArray[Any, Any]] = mnist_noniid(dataset_train, num)
