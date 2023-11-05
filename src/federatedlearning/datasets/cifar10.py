import torchvision.datasets as datasets
from torch.utils.data import random_split


class Cifar10Dataset:
    def __init__(
        self,
        transform,
        data_root_path: str = "/workspace/data",
    ) -> None:
        """Load CIFAR-10 datasets

        Args:
            transform (callable): functional image transform
            data_root_path (str): path to data directory.
        """
        self.train_dataset = datasets.CIFAR10(
            root=data_root_path,
            train=True,
            transform=transform,
            download=True,
        )
        self.val_dataset = datasets.CIFAR10(
            root=data_root_path,
            train=False,
            transform=transform,
            download=True,
        )
        self.train_dataset, self.zeno_dataset = random_split(
            dataset=self.train_dataset, lengths=[45000, 5000]
        )


"""
# Create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=is_shuffle, drop_last=True
)
val_train_loader = DataLoader(
    val_train_dataset, batch_size=batch_size, shuffle=False, drop_last=False
)
val_test_loader = DataLoader(
    val_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
)
zeno_data = DataLoader(
    zeno_dataset, batch_size=zeno_batch_size, shuffle=True, drop_last=False
)

zeno_iter = iter(zeno_data)
# zeno_iter = itertools.cycle(zeno_data)
"""

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
