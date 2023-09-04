import torch


def no_byzantine(v, f):
    """no faulty workers

    Args:
        v (_type_): gradients
        f (int): num_byzantines
    """
    return v


def gaussian_attack(v, f: int):
    """failures that add Gaussian noise

    Args:
        v (_type_): gradients
        f (int): num_byzantines
    """
    for i in range(f):
        v[i] = torch.randn(v[i].size()) * 200


def bitflip_attack(v, f: int):
    """bit-flipping failure

    Args:
        v (_type_): gradients
        f (int): num_byzantines
    """
    for i in range(f):
        if i > 0:
            v[i][:] = -v[0]
    v[0][:] = -v[0]


def labelflip_attack(label: torch.Tensor) -> torch.Tensor:
    """label-flipping failure

    Args:
        label (torch.Tensor): label data

    Returns:
        torch.Tensor: flipped label data
    """
    return 9 - label
