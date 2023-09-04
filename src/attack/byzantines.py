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


def bitflip_attack(v: list[torch.Tensor], f: int) -> list[torch.Tensor]:
    """bit-flipping failure

    Args:
        v (list[torch.Tensor]): gradients
        f (int): num_byzantines

    Returns:
        list[torch.Tensor]: bit flipped data
    """
    flipped_v = []
    for worker, tensor in enumerate(v):
        flipped_tensor = tensor.clone()
        if worker < f:
            flipped_tensor.view(-1)[:] = 1 - flipped_tensor.view(-1)[:]
        flipped_v.append(flipped_tensor)
    return flipped_v


def labelflip_attack(label: torch.Tensor) -> torch.Tensor:
    """label-flipping failure

    Args:
        label (torch.Tensor): label data

    Returns:
        torch.Tensor: flipped label data
    """
    return 9 - label
