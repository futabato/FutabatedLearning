import torch


def no_byzantine(weights: torch.Tensor) -> torch.Tensor:
    """no faulty client

    Args:
        weights (torch.Tensor): weights tensor
    """
    return weights


def gaussian_attack(weights: torch.Tensor) -> torch.Tensor:
    """failures that add Gaussian noise

    Args:
        weights(torch.Tensor): weights tensor

    Returns:
        torch.Tensor: weights tensor added gaussian noise
    """
    added_noise_weights: torch.Tensor = weights.clone()
    added_noise_weights = (
        torch.randn(  # Give a random number on a scale of 0 ~ 200
            weights.size() * 200
        )
    )
    return added_noise_weights


def bitflip_attack(weights: torch.Tensor) -> torch.Tensor:
    """bit-flipping failure
    Assuming the sign of the floating point is inverted.
    The implementation is to calculate 1-value.

    Args:
        weights (torch.Tensor): weights tensor

    Returns:
        torch.Tensor: bit flipped weights tensor
    """
    flipped_weights: torch.Tensor = weights.clone()
    flipped_weights.view(-1)[:] = 1 - flipped_weights.view(-1)[:]
    return flipped_weights


def labelflip_attack(label: torch.Tensor) -> torch.Tensor:
    """label-flipping failure

    Args:
        label (torch.Tensor): label tensor

    Returns:
        torch.Tensor: flipped label tensor
    """
    return torch.Tensor(9 - label)


def chosen_labelflip_attack(
    label: torch.Tensor,
    choice_src_label: int = 5,
    choice_dst_label: int = 3,
) -> torch.Tensor:
    """chosen label-flipping failure

    Args:
        label (torch.Tensor): label tensor
        choice_src_label (int, optional):
            label number of the source. Defaults to 5(dog in CIFAR10).
        choice_dst_label (int, optional):
            label number of the destination. Defaults to 3(cat in CIFAR10).

    Returns:
        torch.Tensor: flipped label tensor
    """
    return torch.where(label == choice_src_label, choice_dst_label, label)
