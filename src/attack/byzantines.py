import torch


def no_byzantine(weights: torch.Tensor) -> torch.Tensor:
    """no faulty client

    Args:
        weights (torch.Tensor): weights tensor
    """
    return weights


def gaussian_attack(
    weight: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """
    Introduce failures by adding Gaussian noise to the given weights.

    Args:
        weight (dict[str, torch.Tensor]): The original weights of a neural network model.

    Returns:
        dict[str, torch.Tensor]: New weights with Gaussian noise added.
    """
    # Define parameters for Gaussian noise
    mean: float = 0.0  # Mean of the Gaussian distribution
    std: float = 0.01  # Standard deviation of the Gaussian distribution

    # Create a new dictionary to store the perturbed weights
    noisy_weight: dict[str, torch.Tensor] = {}

    # Iterate over all weight tensors in the input dictionary
    for name, tensor in weight.items():
        # Generate Gaussian noise with the same shape as the weight tensor
        noise = torch.randn(tensor.size(), device=device) * std + mean

        # Add the Gaussian noise to the weights to introduce perturbation
        noisy_weight[name] = tensor + noise

    return noisy_weight


def bitflip_attack(
    weight: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Simulate bit-flipping failure by inverting the sign of the floating point values.
    This is roughly approximated by calculating 1-value for each weight.

    Args:
        weight (dict[str, torch.Tensor]): The original weights of a neural network model.

    Returns:
        dict[str, torch.Tensor]: Weights with bit flipped (sign inverted).
    """
    # Create a new dictionary to store the bit-flipped weights
    flipped_weights: dict[str, torch.Tensor] = {}

    # Iterate over all weight tensors in the input dictionary
    for name, tensor in weight.items():
        # Invert the sign of the weight values by applying the transformation 1 - value
        flipped_weights[name] = 1 - tensor

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
