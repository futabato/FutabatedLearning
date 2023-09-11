import torch


def no_byzantine(
    gradients_list: list[torch.Tensor], _: int
) -> list[torch.Tensor]:
    """no faulty workers

    Args:
        gradients_list (list[torch.Tensor]): gradients tensor list
        _ (int): number of byzantines
    """
    return gradients_list


def gaussian_attack(
    gradients_list: list[torch.Tensor], num_byzantines: int
) -> list[torch.Tensor]:
    """failures that add Gaussian noise

    Args:
        gradients_list (list[torch.Tensor]): gradients tensor list
        num_byzantines (int): number of byzantines

    Returns:
        list[torch.Tensor]: gradients tensor list added gaussian noise
    """
    for i in range(num_byzantines):
        gradients_list[i] = torch.randn(gradients_list[i].size()) * 200
    return gradients_list


def bitflip_attack(
    gradients_list: list[torch.Tensor], num_byzantines: int
) -> list[torch.Tensor]:
    """bit-flipping failure

    Args:
        gradients_list (list[torch.Tensor]): gradients tensor list
        num_byzantines (int): number of byzantines

    Returns:
        list[torch.Tensor]: bit flipped gradients tensor list
    """
    bitflipped_gradients_list = []
    for worker_idx, gradients_tensor in enumerate(gradients_list):
        flipped_gradients_tensor = gradients_tensor.clone()
        if worker_idx < num_byzantines:
            flipped_gradients_tensor.view(-1)[:] = (
                1 - flipped_gradients_tensor.view(-1)[:]
            )
        bitflipped_gradients_list.append(flipped_gradients_tensor)
    return bitflipped_gradients_list


def labelflip_attack(label: torch.Tensor) -> torch.Tensor:
    """label-flipping failure

    Args:
        label (torch.Tensor): label tensor

    Returns:
        torch.Tensor: flipped label tensor
    """
    return 9 - label


def clever_labelflip_attack(label: torch.Tensor) -> torch.Tensor:
    """more clever label-flipping failure

    Args:
        label (torch.Tensor): label tensor

    Returns:
        torch.Tensor: flipped label tensor
    """
    flipped_label = torch.where(label == 3, 10, label)
    flipped_label = torch.where(flipped_label == 5, 3, flipped_label)
    flipped_label = torch.where(flipped_label == 10, 5, flipped_label)
    return flipped_label
