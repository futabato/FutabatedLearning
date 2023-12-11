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
    浮動小数点の符号を反転させることを想定。結果的に 1-value を計算している。

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


def chosen_labelflip_attack(
    label: torch.Tensor,
    choice_src_label: int = 5,
    choice_dst_label: int = 3,
) -> torch.Tensor:
    """chosen label-flipping failure

    Args:
        label (torch.Tensor): label tensor
        choice_src_label (int, optional):
            label number of the source. Defaults to 5(dog).
        choice_dst_label (int, optional):
            label number of the destination. Defaults to 3(cat).

    Returns:
        torch.Tensor: flipped label tensor
    """
    return torch.where(label == choice_src_label, choice_dst_label, label)
