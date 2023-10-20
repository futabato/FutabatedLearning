from typing import Callable

import numpy as np
import torch

from federatedlearning.aggregations.scores import calc_sum_distances
from federatedlearning.models.model import Net


def no_byzantine(v, f):
    pass


def marginal_median(gradients, net, lr, f=0, byzantine_fn=no_byzantine):
    # X is a 2d list of nd array

    # Concatenate all elements in gradients into param_list
    param_list = [
        torch.cat([xx.view(-1, 1) for xx in x], dim=0) for x in gradients
    ]

    # Apply the byzantine function to param_list
    byzantine_fn(param_list, f)

    # Sort the concatenated array
    sorted_array = torch.sort(torch.cat(param_list, dim=-1))

    # Calculate the median_nd based on
    # whether the shape of sorted_array is odd or even
    if sorted_array.shape[-1] % 2 == 1:
        median_nd = sorted_array[..., sorted_array.shape[-1] // 2]
    else:
        median_nd = (
            sorted_array[..., sorted_array.shape[-1] // 2 - 1]
            + sorted_array[..., sorted_array.shape[-1] // 2]
        ) / 2.0

    # Update the parameters in net using the calculated median_nd and lr
    idx = 0
    for _, param in enumerate(net.parameters()):
        if param.requires_grad:
            numel = param.data.numel()
            param.data -= lr * median_nd[idx : (idx + numel)].view(
                param.data.shape
            )
            idx += numel


def simple_mean(
    gradients: list[list[torch.Tensor]],
    net: Net,
    lr: float,
    num_byzantines: int = 0,
    byzantine_fn: Callable = no_byzantine,
) -> None:
    """simple mean aggregation

    Args:
        gradients (list[list[torch.Tensor]]): gradients tensor
        net (Net): Neural Network Model
        lr (float): learning rate
        num_byzantines (int, optional): number of byzantines
        byzantine_fn (Callable, optional): byzantine attack function
    """
    # X is a 2d list of nd array

    # Concatenate all elements in gradients into param_list
    param_list: list[torch.Tensor] = [
        torch.cat([element.view(-1, 1) for element in row], dim=0)
        for row in gradients
    ]

    # Apply the byzantine function to param_list
    manipulated_param_list: list[torch.Tensor] = byzantine_fn(
        param_list, num_byzantines
    )
    # Calculate the mean_manipulated_param_tensor
    # by taking the mean along the last dimension of the concatenated array
    mean_manipulated_param_tensor: torch.Tensor = torch.mean(
        torch.cat(manipulated_param_list, dim=-1), dim=-1
    )
    # Update the parameters in net
    # using the calculated mean_manipulated_param_tensor and lr
    for param, manipulated_param in zip(
        net.parameters(), mean_manipulated_param_tensor
    ):
        param.data -= lr * manipulated_param


def krum(gradients, net, lr, f=0, byzantine_fn=no_byzantine):
    # X is a 2d list of nd array

    # Concatenate all elements in gradients into param_list
    param_list = [
        torch.cat([xx.view(-1, 1) for xx in x], dim=0) for x in gradients
    ]

    # Apply the byzantine function to param_list
    byzantine_fn(param_list, f)

    # Concatenate all elements in param_list into v
    v = torch.cat(param_list, dim=-1)

    # Calculate the scores based on the sum distances
    # between each gradient and v
    scores = torch.tensor(
        [calc_sum_distances(gradient, v, f) for gradient in param_list]
    )

    # Find the index of the minimum score
    min_idx = scores.argmin().item()

    # Extract the krum_nd from param_list using the minimum score index
    krum_nd = param_list[min_idx].view(-1)

    # Update the parameters in net using the calculated krum_nd and lr
    idx = 0
    for _, param in enumerate(net.parameters()):
        if param.requires_grad:
            numel = param.data.numel()
            param.data -= lr * krum_nd[idx : (idx + numel)].view(
                param.data.shape
            )
            idx += numel


def zeno(
    gradients,
    net,
    loss_fun,
    lr,
    sample,
    rho_ratio,
    b,
    device,
    f=0,
    byzantine_fn=no_byzantine,
):
    # X is a 2d list of nd array

    # Concatenate all elements in gradients into param_list
    param_list = [
        torch.cat([xx.view(-1, 1) for xx in x], dim=0) for x in gradients
    ]

    # Clone the current parameters of the network
    param_net = [xx.data.clone() for xx in net.parameters()]

    # Apply the byzantine function to param_list
    byzantine_fn(param_list, f)

    # Get the output of the network on the given sample
    data, label = sample[0].to(device), sample[1].to(device)
    output = net(data)

    # Calculate the initial loss (loss_1)
    loss_1_nd = loss_fun(output, label)
    loss_1 = loss_1_nd.mean().item()

    scores = []
    rho = lr / rho_ratio

    # Compute the scores for each parameter in param_list
    for _, param in enumerate(param_list):
        idx = 0
        for _, p in enumerate(net.parameters()):
            if p.requires_grad:
                numel = p.data.numel()
                p.data = param_net[_] - lr * param[idx : (idx + numel)].view(
                    p.data.shape
                )
                idx += numel
        output = net(data)
        loss_2_nd = loss_fun(output, label)
        loss_2 = loss_2_nd.mean().item()

        scores.append(loss_1 - loss_2 - rho * param.square().mean().item())

    scores_np = np.array(scores)
    scores_idx = np.argsort(scores_np)

    # Select the indices of the highest scores
    scores_idx = scores_idx[-(len(param_list) - b) :].tolist()
    g_aggregated = torch.zeros_like(param_list[0])

    # Aggregate the gradients corresponding to the selected indices
    for idx in scores_idx:
        g_aggregated += param_list[idx]
    g_aggregated /= float(len(scores_idx))

    idx = 0
    for _, param in enumerate(net.parameters()):
        if param.requires_grad:
            numel = param.data.numel()
            param.data = param_net[_] - lr * g_aggregated[
                idx : (idx + numel)
            ].view(param.data.shape)
            idx += numel
