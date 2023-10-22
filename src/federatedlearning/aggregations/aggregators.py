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
    with torch.no_grad():
        for param, manipulated_param in zip(
            net.parameters(), mean_manipulated_param_tensor
        ):
            param.copy_(param.data - lr * manipulated_param)


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
    gradients: list[list[torch.Tensor]],
    net: Net,
    loss_fn: Callable,
    lr: float,
    sample: tuple[torch.Tensor, torch.Tensor],
    rho_ratio: float = 200,
    num_trimmed_values: int = 8,
    device: torch.device = torch.device("cuda:0"),
    num_byzantines: int = 0,
    byzantine_fn: Callable = no_byzantine,
) -> None:
    """zeno
    Reference: https://github.com/xcgoner/icml2019_zeno

    Args:
        gradients (list[list[torch.Tensor]]): gradients tensor
        net (Net): Neural Network Model
        loss_fn (Callable): _description_
        lr (float): learning rate
        sample (tuple[torch.Tensor, torch.Tensor]): zeno mini-batch
        rho_ratio (float, optional): regularization weight. Defaults to 200.
        num_trimmed_values (int, optional):
            number of trimmed workers. Defaults to 8.
        device (_type_, optional):
            device type responsible to load a tensor into memory.
            Defaults to torch.device("cuda:0").
        num_byzantines (int, optional): number of byzantines. Defaults to 0.
        byzantine_fn (Callable, optional):
            byzantine attack function. Defaults to no_byzantine.
    """
    # Create flattened tensors from gradients and store them in param_list
    param_list: list[torch.Tensor] = [
        torch.cat([element.view(-1, 1) for element in row], dim=0)
        for row in gradients
    ]
    param_list_length: int = len(param_list)

    # Save a copy of the original network before updating its parameters.
    original_net = copy.deepcopy(net)

    # Apply the Byzantine function to param_list and get manipulated_param_list
    manipulated_param_list: list[torch.Tensor] = byzantine_fn(
        param_list, num_byzantines
    )

    # Step1: 入力データ(data)を現在のネットワークパラメータで処理し、
    # それに基づいた損失を計算する。ここではまだ何も更新はされない
    # Initialize data and labels, feed it to the network and calculate loss_1
    data, label = sample[0].to(device), sample[1].to(device)
    output = net(data)
    loss_1: torch.Tensor = loss_fn(output, label).mean().item()
    scores: list = []
    # Calculate regularization parameter rho
    rho: float = lr / rho_ratio  # default: 5e-4

    # 勾配を使ってパラメータを一時的に更新する
    # Parameter updates with gradients in a no computational graph environment
    with torch.no_grad():
        for i in range(param_list_length):
            # Update the parameters of the model temporarily
            for param, manipulated_param in zip(
                net.parameters(), manipulated_param_list[i]
            ):
                param.copy_(param.data - lr * manipulated_param)

            # Step2: 更新したパラメータに対して再度同じデータを入力し、
            # その出力と目標値(label)との差を表す新しい損失を計算する
            # Calculate loss_2 after temporary update
            output = net(data)
            loss_2 = loss_fn(output, label).mean().item()

            # このプロセスで求めた loss_1 と loss_2 の差より、
            # 各勾配がどれだけ減らすか（つまり、その勾配がどれだけ重要か）を評価する
            # Evaluate score using the difference between loss_1 and loss_2
            scores.append(
                loss_1
                - loss_2
                - rho * torch.mean(param_list[i].square()).item()
            )
        # Check if number of gradients is equal to number of calculated scores
        assert param_list_length == len(scores)

        # 重みパラメータを Aggregator 呼び出し時のものに戻す
        # Restore the original weights of the network
        for param, original_param in zip(
            net.parameters(), original_net.parameters()
        ):
            param.copy_(original_param)

    # scores の情報をもとに、効果的な勾配だけを選び出し、
    # パラメータを最終的に更新していくindexを用意する
    # Sort the scores in descending order
    # and ignore the workers with the lowest scores
    # scores = [value, ...]
    # sorted_scores = [(sort 前の value の index, value), ...]
    sorted_scores: list[tuple] = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )

    # 最低スコアの num_trimmed_values 個の worker を無視する
    # つまり更新に採用するのは num_workers - num_trimmed_values 個 の worker になる
    # Use top scoring workers only for final update
    selected_param_list: list[torch.Tensor] = [
        param_list[i] for i, _ in sorted_scores[:-num_byzantines]
    ]
    assert num_trimmed_values == param_list_length - len(selected_param_list)

    # Calculate the mean of the selected parameters
    mean_manipulated_param_tensor: torch.Tensor = torch.mean(
        torch.cat(selected_param_list, dim=-1), dim=-1
    )
    print(f"mean_manipulated_param_tensor: {mean_manipulated_param_tensor}")

    # Update the parameters in net
    # using the calculated mean_manipulated_param_tensor and lr
    with torch.no_grad():
        for param, mean_manipulated_param in zip(
            net.parameters(), mean_manipulated_param_tensor
        ):
            param.copy_(param.data - lr * mean_manipulated_param)
