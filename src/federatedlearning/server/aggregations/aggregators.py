import copy

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def average_weights(
    local_weights: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Averages the weights from multiple state dictionaries (each representing model parameters).

    Args:
        local_weights (list of dict): A list where each element is a state dictionary of model weights.

    Returns:
        A dict of the same structure as the input but with averaged weights.
    """
    # Initialize the averaged weights with deep copied weights from the first model
    weight_avg: dict[str, torch.Tensor] = copy.deepcopy(local_weights[0])

    # Iterate over each key in the weight dictionary
    for weight_key in weight_avg.keys():
        # Sum the corresponding weights from all models starting from the second one
        for weight_i in range(1, len(local_weights)):
            weight_avg[weight_key] += local_weights[weight_i][weight_key]
        # Divide the summed weights by the number of models to get the average
        weight_avg[weight_key] = torch.div(
            weight_avg[weight_key], len(local_weights)
        )

    # Return the averaged weights
    return weight_avg


def median_weights(
    local_weights: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Computes the median weights from multiple state dictionaries (each representing model parameters).

    Args:
        local_weights (list of dict): A list where each element is a state dictionary of model weights.

    Returns:
        A dict of the same structure as the input but with median weights.
    """
    # Initialize the median weights with deep copied weights from the first model
    weight_median: dict[str, torch.Tensor] = copy.deepcopy(local_weights[0])

    # Iterate over each key in the weight dictionary
    for weight_key in weight_median.keys():
        # Collect all weights for this specific key across all models
        stacked_weights = torch.stack(
            [
                local_weights[weight_i][weight_key]
                for weight_i in range(len(local_weights))
            ]
        )

        # Compute the median along the 0th dimension
        weight_median[weight_key] = torch.median(stacked_weights, dim=0).values

    # Return the median weights
    return weight_median


def krum(
    weights: list[dict[str, torch.Tensor]], f: int
) -> dict[str, torch.Tensor]:
    """
    Implementation of the Krum algorithm.

    Args:
        weights (list[dict[str, torch.Tensor]]): List of model weights from different workers.
        f (int): Maximum number of Byzantine (malicious) workers.

    Returns:
        dict[str, torch.Tensor]: Selected model weights after applying Krum algorithm.
    """
    num_clients = len(weights)

    # Check if the number of weights is sufficient
    if num_clients <= 2 * f:
        raise ValueError("Not enough weights to tolerate f Byzantine workers.")

    distances = torch.zeros((num_clients, num_clients))

    # Compute pairwise distances between all weight vectors
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = 0
            # Iterate over the named parameters of both models simultaneously.
            for (layer_i, param_i), (layer_j, param_j) in zip(
                weights[i].items(), weights[j].items()
            ):
                # Ensure the layers compared are corresponding layers by checking their names.
                assert layer_i == layer_j, "Layer names do not match"
                # Calculate Euclidean distance for the current layer's parameters and add to the total distance.
                # p=2 specifies that this is the L2 norm, which corresponds to Euclidean distance.
                dist += torch.norm(param_i - param_j, p=2).item()
            distances[i, j] = dist
            distances[j, i] = dist

    scores = torch.zeros(num_clients)

    # Calculate scores for each weight vector
    for i in range(num_clients):
        sorted_dists, _ = torch.sort(distances[i], dim=-1)
        scores[i] = torch.sum(sorted_dists[: (num_clients - f - 1)], dim=-1)
    # Select the weight vector with the smallest score
    selected_index = torch.argmin(scores)

    weight_krum = copy.deepcopy(weights[selected_index])
    return weight_krum


def foolsgold(
    local_weights: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Aggregates the weights from multiple state dictionaries using FoolsGold algorithm.

    Args:
        local_weights (list of dict): A list where each element is a state dictionary of model weights.

    Returns:
        A dict of the same structure as the input but with aggregated weights.
    """

    # Initialize the averaged weights with deep copied weights from the first model
    weight_foolsgold: dict[str, torch.Tensor] = copy.deepcopy(local_weights[0])

    # Convert the local weights into a matrix for cosine similarity calculation
    weight_list = []
    for weights in local_weights:
        weight_vector = torch.cat(
            [torch.flatten(weights[key]) for key in sorted(weights.keys())]
        ).numpy()
        weight_list.append(weight_vector)

    weight_matrix = np.array(weight_list)

    # Calculate cosine similarity matrix
    cosine_sim = cosine_similarity(weight_matrix)

    # Calculate trust scores based on cosine similarity
    similarities_sum = np.sum(cosine_sim, axis=1)
    max_similarity = np.max(similarities_sum)
    min_similarity = np.min(similarities_sum)

    trust_scores = 1 - (
        (similarities_sum - min_similarity) / (max_similarity - min_similarity)
    )

    # Normalize trust scores so that their sum equals to the number of clients
    trust_scores = trust_scores / np.mean(trust_scores)

    # Aggregate weights based on trust scores
    for weight_key in weight_foolsgold.keys():
        weight_foolsgold[weight_key] = sum(
            trust_scores[i] * local_weights[i][weight_key]
            for i in range(len(local_weights))
        )

    return weight_foolsgold


# def marginal_median(
#     gradients: list[list[torch.Tensor]],
#     net: Net,
#     lr: float,
#     num_byzantines: int = 0,
#     byzantine_fn: Callable = no_byzantine,
# ) -> None:
#     """marignal median

#     Args:
#         gradients (list[list[torch.Tensor]]): gradients tensor
#         net (Net): Neural Network Model
#         lr (float): learning rate
#         num_byzantines (int, optional): number of byzantines
#         byzantine_fn (Callable, optional): byzantine attack function
#     """
#     # Concatenate all elements in gradients into param_list
#     param_list: list[torch.Tensor] = [
#         torch.cat([element.view(-1, 1) for element in row], dim=0)
#         for row in gradients
#     ]

#     # Apply the byzantine function to param_list
#     manipulated_param_list: list[torch.Tensor] = byzantine_fn(
#         param_list, num_byzantines
#     )

#     # Sort the concatenated array
#     sorted_array = torch.sort(torch.cat(manipulated_param_list, dim=-1))

#     # Calculate the median_nd based on
#     # whether the shape of sorted_array is odd or even
#     if sorted_array.shape[-1] % 2 == 1:
#         median_manipulated_param_tensor = sorted_array[
#             ..., sorted_array.shape[-1] // 2
#         ]
#     else:
#         median_manipulated_param_tensor = (
#             sorted_array[..., sorted_array.shape[-1] // 2 - 1]
#             + sorted_array[..., sorted_array.shape[-1] // 2]
#         ) / 2.0

#     # Update the parameters in net
#     # using the calculated median_manipulated_param_tensor and lr
#     with torch.no_grad():
#         for param, median_manipulated_param in zip(
#             net.parameters(), median_manipulated_param_tensor
#         ):
#             param.copy_(param.data - lr * median_manipulated_param)


# def zeno(
#     gradients: list[list[torch.Tensor]],
#     net: Net,
#     loss_fn: Callable,
#     lr: float,
#     sample: tuple[torch.Tensor, torch.Tensor],
#     rho_ratio: float = 200,
#     num_trimmed_values: int = 8,
#     device: torch.device = torch.device("cuda:0"),
#     num_byzantines: int = 0,
#     byzantine_fn: Callable = no_byzantine,
# ) -> None:
#     """zeno
#     Reference: https://github.com/xcgoner/icml2019_zeno

#     Args:
#         gradients (list[list[torch.Tensor]]): gradients tensor
#         net (Net): Neural Network Model
#         loss_fn (Callable): _description_
#         lr (float): learning rate
#         sample (tuple[torch.Tensor, torch.Tensor]): zeno mini-batch
#         rho_ratio (float, optional): regularization weight. Defaults to 200.
#         num_trimmed_values (int, optional):
#             number of trimmed workers. Defaults to 8.
#         device (_type_, optional):
#             device type responsible to load a tensor into memory.
#             Defaults to torch.device("cuda:0").
#         num_byzantines (int, optional): number of byzantines. Defaults to 0.
#         byzantine_fn (Callable, optional):
#             byzantine attack function. Defaults to no_byzantine.
#     """
#     # Create flattened tensors from gradients and store them in param_list
#     param_list: list[torch.Tensor] = [
#         torch.cat([element.view(-1, 1) for element in row], dim=0)
#         for row in gradients
#     ]
#     param_list_length: int = len(param_list)  # number of workers

#     # Save a copy of the original network before updating its parameters.
#     original_net = copy.deepcopy(net)

#     # Apply the Byzantine function to param_list and get manipulated_param_list
#     manipulated_param_list: list[torch.Tensor] = byzantine_fn(
#         param_list, num_byzantines
#     )

#     # Step1: 入力データ(data)を現在のネットワークパラメータで処理し、
#     # それに基づいた損失を計算する。ここではまだ何も更新はされない
#     # Initialize data and labels, feed it to the network and calculate loss_1
#     data, label = sample[0].to(device), sample[1].to(device)
#     output = net(data)
#     loss_1: torch.Tensor = loss_fn(output, label).mean().item()
#     scores: list = []
#     # Calculate regularization parameter rho
#     rho: float = lr / rho_ratio  # default: 5e-4

#     # 勾配を使ってパラメータを一時的に更新する
#     # Parameter updates with gradients in a no computational graph environment
#     with torch.no_grad():
#         for i in range(param_list_length):
#             # Update the parameters of the model temporarily
#             for param, manipulated_param in zip(
#                 net.parameters(), manipulated_param_list[i]
#             ):
#                 param.copy_(param.data - lr * manipulated_param)

#             # Step2: 更新したパラメータに対して再度同じデータを入力し、
#             # その出力と目標値(label)との差を表す新しい損失を計算する
#             # Calculate loss_2 after temporary update
#             output = net(data)
#             loss_2 = loss_fn(output, label).mean().item()

#             # このプロセスで求めた loss_1 と loss_2 の差より、
#             # 各勾配がどれだけ減らすか（つまり、その勾配がどれだけ重要か）を評価する
#             # Evaluate score using the difference between loss_1 and loss_2
#             scores.append(
#                 loss_1
#                 - loss_2
#                 - rho * torch.mean(param_list[i].square()).item()
#             )
#         # Check if number of workers is equal to number of calculated scores
#         assert param_list_length == len(scores)

#         # ネットワークパラメータを Aggregator 呼び出し時のものに戻す
#         # Restore the original weights of the network
#         for param, original_param in zip(
#             net.parameters(), original_net.parameters()
#         ):
#             param.copy_(original_param)

#     # scores の情報をもとに、効果的な勾配だけを選び出し、
#     # パラメータを最終的に更新していくindexを用意する
#     # Sort the scores in descending order
#     # and ignore the workers with the lowest scores
#     # scores = [value, ...]
#     # sorted_scores = [(sort 前の value の index, value), ...]
#     sorted_scores: list[tuple] = sorted(
#         enumerate(scores), key=lambda x: x[1], reverse=True
#     )

#     # 最低スコアの num_trimmed_values 個の worker を無視する
#     # つまり更新に採用するのは {num_clientss - num_trimmed_values} 個 の worker になる
#     # Use top {num_clientss - num_trimmed_values} scoring workers
#     # only for final update
#     selected_param_list: list[torch.Tensor] = [
#         param_list[i] for i, _ in sorted_scores[:-num_trimmed_values]
#     ]

#     # Calculate the mean of the selected parameters
#     mean_manipulated_param_tensor: torch.Tensor = torch.mean(
#         torch.cat(selected_param_list, dim=-1), dim=-1
#     )

#     # Update the parameters in net
#     # using the calculated mean_manipulated_param_tensor and lr
#     with torch.no_grad():
#         for param, mean_manipulated_param in zip(
#             net.parameters(), mean_manipulated_param_tensor
#         ):
#             param.copy_(param.data - lr * mean_manipulated_param)
