from logging import getLogger
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from federatedlearning.models.cnn import CNNMnist
from models.resnet import ResNet18
from nptyping import DataFrame
from omegaconf import DictConfig
from scipy.spatial.distance import cosine
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

logger = getLogger(__name__)


def calc_total_distances(
    global_model: CNNMnist | Any, local_model: CNNMnist | Any
) -> float:
    """
    Calculate the total Euclidean distance between the parameters of two models.

    This function assumes that both models have the same architecture and they
    are instances of the class `CNNMnist` or any other class with an accessible
    `named_parameters()` method providing name-parameter pairs.

    Args:
        global_model (CNNMnist|Any): The global model.
        local_model (CNNMnist|Any): The local model updated by a client.

    Returns:
        float: The total Euclidean distance between the parameters of the two models.
    """
    # Initialize the distance to zero.
    distance: float = 0.0

    # Iterate over the named parameters of both models simultaneously.
    for (layer_global, param_global), (layer_local, param_local) in zip(
        global_model.named_parameters(), local_model.named_parameters()
    ):
        # Ensure the layers compared are corresponding layers by checking their names.
        assert layer_global == layer_local, "Layer names do not match"

        # Calculate Euclidean distance for the current layer's parameters and add to the total distance.
        # p=2 specifies that this is the L2 norm, which corresponds to Euclidean distance.
        distance += torch.norm(param_global - param_local, p=2).item()

    # Return the total Euclidean distance calculated.
    return distance


# Analyse variations in output layer weights.
def extract_output_layer_weights(model: nn.Module) -> np.ndarray:
    update = model.state_dict()
    return update["fc2.weight"].detach().numpy()


# Calculate the amount of weight variation to detect anomalies (compared with previous rounds)
def compare_previous_round(
    current_weights: nn.Module,
    previous_weights: nn.Module,
    threshold: float = 0.5,
) -> tuple[bool, list[list[float]]]:
    """
    Function to detect abnormal weight variations.

    This function compares the current weights with the previous weights to detect any abnormal variations.
    It calculates the variation of each weight and checks if the total variation exceeds a specified threshold.

    Parameters:
    current_weights (list of np.ndarray): List of current client weights.
    previous_weights (list of np.ndarray): List of previous client weights.
    threshold (float): Threshold for detecting anomalies. Default is 0.5.

    Returns:
    tuple:
        bool: True if the total variation exceeds the threshold, otherwise False.
        list: List of variations for each client.
    """
    # Calculate the weight variations for each client
    variations = [
        np.linalg.norm(current - previous)
        for current, previous in zip(current_weights, previous_weights)
    ]
    # print('Weight variations:', variations)

    # Check if the total variation exceeds the threshold
    total_variations = sum(variations)
    # print(f"{total_variations}")

    if total_variations > threshold:
        return True, variations
    return False, variations


def check_output_layer_changes(
    current_weight: nn.Module, previous_weight: nn.Module, threshold: float
) -> tuple[bool, list[float]]:
    """
    Check for significant changes in output layer weights between two rounds.

    Args:
    - model_current (nn.Module): Current round's model.
    - model_previous (nn.Module): Previous round's model.
    - threshold (float): Cosine similarity threshold for detecting abnormal changes.

    Returns:
    - bool: True if an abnormal change is not detected, False otherwise.
    """
    is_reliable_list: list[bool] = [True] * len(current_weight)
    cos_sim_list = []
    for node in range(len(current_weight)):
        # Calculate cosine similarity
        cos_sim = cosine_similarity(
            current_weight[node].reshape(1, -1),
            previous_weight[node].reshape(1, -1),
        )
        cos_sim_list.append(cos_sim)
        # Check against the threshold
        # 遠ざかるほど小さくなるはず -> 小さいと怪しい
        if cos_sim < threshold:
            print(f"[Node {node}] {cos_sim=}")
            is_reliable_list[node] = False

    # 1つでもFalseがあってはいけない
    return all(is_reliable_list), cos_sim_list


def tsne_clustering(
    data: np.ndarray, n_clusters: int = 2
) -> tuple[np.ndarray, np.ndarray, Iterable[int]]:
    # Dimensional compression with t-SNE.
    tsvd_data = TruncatedSVD(n_components=50).fit_transform(data)
    tsne = TSNE(n_components=n_clusters, random_state=42, perplexity=3)
    tsne_result = tsne.fit_transform(tsvd_data)

    # Clustering with k-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tsne_result)
    cluster_labels = kmeans.labels_

    return tsne_result, clusters, cluster_labels


def visualize_clusters(
    tsne_result: np.ndarray, clusters: Iterable[int], round_number: int
) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(
        tsne_result[:, 0],
        tsne_result[:, 1],
        c=clusters,
        cmap="viridis",
        alpha=0.5,
    )
    plt.title(
        f"Clients Clustering with t-SNE and k-means (Round {round_number})"
    )
    plt.xlabel("t-SNE feature-1")
    plt.ylabel("t-SNE feature-2")
    plt.colorbar()
    plt.show()


def flatten_state_dict(state_dict: nn.Module) -> np.ndarray:
    flattened = []
    for _, value in state_dict.named_parameters():
        flattened.append(value.flatten().detach().numpy())
    return np.concatenate(flattened)


def eliminate_byzantine_clients(
    local_weights: list[dict[str, torch.Tensor]],
    local_losses: list[float],
    byzantine_clients: set[int],
) -> tuple[list[dict[str, torch.Tensor]], list[float]]:
    local_weights = [
        v for i, v in enumerate(local_weights) if i not in byzantine_clients
    ]
    local_losses = [
        v for i, v in enumerate(local_losses) if i not in byzantine_clients
    ]
    return local_weights, local_losses


def monitor_time_series_convergence(
    client_id: int,
    round: int,
    global_model_record_df: DataFrame,
    byzantine_clients: set[int],
    client_history_df: list[DataFrame],
    euclidean_distance_list: list[list[float]],
    cfg: DictConfig,
) -> bool:
    """
    Monitors the time series of Euclidean distances between the local and global models
    for a given client and rounds. If the slope of the Euclidean distances over the last
    two rounds exceeds the specified threshold, the client is marked as a Byzantine client.

    Args:
        client_id (int): The ID of the client being monitored.
        round (int): The current round of federated learning.
        client_history_df (list[DataFrame]): A list of DataFrames containing client history information.
        euclidean_distance_list (list[list[float]]): A list of lists containing the Euclidean distances for each client and round.
        cfg (DictConfig): The configuration parameters.
        time_series_threshold (float, optional): The threshold for the slope of Euclidean distances. Defaults to 2.0.

    Returns:
        tuple[bool, list[list[float]]]: A tuple containing a boolean indicating whether the client is reliable
        and the updated list of Euclidean distances for each client and round.
    """
    if client_id in byzantine_clients:
        return False

    is_reliable: bool = True  # Initialize is_reliable as True

    if cfg.train.dataset == "mnist":
        global_model = CNNMnist(cfg=cfg)
        local_model = CNNMnist(cfg=cfg)
    elif cfg.train.dataset == "cifar":
        global_model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        global_model.fc = torch.nn.Linear(global_model.fc.in_features, 10)
        local_model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        local_model.fc = torch.nn.Linear(local_model.fc.in_features, 10)
    # Check if it's not the first round
    if round > 0:
        # Load the local model weights for the current client and round
        local_model.load_state_dict(
            torch.load(
                client_history_df[client_id]["local_weight_path"][round]
            )
        )

        # Load the global model weights from the previous round
        global_model.load_state_dict(
            torch.load(global_model_record_df["global_weight_path"][round - 1])
        )

        # Calculate and store the Euclidean distance between the local and global models
        euclidean_distance_list[client_id][round] = calc_total_distances(
            global_model, local_model
        )

        # Calculate the slope of the Euclidean distances over the last two rounds
        slope, _, _, _, _ = linregress(
            [round - 1, round],
            [
                euclidean_distance_list[client_id][round - 1],
                euclidean_distance_list[client_id][round],
            ],
        )
        # Check if the slope exceeds the threshold
        if cfg.federatedlearning.time_series_convergence_threshold <= slope:
            print(
                f"[TimeSeriesConvergence] CLIENT {client_id} is BYZANTINE CLIENT!!!"
            )
            is_reliable = False  # Mark the client as unreliable

    # return (is_reliable, euclidean_distance_list)
    return is_reliable


def monitor_time_series_similarity(
    client_id: int,
    round: int,
    byzantine_clients: set[int],
    client_history_df: list[DataFrame],
    cfg: DictConfig,
) -> bool:
    if client_id in byzantine_clients:
        return False

    if cfg.train.dataset == "mnist":
        previous_local_model = CNNMnist(cfg)
        current_local_model = CNNMnist(cfg)
    elif cfg.train.dataset == "cifar":
        previous_local_model = torchvision.models.resnet18(
            weights="IMAGENET1K_V1"
        )
        previous_local_model.fc = torch.nn.Linear(
            previous_local_model.fc.in_features, 10
        )
        current_local_model = torchvision.models.resnet18(
            weights="IMAGENET1K_V1"
        )
        current_local_model.fc = torch.nn.Linear(
            current_local_model.fc.in_features, 10
        )

    is_reliable: bool = True
    if round > 0:
        previous_local_model.load_state_dict(
            torch.load(
                client_history_df[client_id]["local_weight_path"][round - 1]
            )
        )
        current_local_model.load_state_dict(
            torch.load(
                client_history_df[client_id]["local_weight_path"][round]
            )
        )
        previous_weights = extract_output_layer_weights(previous_local_model)
        current_weights = extract_output_layer_weights(current_local_model)

        is_reliable, _ = check_output_layer_changes(
            current_weight=current_weights,
            previous_weight=previous_weights,
            threshold=cfg.federatedlearning.time_series_similarity_threshold,
        )

        if not is_reliable:
            print(
                f"[TimeSeriesSimilarity] CLIENT {client_id} is BYZANTINE CLIENT!!!"
            )

    return is_reliable


def monitor_trust_scored_clustering(
    round: int,
    selected_client_idx: set[int],
    byzantine_clients: set[int],
    client_history_df: list[DataFrame],
    cfg: DictConfig,
) -> set[int]:
    local_model_updates = []
    if cfg.train.dataset == "mnist":
        local_model = CNNMnist(cfg=cfg)
    elif cfg.train.dataset == "cifar":
        local_model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        local_model.fc = torch.nn.Linear(local_model.fc.in_features, 10)

    print(f"[TrustScoredClustering] {selected_client_idx=}")
    print(f"[TrustScoredClustering] {byzantine_clients=}")

    for client_id in selected_client_idx:
        if client_id not in byzantine_clients:
            local_model.load_state_dict(
                torch.load(
                    client_history_df[client_id]["local_weight_path"][round]
                )
            )
            weight_local_model = flatten_state_dict(local_model)
            local_model_updates.append(weight_local_model)

    updates = np.array(local_model_updates)

    # perplexity must be less than n_samples
    if len(updates) <= 3:
        return byzantine_clients

    # Perform clustering in each round and record the results
    # Step 1: Apply t-SNE for dimensionality reduction to 2D
    # Step 2: Clustering with K-means
    tsne_result, clusters, cluster_labels = tsne_clustering(updates)
    # visualize_clusters(tsne_result, clusters, round + 1)

    # Step 2: Calculate silhouette score
    clustering_score = silhouette_score(tsne_result, clusters)
    print(
        f"Round {round + 1} Clustering Silhouette Score: {clustering_score:.2f}"
    )

    if (
        clustering_score
        < cfg.federatedlearning.clustering_silhouette_score_threshold
    ):
        print(f"Round {round + 1}: Clusters are not well separated.")
        return byzantine_clients  # Abort and move to the next round

    # Step 3: Calculate trust scores using FoolsGold
    similarity_matrix = np.zeros(
        (
            len(selected_client_idx) - len(byzantine_clients),
            len(selected_client_idx) - len(byzantine_clients),
        )
    )
    for i in range(len(selected_client_idx)):
        for j in range(i, len(selected_client_idx)):
            if i in byzantine_clients or j in byzantine_clients:
                continue
            sim = 1 - cosine(updates[i], updates[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    trust_scores = np.ones(len(selected_client_idx))
    for i in selected_client_idx:
        for j in selected_client_idx:
            if i in byzantine_clients or j in byzantine_clients:
                continue
            if i != j:
                trust_scores[i] *= 1 - similarity_matrix[i, j]

    trust_scores /= np.max(trust_scores)  # Normalize

    # Step 4: Calculate trust score sum for each cluster and identify anomalous cluster
    cluster_0_sum = trust_scores[cluster_labels == 0].sum()
    cluster_1_sum = trust_scores[cluster_labels == 1].sum()

    anomalous_cluster = 0 if cluster_0_sum < cluster_1_sum else 1

    for client_id in selected_client_idx:
        if client_id in byzantine_clients:
            continue
        if cluster_labels[client_id] == anomalous_cluster:
            byzantine_clients.add(client_id)
            # is_reliable_list[client_id] = False

    print(f"{clusters=}")
    print(f"{cluster_0_sum=}, {cluster_1_sum=}")
    print(f"{anomalous_cluster=}")

    return byzantine_clients
