from typing import Any

import torch
from federatedlearning.models.cnn import CNNMnist
from nptyping import DataFrame
from omegaconf import DictConfig
from scipy.stats import linregress


def log_total_distances(
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


def monitore_time_series(
    client_id: int,
    round: int,
    client_behavior_df: list[DataFrame],
    euclidean_distance_list: list[list[float]],
    cfg: DictConfig,
    time_series_threshold: float = 2.0,
) -> tuple[bool, list[list[float]]]:
    """
    Monitors the time series of Euclidean distances between the local and global models
    for a given client and rounds. If the slope of the Euclidean distances over the last
    two rounds exceeds the specified threshold, the client is marked as a Byzantine client.

    Args:
        client_id (int): The ID of the client being monitored.
        round (int): The current round of federated learning.
        client_behavior_df (list[DataFrame]): A list of DataFrames containing client behavior information.
        euclidean_distance_list (list[list[float]]): A list of lists containing the Euclidean distances for each client and round.
        cfg (DictConfig): The configuration parameters.
        time_series_threshold (float, optional): The threshold for the slope of Euclidean distances. Defaults to 2.0.

    Returns:
        tuple[bool, list[list[float]]]: A tuple containing a boolean indicating whether the client is reliable
        and the updated list of Euclidean distances for each client and round.
    """
    is_reliable: bool = True  # Initialize is_reliable as True
    # Check if it's not the first round
    if round > 0:
        # Load the local model weights for the current client and round
        local_model = CNNMnist(cfg)
        local_model.load_state_dict(
            torch.load(
                client_behavior_df[client_id]["local_weight_path"][round]
            )
        )

        # Load the global model weights from the previous round
        global_model = CNNMnist(cfg)
        global_model.load_state_dict(
            torch.load(
                f"/workspace/outputs/weights/server/global_model_round_{round-1}.pth"
            )
        )

        # Calculate and store the Euclidean distance between the local and global models
        euclidean_distance_list[client_id][round] = log_total_distances(
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
        if time_series_threshold <= slope:
            print(f"CLIENT {client_id} is BYZANTINE CLIENT!!!")
            is_reliable = False  # Mark the client as unreliable

    return (is_reliable, euclidean_distance_list)
