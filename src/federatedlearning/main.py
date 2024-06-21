#!/usr/bin/env python

import copy
import math
import os
import pickle
import time
from logging import getLogger
from logging.config import dictConfig

import hydra
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from nptyping import Int, NDArray, Shape
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from federatedlearning.client.training import LocalUpdate
from federatedlearning.datasets.common import get_dataset
from federatedlearning.models.cnn import CNNCifar, CNNMnist
from federatedlearning.reputation.monitoring import (
    monitor_cross_sectional,
    monitore_time_series,
)
from federatedlearning.server.aggregations.aggregators import average_weights
from federatedlearning.server.inferencing import inference

# Set matplotlib backend to 'Agg' to avoid the need for a GUI backend
matplotlib.use("Agg")


@hydra.main(
    version_base="1.1", config_path="/workspace/config", config_name="default"
)
def main(cfg: DictConfig) -> float:  # noqa: C901
    # Record the start time for run duration
    start_time: float = time.time()

    dictConfig(yaml.safe_load(open("/workspace/config/logger.yaml").read()))
    logger = getLogger("Logger")

    # Setup paths and logging utilities
    mlflow.set_tracking_uri(
        "file://" + hydra.utils.get_original_cwd() + "/mlruns"
    )
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Start an MLFlow run and log the Hydra-generated configuration files
    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        RUN_ID = run.info.run_id
        EXPERIMENT_ID = run.info.experiment_id

        logger.info(f"{EXPERIMENT_ID=}")
        logger.info(f"{RUN_ID=}")
        logger.info(f"cfg:\n{OmegaConf.to_yaml(cfg)}")

        mlflow.log_artifact("/workspace/outputs/.hydra/config.yaml")
        mlflow.log_artifact("/workspace/outputs/.hydra/hydra.yaml")
        mlflow.log_artifact("/workspace/outputs/.hydra/overrides.yaml")

        # Log federated learning and training parameters to MLFlow
        mlflow.log_params(cfg.federatedlearning)
        mlflow.log_params(cfg.train)

        # Determine the computational device based on the configuration
        device: torch.device = (
            torch.device(f"cuda:{cfg.train.gpu}")
            if cfg.train.gpu is not None and cfg.train.gpu >= 0
            else torch.device("cpu")
        )

        # Initialize DataFrame to track client update histories during training
        client_history_df: list[pd.DataFrame] = [
            pd.DataFrame(columns=["round", "local_loss", "local_weight_path"])
            for _ in range(cfg.federatedlearning.num_clients)
        ]
        # Create directories for each client
        # to store their model weights after local updates
        for client_id in range(cfg.federatedlearning.num_clients):
            os.makedirs(
                f"/workspace/outputs/weights/client_{client_id}", exist_ok=True
            )
        # Initialize DataFrame to record global model updates after aggregation
        global_model_record_df: pd.DataFrame = pd.DataFrame(
            columns=["round", "global_weight_path"]
        )
        # Create necessary directories for server-side outputs
        os.makedirs("/workspace/outputs/weights/server", exist_ok=True)
        os.makedirs("/workspace/outputs/csv", exist_ok=True)
        os.makedirs("/workspace/outputs/objects", exist_ok=True)

        # Load the dataset and partition it according to the client groups
        train_dataset, test_dataset, client_groups = get_dataset(cfg)

        # Instantiate the appropriate global model based on the dataset being used
        global_model: nn.Module
        if cfg.train.dataset == "mnist":
            global_model = CNNMnist(cfg=cfg)
        elif cfg.train.dataset == "cifar":
            global_model = CNNCifar(cfg=cfg)

        # Prepare the global model for training and
        # send it to the designated computational device
        global_model.to(device)
        global_model.train()
        logger.info(f"global_model:\n{global_model}")

        # Capture initial global model weights before training begins
        global_weights: dict[str, torch.Tensor] = global_model.state_dict()

        # Initialize save_path
        save_path: str
        save_path = "/workspace/outputs/weights/server/global_round_0.pth"
        torch.save(global_weights, save_path)

        # Issue-23: NFS
        [
            os.makedirs(f"/shared/round_{round}")
            for round in range(cfg.federatedlearning.rounds)
        ]

        # Initialize lists to record the training progress
        train_loss: list[float] = []
        train_accuracy: list[float] = []
        # Interval for printing aggregated training stats
        print_every: int = 2

        # Initialize lists to store euclidean distances for a each client across all rounds.
        euclidean_distance_list: list[list[float]] = [
            [math.inf] * cfg.federatedlearning.rounds
        ] * cfg.federatedlearning.num_clients
        cluster_distance_list: list[float] = [
            math.inf
        ] * cfg.federatedlearning.rounds
        finish_cross_sectional: bool = False
        # Initialize an empty set to store Byzantine clients
        byzantine_clients: set[int] = set()

        # Begin federated training loop across specified number of rounds
        for round in tqdm(range(cfg.federatedlearning.rounds)):
            # Collect weights and losses from clients participating in this round
            local_weights: list[dict[str, torch.Tensor]] = []
            local_losses: list[float] = []
            logger.info(f"\n | Global Training Round : {round+1} |\n")

            # Re-enter training mode at the start of each round
            global_model.train()
            # Randomly select clients to participate in this training round
            num_selected_clients: int = max(
                int(
                    cfg.federatedlearning.frac
                    * cfg.federatedlearning.num_clients
                ),
                1,
            )
            selected_clients_idx: NDArray[
                Shape[f"1, {num_selected_clients}"], Int
            ] = np.random.choice(
                range(cfg.federatedlearning.num_clients),
                num_selected_clients,
                replace=False,
            )

            # Loop over each selected client to perform local model updates
            for client_id in selected_clients_idx:
                if client_id in byzantine_clients:
                    continue
                local_model = LocalUpdate(
                    cfg=cfg,
                    dataset=train_dataset,
                    client_id=client_id,
                    idxs=client_groups[client_id],
                )
                # Check if the current index is within the number of byzantine clients specified in the configuration
                if (
                    client_id < cfg.federatedlearning.num_byzantines
                    and cfg.federatedlearning.warmup_rounds <= round
                ):
                    # Perform a byzantine attack on the local model by altering its weights and compute loss
                    weight, loss = local_model.byzantine_attack(
                        model=copy.deepcopy(global_model), global_round=round
                    )
                else:
                    # Otherwise, perform a standard update of model weights based on local data and compute loss
                    weight, loss = local_model.update_weights(
                        model=copy.deepcopy(global_model), global_round=round
                    )
                # Store the updated weights and reported loss
                local_weights.append(copy.deepcopy(weight))
                local_losses.append(copy.deepcopy(loss))
                # Save local model weights and record training details
                save_path = f"/workspace/outputs/weights/client_{client_id}/client_{client_id}_round_{round}.pth"
                torch.save(weight, save_path)
                mlflow.log_artifact(save_path)
                # Issue-23: NFS
                save_path = f"/shared/round_{round}/client_{client_id}.pth"
                torch.save(weight, save_path)

                local_training_info: dict = {
                    "round": round,
                    "local_loss": copy.deepcopy(loss),
                    "local_weight_path": f"/workspace/mlruns/{EXPERIMENT_ID}/{RUN_ID}/artifacts/client_{client_id}_round_{round}.pth",
                }
                # Append recorded details to the client history DataFrame
                client_history_df[client_id] = pd.concat(
                    [
                        client_history_df[client_id],
                        pd.DataFrame(local_training_info, index=[round]),
                    ],
                    ignore_index=True,
                )
                # Export client history data to CSV for analysis or audit
                save_path = (
                    f"/workspace/outputs/csv/client_{client_id}_history.csv"
                )
                client_history_df[client_id].to_csv(
                    save_path,
                    index=False,
                )
                mlflow.log_artifact(save_path)
                # Time-Series Monitoring
                if cfg.federatedlearning.enable_time_series_monitoring:
                    (
                        is_reliable,
                        euclidean_distance_list,
                    ) = monitore_time_series(
                        client_id=client_id,
                        round=round,
                        client_history_df=client_history_df,
                        euclidean_distance_list=euclidean_distance_list,
                        cfg=cfg,
                    )
                    # Check if the client is marked as unreliable (Byzantine client)
                    if not is_reliable:
                        # Remove the local weights and losses of the unreliable client
                        local_weights.pop()
                        local_losses.pop()
                        # Add the client ID to the set of Byzantine clients
                        byzantine_clients.add(client_id)

            # TODO: Cross-sectional Monitoring
            if (
                cfg.federatedlearning.enable_cross_sectional_monitoring
                and not finish_cross_sectional
            ):
                (
                    is_reliable_list,
                    cluster_distance_list,
                ) = monitor_cross_sectional(
                    round,
                    num_selected_clients,
                    client_history_df,
                    cfg,
                    cluster_distance_list,
                )
                exclude_clients: list[int] = []
                for i in range(len(selected_clients_idx)):
                    if not is_reliable_list[i]:
                        exclude_clients.append(selected_clients_idx[i])
                # for a, is_reliable in zip(selected_clients_idx, is_reliable_list):
                # for i, is_reliable in enumerate(
                #     sorted(is_reliable_list, reverse=True)
                # ):
                #     if not is_reliable:
                #         # Remove the local weights and losses of the unreliable client
                #         local_weights.pop(i)
                #         local_losses.pop(i)
                #         # Add the client ID to the set of Byzantine clients
                #         byzantine_clients.add(selected_clients_idx[i])
                #         # cross-sectionalは一度きり
                #         finish_cross_sectional = True
                # Cross-Sectional Monitoring の 除外コード部分
                # for i in sorted(exclude_clients, reverse=True):
                #     local_weights.pop(i)
                #     local_losses.pop(i)
                #     byzantine_clients.add(selected_clients_idx[i])
                # finish_cross_sectional = True

            # TODO: if cfg.federatedlearning.num_byzantines / num_selected_clients > 0.5 and cfg.federatedlearning.warmup_rounds == 0
            # のときどうにかする

            # Aggregate local weights to form new global model weights (FedAVG)
            global_weights = average_weights(local_weights)
            # Save updated global model weights
            save_path = (
                f"/workspace/outputs/weights/server/global_round_{round}.pth"
            )
            torch.save(global_weights, save_path)
            mlflow.log_artifact(save_path)
            # Load the newly aggregated weights into the global model
            global_model.load_state_dict(global_weights)

            # Record and log the average training loss across all participating clients
            loss_avg: float = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            mlflow.log_metric("Train-Loss", loss_avg, step=round)

            # Evaluation phase: compute and log the average training accuracy across all clients
            list_acc: list[float] = []
            list_loss: list[float] = []
            global_model.eval()  # Set model to evaluation mode for inference
            for _ in range(cfg.federatedlearning.num_clients):
                local_model = LocalUpdate(
                    cfg=cfg,
                    dataset=train_dataset,
                    client_id=client_id,
                    idxs=client_groups[client_id],
                )
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)
            train_accuracy.append(sum(list_acc) / len(list_acc))
            mlflow.log_metric(
                "Train-Accuracy", sum(list_acc) / len(list_acc), step=round
            )

            # Record details of the global model's state after the round
            global_model_info: dict = {
                "round": round,
                "global_weight_path": f"/workspace/mlruns/{EXPERIMENT_ID}/{RUN_ID}/artifacts/global_round_{round}.pth",
            }
            global_model_record_df = pd.concat(
                [
                    global_model_record_df,
                    pd.DataFrame(global_model_info, index=[round]),
                ],
                ignore_index=True,
            )
            # Export server-side global model records to CSV
            save_path = "/workspace/outputs/csv/server_record.csv"
            global_model_record_df.to_csv(
                save_path,
                index=False,
            )
            mlflow.log_artifact(save_path)

            # Occasionally print summary statistics of the training progress
            if (round + 1) % print_every == 0:
                logger.info(
                    f" \nAvg Training Stats after {round+1} global rounds:"
                )
                logger.info(f"Training Loss : {np.mean(np.array(train_loss))}")
                logger.info(
                    "Train Accuracy: {:.2f}% \n".format(
                        100 * train_accuracy[-1]
                    )
                )

        # After training completion, evaluate the global model on the test dataset
        test_acc, test_loss = inference(cfg, global_model, test_dataset)
        # Log final test metrics to MLFlow
        mlflow.log_metric("Test-Accuracy", test_acc)
        mlflow.log_metric("Test-Loss", test_loss)

        # Print final training results
        logger.info(
            f" \n Results after {cfg.federatedlearning.rounds} global rounds of training:"
        )
        logger.info(
            "|---- Avg Train Accuracy: {:.2f}%".format(
                100 * train_accuracy[-1]
            )
        )
        logger.info("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

        # Save the training loss and accuracy data for future reference
        file_name: str = "{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]".format(
            cfg.train.dataset,
            global_model.__class__.__name__,
            cfg.federatedlearning.rounds,
            cfg.federatedlearning.frac,
            cfg.federatedlearning.iid,
            cfg.train.local_epochs,
            cfg.train.local_batch_size,
        )

        # Open a file to save the train_loss and train_accuracy lists using pickle
        save_path = f"/workspace/outputs/objects/{file_name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump([train_loss, train_accuracy], f)
        # Log the file containing training data as an artifact in MLFlow
        mlflow.log_artifact(save_path)

        # Plot and save the Training Loss vs Communication rounds
        save_path = f"/workspace/outputs/objects/fed_{file_name}_loss.png"
        plt.figure()
        plt.title("Training Loss vs Communication rounds")
        plt.plot(range(len(train_loss)), train_loss, color="r")
        plt.ylabel("Training loss")
        plt.xlabel("Communication Rounds")
        plt.savefig(save_path)
        mlflow.log_artifact(save_path)

        # Plot and save the Average Accuracy vs Communication rounds
        save_path = f"/workspace/outputs/objects/fed_{file_name}_acc.png"
        plt.figure()
        plt.title("Average Accuracy vs Communication rounds")
        plt.plot(range(len(train_accuracy)), train_accuracy, color="k")
        plt.ylabel("Average Accuracy")
        plt.xlabel("Communication Rounds")
        plt.savefig(save_path)
        mlflow.log_artifact(save_path)

        # Output total training time
        logger.info(
            "\n Total Run Time: {0:0.4f}".format(time.time() - start_time)
        )
        mlflow.log_artifact("/workspace/outputs/main.log")
        return test_acc


if __name__ == "__main__":
    main()
