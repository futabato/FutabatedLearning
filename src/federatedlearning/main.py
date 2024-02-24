#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import os
import pickle
import time

import hydra
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nptyping import Int, NDArray, Shape
from omegaconf import DictConfig
from tensorboardX import SummaryWriter
from tqdm import tqdm

from federatedlearning.aggregations.aggregators import average_weights
from federatedlearning.client.training import LocalUpdate
from federatedlearning.datasets.common import get_dataset
from federatedlearning.models.cnn import CNNCifar, CNNMnist
from federatedlearning.server.inferencing import inference

# Set matplotlib backend to 'Agg'
# to avoid the need for a GUI backend
matplotlib.use("Agg")


@hydra.main(
    version_base="1.1", config_path="/workspace/config", config_name="default"
)
def main(cfg: DictConfig) -> float:
    # Record the start time for run duration
    start_time: float = time.time()

    # Setup paths and logging utilities
    logger: SummaryWriter = SummaryWriter("/workspace/logs")
    mlflow.set_tracking_uri(
        "file://" + hydra.utils.get_original_cwd() + "/mlruns"
    )
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Start an MLFlow run and log the Hydra-generated configuration files
    with mlflow.start_run(run_name=cfg.mlflow.run_name):
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

        # Initialize DataFrame to track client update behaviors during training
        client_behavior_df: list[pd.DataFrame] = [
            pd.DataFrame(columns=["epoch", "local_loss", "local_weight_path"])
            for _ in range(cfg.federatedlearning.num_clients)
        ]
        # Create directories for each client
        # to store their model weights after local updates
        for client_i in range(cfg.federatedlearning.num_clients):
            os.makedirs(
                f"/workspace/outputs/weights/client_{client_i}", exist_ok=True
            )
        # Initialize DataFrame to record global model updates after aggregation
        global_model_record_df: pd.DataFrame = pd.DataFrame(
            columns=["epoch", "global_weight_path"]
        )
        # Create necessary directories for server-side outputs
        os.makedirs("/workspace/outputs/weights/server", exist_ok=True)
        os.makedirs("/workspace/outputs/csv", exist_ok=True)

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
        print(global_model)

        # Capture initial global model weights before training begins
        global_weights: dict[str, torch.Tensor] = global_model.state_dict()

        # Initialize lists to record the training progress
        train_loss: list[float] = []
        train_accuracy: list[float] = []
        # Interval for printing aggregated training stats
        print_every: int = 2

        # Begin federated training loop across specified number of epochs
        for epoch in tqdm(range(cfg.train.epochs)):
            # Collect weights and losses from clients participating in this round
            local_weights: list[dict[str, torch.Tensor]] = []
            local_losses: list[float] = []
            print(f"\n | Global Training Round : {epoch+1} |\n")

            # Re-enter training mode at the start of each epoch
            global_model.train()
            # Randomly select clients to participate in this training round
            num_selected_clients: int = max(
                int(
                    cfg.federatedlearning.frac
                    * cfg.federatedlearning.num_clients
                ),
                1,
            )
            # TODO: prepare flag for client selection
            idxs_clients: NDArray[
                Shape[f"1, {num_selected_clients}"], Int
            ] = np.random.choice(
                range(cfg.federatedlearning.num_clients),
                num_selected_clients,
                replace=False,
            )

            # Loop over each selected client to perform local model updates
            for idx in idxs_clients:
                local_model = LocalUpdate(
                    cfg=cfg,
                    dataset=train_dataset,
                    idxs=client_groups[idx],
                    logger=logger,
                )
                weight, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch
                )
                # Store the updated weights and reported loss
                local_weights.append(copy.deepcopy(weight))
                local_losses.append(copy.deepcopy(loss))
                # Save local model weights and record training details
                torch.save(
                    weight,
                    f"/workspace/outputs/weights/client_{idx}/local_model_epoch_{epoch}.pth",
                )
                local_training_info: dict = {
                    "epoch": epoch,
                    "local_loss": copy.deepcopy(loss),
                    "local_weight_path": f"/workspace/outputs/weights/client_{idx}/local_model_epoch_{epoch}.pth",
                }
                # Append recorded details to the client behavior DataFrame
                client_behavior_df[idx] = pd.concat(
                    [
                        client_behavior_df[idx],
                        pd.DataFrame(local_training_info, index=[epoch]),
                    ],
                    ignore_index=True,
                )
                # Export client behavior data to CSV for analysis or audit
                client_behavior_df[idx].to_csv(
                    f"/workspace/outputs/csv/client_{idx}_behavior.csv",
                    index=False,
                )

            # Aggregate local weights to form new global model weights (FedAVG)
            global_weights = average_weights(local_weights)
            # Save updated global model weights
            torch.save(
                global_weights,
                f"/workspace/outputs/weights/server/global_model_epoch_{epoch}.pth",
            )
            # Load the newly aggregated weights into the global model
            global_model.load_state_dict(global_weights)

            # Record and log the average training loss across all participating clients
            loss_avg: float = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            mlflow.log_metric("Train-Loss", loss_avg, step=epoch)

            # Evaluation phase: compute and log the average training accuracy across all clients
            list_acc: list[float] = []
            list_loss: list[float] = []
            global_model.eval()  # Set model to evaluation mode for inference
            for _ in range(cfg.federatedlearning.num_clients):
                local_model = LocalUpdate(
                    cfg=cfg,
                    dataset=train_dataset,
                    idxs=client_groups[idx],
                    logger=logger,
                )
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)
            train_accuracy.append(sum(list_acc) / len(list_acc))
            mlflow.log_metric(
                "Train-Accuracy", sum(list_acc) / len(list_acc), step=epoch
            )

            # Record details of the global model's state after the epoch
            global_model_info: dict = {
                "epoch": epoch,
                "global_weight_path": f"/workspace/outputs/weights/server/global_model_epoch_{epoch}.pth",
            }
            global_model_record_df = pd.concat(
                [
                    global_model_record_df,
                    pd.DataFrame(global_model_info, index=[epoch]),
                ],
                ignore_index=True,
            )
            # Export server-side global model records to CSV
            global_model_record_df.to_csv(
                "/workspace/outputs/csv/server_record.csv",
                index=False,
            )

            # Occasionally print summary statistics of the training progress
            if (epoch + 1) % print_every == 0:
                print(f" \nAvg Training Stats after {epoch+1} global rounds:")
                print(f"Training Loss : {np.mean(np.array(train_loss))}")
                print(
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
        print(
            f" \n Results after {cfg.train.epochs} global rounds of training:"
        )
        print(
            "|---- Avg Train Accuracy: {:.2f}%".format(
                100 * train_accuracy[-1]
            )
        )
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

        # Save the training loss and accuracy data for future reference
        file_name: str = "{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]".format(
            cfg.train.dataset,
            cfg.train.model,
            cfg.train.epochs,
            cfg.federatedlearning.frac,
            cfg.federatedlearning.iid,
            cfg.train.local_epochs,
            cfg.train.local_batch_size,
        )

        # Open a file to save the train_loss and train_accuracy lists using pickle
        with open(f"/workspace/outputs/objects/{file_name}.pkl", "wb") as f:
            pickle.dump([train_loss, train_accuracy], f)
        # Log the file containing training data as an artifact in MLFlow
        mlflow.log_artifact(f"/workspace/outputs/objects/{file_name}.pkl")

        # Output total training time
        print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))

        # Plot and save the Training Loss vs Communication rounds
        plt.figure()
        plt.title("Training Loss vs Communication rounds")
        plt.plot(range(len(train_loss)), train_loss, color="r")
        plt.ylabel("Training loss")
        plt.xlabel("Communication Rounds")
        plt.savefig(f"/workspace/outputs/objects/fed_{file_name}_loss.png")
        mlflow.log_artifact(
            f"/workspace/outputs/objects/fed_{file_name}_loss.png"
        )

        # Plot and save the Average Accuracy vs Communication rounds
        plt.figure()
        plt.title("Average Accuracy vs Communication rounds")
        plt.plot(range(len(train_accuracy)), train_accuracy, color="k")
        plt.ylabel("Average Accuracy")
        plt.xlabel("Communication Rounds")
        plt.savefig(f"/workspace/outputs/objects/fed_{file_name}_acc.png")
        mlflow.log_artifact(
            f"/workspace/outputs/objects/fed_{file_name}_acc.png"
        )
        return test_acc


if __name__ == "__main__":
    main()
