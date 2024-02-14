#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import os
import pickle
import time
from typing import Any

import hydra
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from nptyping import Int, NDArray, Shape
from omegaconf import DictConfig
from tensorboardX import SummaryWriter
from tqdm import tqdm

from federatedlearning.aggregations.aggregators import average_weights
from federatedlearning.client.training import LocalUpdate
from federatedlearning.datasets.common import get_dataset
from federatedlearning.models.cnn import CNNCifar, CNNMnist
from federatedlearning.server.inferencing import inference

matplotlib.use("Agg")


@hydra.main(
    version_base="1.1", config_path="/workspace/config", config_name="default"
)
def main(cfg: DictConfig) -> None:
    start_time: float = time.time()

    # define paths
    logger: SummaryWriter = SummaryWriter("/workspace/logs")
    mlflow.set_tracking_uri(
        "file://" + hydra.utils.get_original_cwd() + "/mlruns"
    )
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        mlflow.log_artifact("/workspace/outputs/.hydra/config.yaml")
        mlflow.log_artifact("/workspace/outputs/.hydra/hydra.yaml")
        mlflow.log_artifact("/workspace/outputs/.hydra/overrides.yaml")

        mlflow.log_params(cfg.federatedlearning)
        mlflow.log_params(cfg.train)

        device: torch.device = (
            torch.device(f"cuda:{cfg.train.gpu}")
            if cfg.train.gpu is not None and cfg.train.gpu >= 0
            else torch.device("cpu")
        )

        # Empty DataFrame for tracking client behaviors
        client_behavior_df: list[pd.DataFrame] = [
            pd.DataFrame(columns=["epoch", "local_loss", "local_weight_path"])
            for _ in range(cfg.federatedlearning.num_users)
        ]
        for i in range(cfg.federatedlearning.num_users):
            os.makedirs(
                f"/workspace/outputs/weights/client_{i}", exist_ok=True
            )
        # Empty DataFrame for recoding global model
        global_model_record_df: pd.DataFrame = pd.DataFrame(
            columns=["epoch", "global_weight_path"]
        )
        os.makedirs("/workspace/outputs/weights/server", exist_ok=True)
        os.makedirs("/workspace/outputs/csv", exist_ok=True)

        # load dataset and user groups
        train_dataset, test_dataset, user_groups = get_dataset(cfg)

        # BUILD MODEL
        global_model: Any
        if cfg.train.dataset == "mnist":
            global_model = CNNMnist(cfg=cfg)
        elif cfg.train.dataset == "cifar":
            global_model = CNNCifar(cfg=cfg)

        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()
        print(global_model)

        # copy weights
        global_weights: dict[str, torch.Tensor] = global_model.state_dict()

        # Training
        train_loss: list[float] = []
        train_accuracy: list[float] = []
        print_every: int = 2

        for epoch in tqdm(range(cfg.train.epochs)):
            local_weights: list[dict[str, torch.Tensor]] = []
            local_losses: list[float] = []
            print(f"\n | Global Training Round : {epoch+1} |\n")

            global_model.train()
            m: int = max(
                int(
                    cfg.federatedlearning.frac
                    * cfg.federatedlearning.num_users
                ),
                1,
            )
            idxs_users: NDArray[Shape[f"1, {m}"], Int] = np.random.choice(
                range(cfg.federatedlearning.num_users), m, replace=False
            )

            for idx in idxs_users:
                local_model = LocalUpdate(
                    cfg=cfg,
                    dataset=train_dataset,
                    idxs=user_groups[idx],
                    logger=logger,
                )
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch
                )
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
                # Preparation for storing local training information in the DataFrame for attestedFL
                torch.save(
                    w,
                    f"/workspace/outputs/weights/client_{idx}/local_model_epoch_{epoch}.pth",
                )
                local_training_info: dict = {
                    "epoch": epoch,
                    "local_loss": copy.deepcopy(loss),
                    "local_weight_path": f"/workspace/outputs/weights/client_{idx}/local_model_epoch_{epoch}.pth",
                }
                # Store local training information
                client_behavior_df[idx] = pd.concat(
                    [
                        client_behavior_df[idx],
                        pd.DataFrame(local_training_info, index=[epoch]),
                    ],
                    ignore_index=True,
                )
                client_behavior_df[idx].to_csv(
                    f"/workspace/outputs/csv/client_{idx}_behavior.csv",
                    index=False,
                )

            # update global weights (FedAVG)
            global_weights = average_weights(local_weights)
            torch.save(
                global_weights,
                f"/workspace/outputs/weights/server/global_model_epoch_{epoch}.pth",
            )
            # update global weights
            global_model.load_state_dict(global_weights)

            loss_avg: float = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            mlflow.log_metric("Train-Loss", loss_avg, step=epoch)

            # Calculate avg training accuracy over all users at every epoch
            list_acc: list[float] = []
            list_loss: list[float] = []
            global_model.eval()
            for _ in range(cfg.federatedlearning.num_users):
                local_model = LocalUpdate(
                    cfg=cfg,
                    dataset=train_dataset,
                    idxs=user_groups[idx],
                    logger=logger,
                )
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)
            train_accuracy.append(sum(list_acc) / len(list_acc))
            mlflow.log_metric(
                "Train-Accuracy", sum(list_acc) / len(list_acc), step=epoch
            )

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
            global_model_record_df.to_csv(
                "/workspace/outputs/csv/server_record.csv",
                index=False,
            )

            # print global training loss after every 'i' rounds
            if (epoch + 1) % print_every == 0:
                print(f" \nAvg Training Stats after {epoch+1} global rounds:")
                print(f"Training Loss : {np.mean(np.array(train_loss))}")
                print(
                    "Train Accuracy: {:.2f}% \n".format(
                        100 * train_accuracy[-1]
                    )
                )

        # Test inference after completion of training
        test_acc, test_loss = inference(cfg, global_model, test_dataset)
        mlflow.log_metric("Test-Accuracy", test_acc)
        mlflow.log_metric("Test-Loss", test_loss)

        print(
            f" \n Results after {cfg.train.epochs} global rounds of training:"
        )
        print(
            "|---- Avg Train Accuracy: {:.2f}%".format(
                100 * train_accuracy[-1]
            )
        )
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

        # Saving the objects train_loss and train_accuracy:
        file_name: str = "{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]".format(
            cfg.train.dataset,
            cfg.train.model,
            cfg.train.epochs,
            cfg.federatedlearning.frac,
            cfg.federatedlearning.iid,
            cfg.train.local_ep,
            cfg.train.local_bs,
        )

        with open(f"/workspace/outputs/objects/{file_name}.pkl", "wb") as f:
            pickle.dump([train_loss, train_accuracy], f)
        mlflow.log_artifact(f"/workspace/outputs/objects/{file_name}.pkl")

        print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))

        # Plot Loss curve
        plt.figure()
        plt.title("Training Loss vs Communication rounds")
        plt.plot(range(len(train_loss)), train_loss, color="r")
        plt.ylabel("Training loss")
        plt.xlabel("Communication Rounds")
        plt.savefig(f"/workspace/outputs/objects/fed_{file_name}_loss.png")
        mlflow.log_artifact(
            f"/workspace/outputs/objects/fed_{file_name}_loss.png"
        )

        # Plot Average Accuracy vs Communication rounds
        plt.figure()
        plt.title("Average Accuracy vs Communication rounds")
        plt.plot(range(len(train_accuracy)), train_accuracy, color="k")
        plt.ylabel("Average Accuracy")
        plt.xlabel("Communication Rounds")
        plt.savefig(f"/workspace/outputs/objects/fed_{file_name}_acc.png")
        mlflow.log_artifact(
            f"/workspace/outputs/objects/fed_{file_name}_acc.png"
        )


if __name__ == "__main__":
    main()
