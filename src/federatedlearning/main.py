#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import os
import pickle
import time
from argparse import Namespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from nptyping import Int, NDArray, Shape
from tensorboardX import SummaryWriter
from tqdm import tqdm

from federatedlearning.aggregations.aggregators import average_weights
from federatedlearning.client.training import LocalUpdate
from federatedlearning.datasets.common import get_dataset
from federatedlearning.models.cnn import CNNCifar, CNNMnist
from federatedlearning.options import args_parser
from federatedlearning.server.inferencing import inference

if __name__ == "__main__":
    start_time: float = time.time()

    # define paths
    logger: SummaryWriter = SummaryWriter("/workspace/logs")

    args: Namespace = args_parser()

    device: torch.device = (
        torch.device(f"cuda:{args.gpu}")
        if args.gpu is not None and int(args.gpu) >= 0
        else torch.device("cpu")
    )

    # Empty DataFrame for tracking client behaviors
    client_behavior_df: list[pd.DataFrame] = [
        pd.DataFrame(columns=["epoch", "local_loss", "local_weight_path"])
        for _ in range(args.num_users)
    ]
    for i in range(args.num_users):
        os.makedirs(f"/workspace/outputs/weights/client_{i}", exist_ok=True)
    # Empty DataFrame for recoding global model
    global_model_record_df: pd.DataFrame = pd.DataFrame(
        columns=["epoch", "global_weight_path"]
    )
    os.makedirs("/workspace/outputs/weights/server", exist_ok=True)
    os.makedirs("/workspace/outputs/csv", exist_ok=True)

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    global_model: Any
    if args.dataset == "mnist":
        global_model = CNNMnist(args=args)
    elif args.dataset == "cifar":
        global_model = CNNCifar(args=args)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights: dict[str, Any] = global_model.state_dict()

    # Training
    train_loss: list[float] = []
    train_accuracy: list[float] = []
    val_acc_list: list[float] = []
    net_list: list[float] = []
    cv_loss: list[float] = []
    cv_acc: list[float] = []
    print_every: int = 2
    val_loss_pre: int = 0
    counter: int = 0

    for epoch in tqdm(range(args.epochs)):
        local_weights: list[float] = []
        local_losses: list[float] = []
        print(f"\n | Global Training Round : {epoch+1} |\n")

        global_model.train()
        m: int = max(int(args.frac * args.num_users), 1)
        idxs_users: NDArray[Shape[f"1, {m}"], Int] = np.random.choice(
            range(args.num_users), m, replace=False
        )

        for idx in idxs_users:
            local_model = LocalUpdate(
                args=args,
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

        # Calculate avg training accuracy over all users at every epoch
        list_acc: list[float] = []
        list_loss: list[float] = []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(
                args=args,
                dataset=train_dataset,
                idxs=user_groups[idx],
                logger=logger,
            )
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

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
                "Train Accuracy: {:.2f}% \n".format(100 * train_accuracy[-1])
            )

    # Test inference after completion of training
    test_acc, test_loss = inference(args, global_model, test_dataset)

    print(f" \n Results after {args.epochs} global rounds of training:")
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name: str = "/workspace/outputs/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl".format(
        args.dataset,
        args.model,
        args.epochs,
        args.frac,
        args.iid,
        args.local_ep,
        args.local_bs,
    )

    with open(file_name, "wb") as f:
        pickle.dump([train_loss, train_accuracy], f)

    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
