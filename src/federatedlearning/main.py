#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import pickle
import time
from argparse import Namespace
from typing import Any

import numpy as np
import torch
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
        local_weights, local_losses = [], []
        print(f"\n | Global Training Round : {epoch+1} |\n")

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

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

        # update global weights (FedAVG)
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg: float = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
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
