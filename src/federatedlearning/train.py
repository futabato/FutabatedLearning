import itertools
import random
import time
from typing import Callable

import hydra
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm import tqdm

from attack.byzantines import (
    bitflip_attack,
    chosen_labelflip_attack,
    labelflip_attack,
    no_byzantine,
)
from federatedlearning.aggregations import aggregators
from federatedlearning.datasets.augment import transform
from federatedlearning.datasets.cifar10 import CIFAR10_CLASSES, Cifar10Dataset
from federatedlearning.evaluations.metrics import AverageMeter
from federatedlearning.models.model import Net


@hydra.main(
    version_base="1.1",
    config_path="/workspace/config",
    config_name="default",
)
def main(cfg: DictConfig) -> MulticlassAccuracy:
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

        torch.manual_seed(cfg.train.seed)
        random.seed(cfg.train.seed)
        torch.backends.cudnn.deterministic = True

        device: torch.device = (
            torch.device(f"cuda:{cfg.train.gpu}")
            if cfg.train.gpu is not None and cfg.train.gpu >= 0
            else torch.device("cpu")
        )

        # byzantine
        byzantine_fn: Callable = (
            bitflip_attack
            if cfg.federatedlearning.byzantine_type == "bitflip"
            else no_byzantine
        )

        # iid data or not
        is_shuffle: bool = True if cfg.federatedlearning.iid == 1 else False

        # Load CIFAR-10 datasets
        cifar10: Cifar10Dataset = Cifar10Dataset(transform)

        # Create data loaders
        train_loader: DataLoader = DataLoader(
            cifar10.train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=is_shuffle,
            drop_last=True,
        )
        val_loader: DataLoader = DataLoader(
            cifar10.val_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            drop_last=False,
        )
        zeno_data: DataLoader = DataLoader(
            cifar10.zeno_dataset,
            batch_size=cfg.federatedlearning.zeno_size,
            shuffle=True,
            drop_last=False,
        )

        zeno_iter: itertools.cycle[
            tuple[torch.Tensor, torch.Tensor]
        ] = itertools.cycle(zeno_data)

        net = Net(CLASSES=len(CIFAR10_CLASSES)).to(device)

        # Loss function
        criterion: CrossEntropyLoss = nn.CrossEntropyLoss()

        # Optimizer
        optimizer: SGD = optim.SGD(
            net.parameters(),
            lr=cfg.train.lr,
        )

        # Initialize variables
        lr: float = cfg.train.lr / cfg.train.batch_size
        iteration: int = 0
        grad_list: list[list[torch.Tensor]] = []
        worker_idx: int = 0
        train_start_time: float = time.time()
        train_cross_entropy: AverageMeter = AverageMeter("Train-Cross-Entropy")
        accuracy: MulticlassAccuracy = MulticlassAccuracy(
            average="macro", num_classes=len(CIFAR10_CLASSES)
        )
        test_accuracy: MulticlassAccuracy = MulticlassAccuracy(
            average="macro", num_classes=len(CIFAR10_CLASSES)
        )
        test_precision: MulticlassPrecision = MulticlassPrecision(
            average="macro", num_classes=len(CIFAR10_CLASSES)
        )
        test_recall: MulticlassRecall = MulticlassRecall(
            average="macro", num_classes=len(CIFAR10_CLASSES)
        )
        test_f1score: MulticlassF1Score = MulticlassF1Score(
            average="macro", num_classes=len(CIFAR10_CLASSES)
        )
        confusion_matrix: MulticlassConfusionMatrix = (
            MulticlassConfusionMatrix(len(CIFAR10_CLASSES))
        )

        # Training loop
        for epoch in tqdm(range(1, cfg.train.num_epochs + 1)):
            epoch_start_time: float = time.time()
            net.train()

            for data, label in train_loader:
                data, label = data.to(device), label.to(device)
                if (
                    cfg.federatedlearning.byzantine_type == "labelflip"
                    and worker_idx < cfg.federatedlearning.num_byzantines
                ):
                    label = labelflip_attack(label)
                elif (
                    cfg.federatedlearning.byzantine_type == "chosen-labelflip"
                    and worker_idx < cfg.federatedlearning.num_byzantines
                ):
                    label = chosen_labelflip_attack(
                        label,
                        cfg.federatedlearning.choice_src_label,
                        cfg.federatedlearning.choice_dst_label,
                    )
                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output, label)
                train_cross_entropy.update(loss.item(), data.size(0))
                mlflow.log_metric("Train-Cross-Entropy", loss, step=epoch)
                loss.backward()
                optimizer.step()

                grad_collect: list[torch.Tensor] = []
                with torch.no_grad():
                    for param in net.parameters():
                        grad_collect.append(param.grad.clone())
                grad_list.append(grad_collect)

                iteration += 1
                worker_idx += 1

                if iteration % cfg.federatedlearning.num_workers == 0:
                    worker_idx = 0
                    if cfg.federatedlearning.aggregation == "median":
                        aggregators.marginal_median(
                            grad_list,
                            net,
                            lr,
                            cfg.federatedlearning.num_byzantines,
                            byzantine_fn,
                        )
                    elif cfg.federatedlearning.aggregation == "krum":
                        aggregators.krum(
                            grad_list,
                            net,
                            lr,
                            cfg.federatedlearning.num_byzantines,
                            byzantine_fn,
                        )
                    elif cfg.federatedlearning.aggregation == "mean":
                        aggregators.simple_mean(
                            grad_list,
                            net,
                            lr,
                            cfg.federatedlearning.num_byzantines,
                            byzantine_fn,
                        )
                    elif cfg.federatedlearning.aggregation == "zeno":
                        zeno_sample = next(zeno_iter)
                        aggregators.zeno(
                            grad_list,
                            net,
                            criterion,
                            lr,
                            zeno_sample,
                            cfg.federatedlearning.rho_ratio,
                            cfg.federatedlearning.num_trimmed_values,
                            device,
                            cfg.federatedlearning.num_byzantines,
                            byzantine_fn,
                        )
                    else:
                        aggregators.simple_mean(
                            grad_list,
                            net,
                            lr,
                            cfg.federatedlearning.num_byzantines,
                            byzantine_fn,
                        )

                    grad_list.clear()

            epoch_end_time: float = time.time()

            # Accuracy on testing data
            with torch.no_grad():
                for data, label in val_loader:
                    data, label = data.to(device), label.to(device)
                    output = net(data)
                    accuracy.update(output, label)
            mlflow.log_metric("Accuracy-Top1", accuracy.compute(), step=epoch)

            if (
                epoch % cfg.train.interval == 0
                or epoch == cfg.train.num_epochs
            ):
                print(
                    "\n[Epoch %d] train-loss=%f, epoch_time=%f, elapsed=%f\n \
\t  validation: Accuracy=%f"
                    % (
                        epoch,
                        train_cross_entropy.avg,
                        epoch_end_time - epoch_start_time,
                        time.time() - train_start_time,
                        accuracy.compute(),
                    )
                )
        # Accuracy on testing data
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                output = net(data)
                test_accuracy.update(output, label)
                test_precision.update(output, label)
                test_recall.update(output, label)
                test_f1score.update(output, label)
                confusion_matrix.update(output, label)

        plt.figure(figsize=(10, 8))
        cm: torch.Tensor = confusion_matrix.compute()
        df_cm = pd.DataFrame(
            cm, columns=CIFAR10_CLASSES, index=CIFAR10_CLASSES
        )
        df_cm.columns.name = "Predicted"
        df_cm.index.name = "Actual"
        sns.heatmap(
            df_cm,
            square=True,
            cbar=True,
            annot=True,
            cmap="Blues",
        )
        plt.savefig("confusion_matrix.png")

        mlflow.log_metric("Test-Accuracy-Top1", test_accuracy.compute())
        mlflow.log_metric("Test-Precision", test_precision.compute())
        mlflow.log_metric("Test-Recall", test_recall.compute())
        mlflow.log_metric("Test-F1Score", test_f1score.compute())
        mlflow.log_artifact("confusion_matrix.png")
    return test_accuracy


if __name__ == "__main__":
    main()
