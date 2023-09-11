import itertools
import random
import time

import hydra
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm import tqdm

from attack.byzantines import (
    bitflip_attack,
    clever_labelflip_attack,
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
def main(cfg: DictConfig):
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

        device = (
            torch.device(f"cuda:{cfg.train.gpu}")
            if cfg.train.gpu is not None and cfg.train.gpu >= 0
            else torch.device("cpu")
        )

        # byzantine
        byzantine_fn = (
            bitflip_attack
            if cfg.federatedlearning.byzantine_type == "bitflip"
            else no_byzantine
        )

        # iid data or not
        is_shuffle = True if cfg.federatedlearning.iid == 1 else False

        # Load CIFAR-10 datasets
        cifar10 = Cifar10Dataset(transform=transform)

        # Create data loaders
        train_loader = DataLoader(
            cifar10.train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=is_shuffle,
            drop_last=True,
        )
        val_train_loader = DataLoader(
            cifar10.val_train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            drop_last=False,
        )
        val_test_loader = DataLoader(
            cifar10.val_test_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            drop_last=False,
        )
        zeno_data = DataLoader(
            cifar10.zeno_dataset,
            batch_size=cfg.federatedlearning.zeno_size,
            shuffle=True,
            drop_last=False,
        )

        zeno_iter = itertools.cycle(zeno_data)

        net = Net(CLASSES=len(CIFAR10_CLASSES)).to(device)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = optim.SGD(
            net.parameters(),
            lr=cfg.train.lr / cfg.train.batch_size,
        )

        # Initialize variables
        lr: float = cfg.train.lr / cfg.train.batch_size
        iteration: int = 0
        grad_list: list = []
        worker_idx: int = 0
        train_start_time: float = time.time()

        # Training loop
        for epoch in tqdm(range(cfg.train.num_epochs)):
            epoch_start_time = time.time()
            net.train()

            for _, (data, label) in enumerate(train_loader):
                data, label = data.to(device), label.to(device)
                if (
                    cfg.federatedlearning.byzantine_type == "labelflip"
                    and worker_idx < cfg.federatedlearning.num_byzantines
                ):
                    label = labelflip_attack(label)
                elif (
                    cfg.federatedlearning.byzantine_type == "clever-labelflip"
                    and worker_idx < cfg.federatedlearning.num_byzantines
                ):
                    label = clever_labelflip_attack(label)
                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                grad_collect: list = []
                for param in net.parameters():
                    if param.requires_grad:
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
                    elif cfg.federatedlearning.aggregation == "krun":
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

            train_cross_entropy = AverageMeter("Train-Cross-Entropy")
            acc1 = MulticlassAccuracy(
                average="macro", num_classes=len(CIFAR10_CLASSES)
            )
            acc5 = MulticlassAccuracy(
                average="macro",
                num_classes=len(CIFAR10_CLASSES),
                k=5,
            )
            precision = MulticlassPrecision(
                average="macro", num_classes=len(CIFAR10_CLASSES)
            )
            recall = MulticlassRecall(
                average="macro", num_classes=len(CIFAR10_CLASSES)
            )
            f1score = MulticlassF1Score(
                average="macro", num_classes=len(CIFAR10_CLASSES)
            )
            # Accuracy on testing data
            with torch.no_grad():
                for data, label in val_test_loader:
                    data, label = data.to(device), label.to(device)
                    output = net(data)
                    acc1.update(output, label)
                    acc5.update(output, label)
                    precision.update(output, label)
                    recall.update(output, label)
                    f1score.update(output, label)

            # Cross entropy on training data
            with torch.no_grad():
                for data, label in val_train_loader:
                    data, label = data.to(device), label.to(device)
                    output = net(data)
                    loss = criterion(output, label)
                    train_cross_entropy.update(loss.item(), data.size(0))

            mlflow.log_metric("Accuracy-Top1", acc1.compute(), step=epoch)
            mlflow.log_metric("Accuracy-Top5", acc5.compute(), step=epoch)
            mlflow.log_metric("Precision", precision.compute(), step=epoch)
            mlflow.log_metric("Recall", recall.compute(), step=epoch)
            mlflow.log_metric("F1Score", f1score.compute(), step=epoch)
            mlflow.log_metric("Train-Cross-Entropy", loss, step=epoch)

            if (
                epoch % cfg.train.interval == 0
                or epoch == cfg.train.num_epochs - 1
            ):
                print(
                    "[Epoch %d] validation: Acc-top1=%f Acc-top5=%f, \
trainloss=%f, epoch_time=%f, elapsed=%f\n                      \
Precison=%f, Recall=%f, F1Score=%f"
                    % (
                        epoch,
                        acc1.compute(),
                        acc5.compute(),
                        train_cross_entropy.avg,
                        epoch_end_time - epoch_start_time,
                        time.time() - train_start_time,
                        precision.compute(),
                        recall.compute(),
                        f1score.compute(),
                    )
                )


if __name__ == "__main__":
    main()
