import argparse
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from aggregations import aggregators
from datasets.augment import transform
from datasets.cifar10 import CIFAR10_CLASSES, Cifar10Dataset
from evaluators.metrics import AverageMeter, accuracy
from models.model import Net
from torch.utils.data import DataLoader
from tqdm import tqdm

from attack.byzantines import bitflip_attack, labelflip_attack, no_byzantine

parser = argparse.ArgumentParser()
parser.add_argument(
    "--batch_size", help="batch size of the workers", type=int, default=100
)
parser.add_argument("--lr", help="learning rate", type=float)
parser.add_argument("--num_workers", help="number of workers", type=int)
parser.add_argument("--num_epochs", help="total number of epochs", type=int)
parser.add_argument("--gpu", help="index of GPU to be used", type=int)
parser.add_argument(
    "--num_byzantines", help="number of faulty workers", type=int
)
parser.add_argument(
    "--byzantine_type", help="type of failures, bitflip or labelflip", type=str
)
parser.add_argument(
    "--aggregation",
    help="aggregation method, mean, median, krum, or zeno",
    type=str,
)
parser.add_argument(
    "--zeno_size",
    help="batch size of Zeno, n_{r} in the paper",
    type=int,
    default=4,
)
parser.add_argument(
    "--rho_ratio",
    help="in the paper, r = frac{gamma}{rho}, ratio to learning rate",
    type=float,
)
parser.add_argument(
    "--num_trimmed_values",
    help="number of trimmed values, in the paper",
    type=int,
)
parser.add_argument(
    "--iid",
    help="-iid 1 means the workers are training on IID data",
    type=int,
    default=1,
)
parser.add_argument("--interval", help="log interval", type=int, default=5)
parser.add_argument("--seed", help="random seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

if args.gpu is not None and args.gpu >= 0:
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")

# byzantine
byzantine_type = (
    bitflip_attack if args.byzantine_type == "bitflip" else no_byzantine
)

zeno_batch_size = args.zeno_size
batch_size = args.batch_size

# iid data or not
if args.iid == 1:
    is_shuffle = True
else:
    is_shuffle = False

# Load CIFAR-10 datasets
cifar10 = Cifar10Dataset(transform=transform)

# Create data loaders
train_loader = DataLoader(
    cifar10.train_dataset,
    batch_size=batch_size,
    shuffle=is_shuffle,
    drop_last=True,
)
val_train_loader = DataLoader(
    cifar10.val_train_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)
val_test_loader = DataLoader(
    cifar10.val_test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)
zeno_data = DataLoader(
    cifar10.zeno_dataset,
    batch_size=zeno_batch_size,
    shuffle=True,
    drop_last=False,
)

zeno_iter = iter(zeno_data)
# zeno_iter = itertools.cycle(zeno_data)


net = Net(CLASSES=len(CIFAR10_CLASSES)).to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(net.parameters(), lr=args.lr / args.batch_size)

# Initialize variables
num_workers = args.num_workers
lr = args.lr / args.batch_size
epochs = args.num_epochs
iteration = 0
grad_list = []
worker_idx = 0
train_start_time = time.time()

# Training loop
for epoch in tqdm(range(args.num_epochs)):
    epoch_start_time = time.time()
    net.train()

    for minibatch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)

        if args.byzantine_type == "label":
            label = labelflip_attack(label, minibatch_idx, args.num_byzantines)

        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        grad_collect = []
        for param in net.parameters():
            if param.requires_grad:
                grad_collect.append(param.grad.clone())
        grad_list.append(grad_collect)

        iteration += 1
        worker_idx += 1

        if iteration % num_workers == 0:
            worker_idx = 0
            if args.aggregation == "median":
                aggregators.marginal_median(
                    grad_list, net, lr, args.num_byzantines, byzantine_type
                )
            elif args.aggregation == "krun":
                aggregators.krum(
                    grad_list, net, lr, args.num_byzantines, byzantine_type
                )
            elif args.aggregation == "mean":
                aggregators.simple_mean(
                    grad_list, net, lr, args.num_byzantines, byzantine_type
                )
            elif args.aggregation == "zeno":
                aggregators.zeno(
                    grad_list,
                    net,
                    criterion,
                    lr,
                    args.num_trimmed_values,
                    args.num_byzantines,
                    byzantine_type,
                )
            else:
                aggregators.simple_mean(
                    grad_list, net, lr, args.num_byzantines, byzantine_type
                )

            del grad_list
            grad_list = []

    epoch_end_time = time.time()

    if epoch % args.interval == 0:
        acc_top1 = AverageMeter("acc_top1")
        acc_top5 = AverageMeter("acc_top5")
        train_cross_entropy = AverageMeter("train_cross_entropy")

        # Accuracy on testing data
        with torch.no_grad():
            for data, label in val_test_loader:
                data, label = data.to(device), label.to(device)
                output = net(data)
                acc1, acc5 = accuracy(output, label, K=(1, 5))
                acc_top1.update(acc1[0], data.size(0))
                acc_top5.update(acc5[0], data.size(0))

        # Cross entropy on training data
        with torch.no_grad():
            for data, label in val_train_loader:
                data, label = data.to(device), label.to(device)
                output = net(data)
                loss = criterion(output, label)
                train_cross_entropy.update(loss.item(), data.size(0))

        print(
            "[Epoch %d] validation: acc-top1=%f acc-top5=%f, \
loss=%f, epoch_time=%f, elapsed=%f"
            % (
                epoch,
                acc_top1.avg,
                acc_top5.avg,
                train_cross_entropy.avg,
                epoch_start_time - epoch_end_time,
                time.time() - train_start_time,
            )
        )
