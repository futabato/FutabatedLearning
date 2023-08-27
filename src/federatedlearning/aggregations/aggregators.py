import numpy as np
import torch

from .scores import calc_sum_distances


def no_byzantine(v, f):
    pass


def marginal_median(gradients, net, lr, f=0, byz=no_byzantine):
    # X is a 2d list of nd array
    param_list = [
        torch.cat([xx.view(-1, 1) for xx in x], dim=0) for x in gradients
    ]
    byz(param_list, f)
    sorted_array = torch.sort(torch.cat(param_list, dim=1), dim=-1)[0]
    if sorted_array.shape[-1] % 2 == 1:
        median_nd = sorted_array[..., sorted_array.shape[-1] // 2]
    else:
        median_nd = (
            sorted_array[..., sorted_array.shape[-1] // 2 - 1]
            + sorted_array[..., sorted_array.shape[-1] // 2]
        ) / 2.0

    idx = 0
    for _, (param) in enumerate(net.parameters()):
        if param.requires_grad:
            param.data -= lr * median_nd[
                idx : (idx + param.data.numel())
            ].view(param.data.shape)
            idx += param.data.numel()


def simple_mean(gradients, net, lr, f=0, byz=no_byzantine):
    # X is a 2d list of nd array
    param_list = [
        torch.cat([xx.view(-1, 1) for xx in x], dim=0) for x in gradients
    ]
    byz(param_list, f)
    mean_nd = torch.mean(torch.cat(param_list, dim=1), dim=-1)

    idx = 0
    for _, (param) in enumerate(net.parameters()):
        if param.requires_grad:
            param.data -= lr * mean_nd[idx : (idx + param.data.numel())].view(
                param.data.shape
            )
            idx += param.data.numel()


def krum(gradients, net, lr, f=0, byz=no_byzantine):
    # X is a 2d list of nd array
    param_list = [
        torch.cat([xx.view(-1, 1) for xx in x], dim=0) for x in gradients
    ]
    byz(param_list, f)
    v = torch.cat(param_list, dim=1)

    scores = torch.tensor(
        [calc_sum_distances(gradient, v, f) for gradient in param_list]
    )
    min_idx = scores.argmin().item()
    krum_nd = param_list[min_idx].view(-1)

    idx = 0
    for j, (param) in enumerate(net.parameters()):
        if param.requires_grad:
            param.data -= lr * krum_nd[idx : (idx + param.data.numel())].view(
                param.data.shape
            )
            idx += param.data.numel()


def zeno(
    gradients, net, loss_fun, lr, sample, rho_ratio, b, f=0, byz=no_byzantine
):
    # X is a 2d list of nd array
    param_list = [
        torch.cat([xx.view(-1, 1) for xx in x], dim=0) for x in gradients
    ]

    param_net = [xx.data.clone() for xx in net.parameters()]

    byz(param_list, f)
    output = net(sample[0])
    loss_1_nd = loss_fun(output, sample[1])
    loss_1 = loss_1_nd.mean().item()

    scores = []
    rho = lr / rho_ratio
    for i in range(len(param_list)):
        idx = 0
        for j, param in enumerate(net.parameters()):
            if param.requires_grad:
                param.data = param_net[j] - lr * param_list[i][
                    idx : (idx + param.data.numel())
                ].view(param.data.shape)
                idx += param.data.numel()
        output = net(sample[0])
        loss_2_nd = loss_fun(output, sample[1])
        loss_2 = loss_2_nd.mean().item()
        scores.append(
            loss_1 - loss_2 - rho * param_list[i].square().mean().item()
        )

    scores_np = np.array(scores)
    scores_idx = np.argsort(scores_np)
    scores_idx = scores_idx[-(len(param_list) - b) :].tolist()
    g_aggregated = torch.zeros_like(param_list[0])

    for idx in scores_idx:
        g_aggregated += param_list[idx]
    g_aggregated /= float(len(scores_idx))

    idx = 0
    for j, (param) in enumerate(net.parameters()):
        if param.requires_grad:
            param.data = param_net[j] - lr * g_aggregated[
                idx : (idx + param.data.numel())
            ].view(param.data.shape)
            idx += param.data.numel()
