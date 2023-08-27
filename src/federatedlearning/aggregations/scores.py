import math

import torch


def calc_sum_distances(gradient, v, f):
    if 2 * f + 2 > v.shape[1]:
        f = int(math.floor((v.shape[1] - 2) / 2.0))
    num_neighbours = v.shape[1] - 2 - f
    sorted_distance = torch.sum((v - gradient) ** 2, dim=0).sort()[0]
    return torch.sum(sorted_distance[1 : (1 + num_neighbours)]).item()
