import torch


def no_byzantine(v, f):
    """no faulty workers

    Args:
        v (_type_): _description_
        f (_type_): _description_
    """
    return v


def gaussian_attack(v, f):
    """failures that add Gaussian noise

    Args:
        v (_type_): _description_
        f (_type_): _description_
    """
    for i in range(f):
        v[i] = torch.randn(v[i].size()) * 200


def bitflip_attack(v, f):
    """bit-flipping failure

    Args:
        v (_type_): _description_
        f (_type_): _description_
    """
    for i in range(f):
        if i > 0:
            v[i][:] = -v[0]
    v[0][:] = -v[0]


def labelflip_attack(label, minibatch_idx, num_byzantines):
    if minibatch_idx < num_byzantines:
        return 9 - label
    return label
