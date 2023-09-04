class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, K=(1,)):
    max_k = max(K)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in K:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class TopKAccuracy:
    def __init__(self, k=1):
        self.k = k
        self.reset()

    def reset(self):
        self.topk_correct = [0] * self.k
        self.total = 0

    def update(self, labels, outputs):
        topk_vals, topk_inds = outputs.topk(self.k, 1, True, True)
        topk_correct = topk_inds.eq(labels.view(-1, 1).expand_as(topk_inds))
        for i in range(self.k):
            self.topk_correct[i] += topk_correct[:, i].sum().item()
        self.total += labels.size(0)

    def get(self):
        topk_acc = [
            correct * 100.0 / self.total for correct in self.topk_correct
        ]
        return topk_acc


class CrossEntropy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0
        self.num_samples = 0

    def update(self, labels, outputs):
        self.total_loss += outputs.sum().item()
        self.num_samples += labels.size(0)

    def get(self):
        return self.total_loss / self.num_samples
