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
