import torch.nn.functional as F
import torch

def _max(x, target):
    pass


def f1(x, labels):
    x = F.softmax(x, dim=1)
    return 1.0-F.cross_entropy(x, labels)

def f2(x, labels):
    pass


def f4(x, labels):
    x = F.softmax(x, dim=1)
    return torch.clamp(0.5 - F.cross_entropy(x, labels), min=0.0)


def f5(x, labels):
    pass