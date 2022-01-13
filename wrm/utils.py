import torch
import numpy as np

def l1proximal(x, _lambda):
    return torch.sign(x) * torch.maximum( torch.abs(x) - _lambda, torch.zeros_like(x) )

def l2proximal(x, _lambda):
    return torch.maximum(1.0 - _lambda/torch.norm(x) ,torch.zeros_like(torch.norm(x))) * x

def get_indices(batch_size, length):
    seq = np.arange(length)
    np.random.shuffle(seq)
    return np.split(seq, length // batch_size)