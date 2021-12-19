import torch

def l1proximal(x, _lambda):
    return torch.sign(x) * torch.maximum( torch.abs(x) - _lambda, torch.zeros_like(x) )

def l2proximal(x, _lambda):
    return torch.maximum(1.0 - _lambda/torch.norm(x) ,torch.zeros_like(torch.norm(x))) * x
