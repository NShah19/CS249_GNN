import torch

def normal_transform_train(x):
    lamb = 0
    xt_mean = torch.mean(x).float()
    xt_std = torch.std(x).float()
    xt_norm = (x-xt_mean)/xt_std
    return xt_norm,lamb,xt_mean, xt_std

