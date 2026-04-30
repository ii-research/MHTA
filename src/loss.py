import torch

def emd_loss(pred, target):
    pred_cdf = torch.cumsum(pred, dim=-1)
    target_cdf = torch.cumsum(target, dim=-1)
    emd = torch.abs(pred_cdf - target_cdf).mean(dim=-1)
    return emd.mean()