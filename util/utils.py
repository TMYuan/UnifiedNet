import torch
import torch.nn as nn

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    std = batch.new_tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    out = (batch - mean) / std
    return torch.clamp(out, -1, 1)

def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)