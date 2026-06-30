"""Small utility functions used by model modules."""

import numpy as np
import torch
from torch import nn
from torch.nn import init as nn_init


def to_f32(x):
    return x.to(dtype=torch.float32)


def to_i32(x):
    return x.to(dtype=torch.int32)


def weight_init_(m, fan_type="in"):
    # RMSNorm: initialize scale to 1.
    if isinstance(m, nn.RMSNorm):
        with torch.no_grad():
            m.weight.fill_(1.0)
        return

    weight = getattr(m, "weight", None)
    if weight is None or weight.numel() == 0:
        return

    # This is a torch private API, but widely used and stable.
    in_num, out_num = nn_init._calculate_fan_in_and_fan_out(weight)

    with torch.no_grad():
        fan = {"avg": (in_num + out_num) / 2, "in": in_num, "out": out_num}[fan_type]
        std = 1.1368 * np.sqrt(1 / fan)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        bias = getattr(m, "bias", None)
        if bias is not None:
            bias.fill_(0.0)


def rpad(x, pad):
    for _ in range(pad):
        x = x.unsqueeze(-1)
    return x
