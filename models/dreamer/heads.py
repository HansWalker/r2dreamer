"""Reward and continuation heads for Dreamer."""

from __future__ import annotations

import torch
from torch import nn

from models.shared import distributions as dists
from models.shared.utils import weight_init_


class _DreamerHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config,
        *,
        output_scale: float,
    ):
        super().__init__()
        hidden_dim = int(config.hidden_dim)
        activation = getattr(nn, str(config.act))
        modules = []
        for _ in range(int(config.layers)):
            modules.extend([
                nn.Linear(int(input_dim), int(hidden_dim), bias=True),
                nn.RMSNorm(int(hidden_dim), eps=1e-4, dtype=torch.float32),
                activation(),
            ])
            input_dim = int(hidden_dim)
        self.layers = nn.Sequential(*modules)
        self.output = nn.Linear(int(input_dim), int(output_dim), bias=True)
        self.apply(weight_init_)
        with torch.no_grad():
            self.output.weight.mul_(float(output_scale))

    def logits(self, feature: torch.Tensor) -> torch.Tensor:
        return self.output(self.layers(feature))


class DreamerRewardHead(_DreamerHead):
    def __init__(self, input_dim: int, config):
        super().__init__(
            input_dim,
            int(config.reward_bins),
            config,
            output_scale=0.0,
        )
        self.reward_bins = int(config.reward_bins)

    def forward(self, feature: torch.Tensor):
        return dists.symexp_twohot(self.logits(feature), bin_num=self.reward_bins)


class DreamerContinuationHead(_DreamerHead):
    def __init__(self, input_dim: int, config):
        super().__init__(
            input_dim,
            1,
            config,
            output_scale=1.0,
        )

    def forward(self, feature: torch.Tensor):
        return dists.binary(self.logits(feature))
