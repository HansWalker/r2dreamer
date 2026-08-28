"""Reward and termination heads for STORM."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .objectives import SymLogTwoHotLoss


class _StormHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, layers: int, act: str):
        super().__init__()
        activation = getattr(nn, act)
        modules = []
        for _ in range(int(layers)):
            modules.extend([
                nn.Linear(int(input_dim), int(hidden_dim), bias=False),
                nn.LayerNorm(int(hidden_dim)),
                activation(inplace=True) if activation is nn.ReLU else activation(),
            ])
            input_dim = int(hidden_dim)
        modules.append(nn.Linear(int(input_dim), int(output_dim), bias=True))
        self.layers = nn.Sequential(*modules)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.layers(feature)


class StormRewardHead(_StormHead):
    def __init__(self, input_dim: int, config):
        heads = config.heads
        super().__init__(input_dim, heads.reward_bins, heads.hidden_dim, heads.layers, str(config.act))
        self.objective = SymLogTwoHotLoss(int(heads.reward_bins), -20, 20)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.objective(logits, target)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        return self.objective.decode(logits)


class StormTerminationHead(_StormHead):
    def __init__(self, input_dim: int, config):
        heads = config.heads
        super().__init__(input_dim, 1, heads.hidden_dim, heads.layers, str(config.act))

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits.squeeze(-1),
            target.to(logits.dtype).squeeze(-1),
        )

    @staticmethod
    def probability(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)
