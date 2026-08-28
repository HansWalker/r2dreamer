"""STORM dynamics prior for discrete stochastic states."""

from __future__ import annotations

import torch
from torch import nn

from .posterior import unimix_logits


class StormPrior(nn.Linear):
    """Map STORM's deterministic state to categorical prior logits."""

    def __init__(self, deter_dim: int, stoch_dim: int, unimix_ratio: float = 0.01):
        stoch_dim = int(stoch_dim)
        super().__init__(int(deter_dim), stoch_dim * stoch_dim)
        self.stoch_dim = stoch_dim
        self.unimix_ratio = float(unimix_ratio)

    def forward(self, deter: torch.Tensor) -> torch.Tensor:
        logits = super().forward(deter)
        logits = logits.reshape(*logits.shape[:-1], self.stoch_dim, self.stoch_dim)
        return unimix_logits(logits, self.unimix_ratio)
