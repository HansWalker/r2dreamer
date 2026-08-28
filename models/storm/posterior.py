"""STORM observation posterior for discrete stochastic states."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def unimix_logits(logits: torch.Tensor, mixing_ratio: float = 0.01) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    uniform = torch.ones_like(probs) / logits.shape[-1]
    return torch.log(mixing_ratio * uniform + (1 - mixing_ratio) * probs)


class StormPosterior(nn.Linear):
    """Produce STORM's observation-only categorical posterior."""

    def __init__(self, encoder_feat_dim: int, stoch_dim: int, unimix_ratio: float = 0.01):
        stoch_dim = int(stoch_dim)
        super().__init__(int(encoder_feat_dim), stoch_dim * stoch_dim)
        self.stoch_dim = stoch_dim
        self.unimix_ratio = float(unimix_ratio)

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        logits = super().forward(embed)
        logits = logits.reshape(*logits.shape[:-1], self.stoch_dim, self.stoch_dim)
        return unimix_logits(logits, self.unimix_ratio)
