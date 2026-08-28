"""Dreamer posterior for discrete stochastic states."""

from __future__ import annotations

import torch
from torch import nn


class DreamerPosterior(nn.Sequential):
    """Map the deterministic state and observation embedding to posterior logits."""

    def __init__(self, config, deter_dim: int, embed_dim: int):
        super().__init__()
        posterior = config.posterior
        activation = getattr(nn, str(posterior.act))
        hidden_dim = int(posterior.hidden_dim)
        input_dim = int(deter_dim) + int(embed_dim)
        for index in range(int(posterior.layers)):
            self.add_module(f"obs_net_{index}", nn.Linear(input_dim, int(hidden_dim), bias=True))
            self.add_module(
                f"obs_net_n_{index}",
                nn.RMSNorm(int(hidden_dim), eps=1e-4, dtype=torch.float32),
            )
            self.add_module(f"obs_net_a_{index}", activation())
            input_dim = int(hidden_dim)
        self.add_module(
            "obs_net_logit",
            nn.Linear(input_dim, int(posterior.stoch_dim) * int(posterior.discrete_dim), bias=True),
        )
        self.stoch_dim = int(posterior.stoch_dim)
        self.discrete_dim = int(posterior.discrete_dim)

    def forward(self, deter: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        logits = super().forward(torch.cat([deter, embed], dim=-1))
        return logits.reshape(*logits.shape[:-1], self.stoch_dim, self.discrete_dim)
