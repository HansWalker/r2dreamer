"""Dreamer prior for discrete stochastic states."""

from __future__ import annotations

import torch
from torch import nn


class DreamerPrior(nn.Sequential):
    """Map Dreamer's deterministic state to categorical prior logits."""

    def __init__(self, config, deter_dim: int):
        super().__init__()
        posterior = config.posterior
        prior = config.prior
        activation = getattr(nn, str(prior.act))
        hidden_dim = int(prior.hidden_dim)
        input_dim = int(deter_dim)
        for index in range(int(prior.layers)):
            self.add_module(f"img_net_{index}", nn.Linear(input_dim, int(hidden_dim), bias=True))
            self.add_module(
                f"img_net_n_{index}",
                nn.RMSNorm(int(hidden_dim), eps=1e-4, dtype=torch.float32),
            )
            self.add_module(f"img_net_a_{index}", activation())
            input_dim = int(hidden_dim)
        self.add_module(
            "img_net_logit",
            nn.Linear(input_dim, int(posterior.stoch_dim) * int(posterior.discrete_dim), bias=True),
        )
        self.stoch_dim = int(posterior.stoch_dim)
        self.discrete_dim = int(posterior.discrete_dim)

    def forward(self, deter: torch.Tensor) -> torch.Tensor:
        logits = super().forward(deter)
        return logits.reshape(*logits.shape[:-1], self.stoch_dim, self.discrete_dim)
