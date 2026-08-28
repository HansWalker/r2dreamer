"""Hyena sequence core for STORM."""

from __future__ import annotations

import torch
from torch import nn

from models.shared.hyena import HyenaBlock


class HyenaSequenceCore(nn.Module):
    streaming = True

    def __init__(self, stem: nn.Module, feat_dim: int, config):
        super().__init__()
        if int(config.layers) != 1:
            raise ValueError("HyenaSequenceCore currently supports layers=1.")
        self.stem = stem
        self.layer = HyenaBlock(int(feat_dim), config.hyena)
        self.output_norm = nn.LayerNorm(int(feat_dim), eps=1e-6)

    def _token(self, samples, action):
        if action.dim() == 2:
            action = action.unsqueeze(1)
        action = action.to(device=samples.device, dtype=samples.dtype)
        return self.stem(torch.cat((samples, action), dim=-1))

    def initial_cache(self, batch_size: int, dtype: torch.dtype, device=None):
        return self.layer.initial_state(batch_size, dtype, device)

    def reset_cache(self, cache, reset):
        return tuple(
            torch.where(
                reset.to(value.device).reshape(reset.shape[0], *([1] * (value.ndim - 1))),
                torch.zeros_like(value),
                value,
            )
            for value in cache
        )

    def forward(self, samples, action):
        return self.output_norm(self.layer(self._token(samples, action)))

    def step(self, samples, action, cache=None):
        assert samples.shape[1] == 1
        output, cache = self.layer.step(self._token(samples, action)[:, 0], cache)
        return self.output_norm(output).unsqueeze(1), cache
