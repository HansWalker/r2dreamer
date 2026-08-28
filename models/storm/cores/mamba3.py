"""Mamba3 recurrent core for STORM."""

from __future__ import annotations

import torch
from torch import nn

from models.shared.mamba3 import Mamba3Layer


class MambaSequenceCore(nn.Module):
    """Mamba3 replacement for STORM's Transformer recurrent core."""

    streaming = True

    def __init__(self, stem: nn.Module, feat_dim: int, config):
        super().__init__()
        if int(config.layers) != 1:
            raise ValueError("MambaSequenceCore currently supports num_layers=1.")
        feat_dim = int(feat_dim)
        self.stem = stem
        self.dropout = nn.Dropout(float(config.dropout))
        self.output_norm = nn.LayerNorm(feat_dim, eps=1e-6)
        self.layer = Mamba3Layer(
            d_model=feat_dim,
            config=config.mamba3,
            layer_idx=0,
        )

    def _token(self, samples: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.dim() == 2:
            action = action.unsqueeze(1)
        action = action.to(device=samples.device, dtype=samples.dtype)
        return self.stem(torch.cat([samples, action], dim=-1)).contiguous()

    def initial_cache(self, batch_size: int, dtype: torch.dtype, device=None) -> tuple[torch.Tensor, ...]:
        device = device or next(self.parameters()).device
        return self.layer.initial_context(batch_size, device=device, dtype=dtype)

    def reset_cache(self, cache: tuple[torch.Tensor, ...], reset: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.where(
                reset.to(tensor.device).reshape(reset.shape[0], *([1] * (tensor.ndim - 1))),
                torch.zeros_like(tensor),
                tensor,
            )
            for tensor in cache
        )

    def forward(self, samples: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        token = self._token(samples, action)
        return self.output_norm(token + self.dropout(self.layer(token)))

    def step(
        self,
        samples: torch.Tensor,
        action: torch.Tensor,
        cache: tuple[torch.Tensor, ...] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        assert samples.shape[1] == 1
        token = self._token(samples, action)
        if cache is None:
            cache = self.initial_cache(token.shape[0], dtype=token.dtype, device=token.device)
        output, *cache = self.layer.step(token[:, 0], *cache)
        output = self.output_norm(token[:, 0] + self.dropout(output))
        return output.unsqueeze(1), tuple(cache)
