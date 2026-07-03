"""Sequence cores for STORM-style DMC world models."""

from __future__ import annotations

import torch
from torch import nn

from .world_model import StochasticTransformerKVCache


class TransformerSequenceCore(StochasticTransformerKVCache):
    """Native STORM transformer core with a small neutral wrapper name."""


class MambaSequenceCore(nn.Module):
    """Mamba3 replacement for STORM's transformer sequence core.

    The public contract matches ``StochasticTransformerKVCache``: full sequence
    forward for training and step/cache forward for live context and imagination.
    """

    def __init__(
        self,
        stoch_dim: int,
        action_dim: int,
        feat_dim: int,
        num_layers: int = 1,
        num_heads: int | None = None,
        max_length: int | None = None,
        dropout: float = 0.0,
        *,
        d_state: int = 32,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 16,
        is_mimo: bool = False,
        mimo_rank: int = 1,
        is_outproj_norm: bool = False,
    ):
        super().__init__()
        if int(num_layers) != 1:
            raise ValueError("MambaSequenceCore currently supports num_layers=1.")
        from models.dreamer.rssm import Mamba3Layer

        self.action_dim = int(action_dim)
        self.feat_dim = int(feat_dim)
        self.stem = nn.Sequential(
            nn.Linear(int(stoch_dim) + self.action_dim, self.feat_dim, bias=True),
            nn.LayerNorm(self.feat_dim),
        )
        self.layer = Mamba3Layer(
            deter=self.feat_dim,
            layer_idx=0,
            d_state=int(d_state),
            expand=int(expand),
            headdim=int(headdim),
            is_mimo=bool(is_mimo),
            mimo_rank=int(mimo_rank),
            chunk_size=max(1, int(chunk_size)),
            is_outproj_norm=bool(is_outproj_norm),
        )

    def _prepare_action(self, action: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if action.dim() == 2:
            action = action.unsqueeze(1)
        return action.to(device=device, dtype=dtype)

    def _token(self, samples: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = self._prepare_action(action, dtype=samples.dtype, device=samples.device)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        return self.stem(torch.cat([samples, action], dim=-1)).contiguous()

    def initial_cache(self, batch_size: int, dtype: torch.dtype, device=None) -> tuple[torch.Tensor, ...]:
        device = device or next(self.parameters()).device
        return self.layer.initial_context(batch_size, device=device, dtype=dtype)

    def _prepare_cache(self, cache: tuple[torch.Tensor, ...], token: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor.to(device=token.device, dtype=torch.float32 if idx == 0 else token.dtype).contiguous()
            for idx, tensor in enumerate(cache)
        )

    def reset_cache(self, cache: tuple[torch.Tensor, ...], reset: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = []
        for tensor in cache:
            mask = reset.reshape(reset.shape[0], *([1] * (tensor.dim() - 1))).to(
                device=tensor.device, dtype=torch.bool
            )
            out.append(torch.where(mask, torch.zeros_like(tensor), tensor))
        return tuple(out)

    def forward(self, samples: torch.Tensor, action: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        token = self._token(samples, action)
        cache = self.initial_cache(token.shape[0], dtype=token.dtype, device=token.device)
        outputs = []
        for idx in range(token.shape[1]):
            cache = self._prepare_cache(cache, token)
            out, *cache = self.layer.step(token[:, idx], *cache)
            cache = tuple(cache)
            outputs.append(out)
        return torch.stack(outputs, dim=1)

    def forward_step_with_cache(
        self,
        samples: torch.Tensor,
        action: torch.Tensor,
        cache: tuple[torch.Tensor, ...] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        assert samples.shape[1] == 1
        token = self._token(samples, action)
        if cache is None:
            cache = self.initial_cache(token.shape[0], dtype=token.dtype, device=token.device)
        cache = self._prepare_cache(cache, token)
        out, *cache = self.layer.step(token[:, 0], *cache)
        return out.unsqueeze(1), tuple(cache)
