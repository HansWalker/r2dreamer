"""Sliding-window attention recurrent core for STORM."""

from __future__ import annotations

import torch
from torch import nn

from .transformer import AttentionBlockKVCache


def sliding_causal_mask(length: int, window_size: int, device) -> torch.Tensor:
    position = torch.arange(length, device=device)
    distance = position[:, None] - position[None, :]
    return ((distance >= 0) & (distance < window_size)).unsqueeze(0)


class CyclicPositionalEncoding1D(nn.Module):
    """Learned local positions that remain valid for an unbounded stream."""

    def __init__(self, period: int, embed_dim: int):
        super().__init__()
        self.period = int(period)
        self.embed_dim = int(embed_dim)
        self.pos_emb = nn.Embedding(self.period, self.embed_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        position = torch.arange(feat.shape[1], device=feat.device) % self.period
        return feat + self.pos_emb(position).unsqueeze(0)

    def step(self, feat: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        local_position = position.to(device=feat.device, dtype=torch.long) % self.period
        return feat + self.pos_emb(local_position).unsqueeze(1)


class SlidingWindowSequenceCore(nn.Module):
    """STORM attention with a bounded KV cache over an unbounded stream."""

    streaming = True

    def __init__(self, stem: nn.Module, feat_dim: int, config):
        super().__init__()
        settings = config.sliding_window
        self.feat_dim = int(feat_dim)
        self.window_size = int(settings.window_size)
        self.stem = stem
        self.position_encoding = CyclicPositionalEncoding1D(int(config.transformer.max_length), self.feat_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlockKVCache(
                feat_dim=self.feat_dim,
                hidden_dim=int(config.transformer.ffn_dim),
                num_heads=int(config.transformer.num_heads),
                dropout=float(config.dropout),
            )
            for _ in range(int(config.layers))
        ])
        self.layer_norm = nn.LayerNorm(self.feat_dim, eps=1e-6)

    @staticmethod
    def _prepare_action(action: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if action.dim() == 2:
            action = action.unsqueeze(1)
        return action.to(device=device, dtype=dtype)

    def forward(self, samples: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = self._prepare_action(action, dtype=samples.dtype, device=samples.device)
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.layer_norm(self.position_encoding(feats))
        mask = sliding_causal_mask(feats.shape[1], self.window_size, feats.device)
        for layer in self.layer_stack:
            feats, _ = layer(feats, feats, feats, mask)
        return feats

    def initial_cache(self, batch_size: int, dtype: torch.dtype, device=None) -> tuple[torch.Tensor, ...]:
        device = device or next(self.parameters()).device
        position = torch.zeros(batch_size, dtype=torch.long, device=device)
        layers = tuple(
            torch.zeros(batch_size, self.window_size, self.feat_dim, dtype=dtype, device=device)
            for _ in self.layer_stack
        )
        return (position, *layers)

    def reset_cache(self, cache: tuple[torch.Tensor, ...], reset: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.where(
                reset.to(tensor.device).reshape(reset.shape[0], *([1] * (tensor.ndim - 1))),
                torch.zeros_like(tensor),
                tensor,
            )
            for tensor in cache
        )

    def step(
        self,
        samples: torch.Tensor,
        action: torch.Tensor,
        cache: tuple[torch.Tensor, ...] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        assert samples.shape[1] == 1
        if cache is None:
            cache = self.initial_cache(samples.shape[0], dtype=samples.dtype, device=samples.device)
        position, *layer_cache = cache
        position = position.to(device=samples.device, dtype=torch.long)
        action = self._prepare_action(action, dtype=samples.dtype, device=samples.device)
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.layer_norm(self.position_encoding.step(feats, position))
        layer_cache = tuple(tensor.to(device=feats.device, dtype=feats.dtype) for tensor in layer_cache)

        batch = torch.arange(samples.shape[0], device=samples.device)
        slot = position % self.window_size
        valid = (
            torch.arange(self.window_size, device=samples.device)[None]
            < (position + 1).clamp(max=self.window_size)[:, None]
        )
        next_cache = []
        for index, layer in enumerate(self.layer_stack):
            current = layer_cache[index].clone()
            current[batch, slot] = feats[:, 0]
            feats, _ = layer(feats, current, current, valid.unsqueeze(1))
            next_cache.append(current)
        return feats, (position + 1, *next_cache)
