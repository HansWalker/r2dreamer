"""STORM-style world model adapted to DMC state observations.

This module keeps the upstream STORM transformer dynamics close to the original
implementation while replacing the Atari CNN observation edge with a dense
encoder for DMC proprioceptive states.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import OneHotCategorical


def get_subsequent_mask_with_batch_length(batch_length: int, device) -> torch.Tensor:
    """Return STORM's causal attention mask where True means visible."""
    return (1 - torch.triu(torch.ones((1, batch_length, batch_length), device=device), diagonal=1)).bool()


def _activation(name: str):
    if name.lower() == "relu":
        return nn.ReLU
    return getattr(nn, name)


def _is_mapping_like(value) -> bool:
    return isinstance(value, Mapping) or hasattr(value, "keys")


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    layers: int,
    *,
    act: str = "ReLU",
    norm: bool = True,
    bias: bool = False,
) -> nn.Sequential:
    act_cls = _activation(act)
    modules: list[nn.Module] = []
    dim = int(in_dim)
    for _ in range(int(layers)):
        modules.append(nn.Linear(dim, int(hidden_dim), bias=bias))
        if norm:
            modules.append(nn.LayerNorm(int(hidden_dim)))
        modules.append(act_cls(inplace=True) if act_cls is nn.ReLU else act_cls())
        dim = int(hidden_dim)
    modules.append(nn.Linear(dim, int(out_dim), bias=bias))
    return nn.Sequential(*modules)


class MLPObservationEncoder(nn.Module):
    """Dense replacement for STORM's image CNN encoder."""

    def __init__(
        self,
        obs_shapes: Mapping[str, tuple[int, ...]] | tuple[int, ...],
        embedding_dim: int,
        hidden_dim: int,
        layers: int,
        *,
        act: str = "ReLU",
        norm: bool = True,
        keys: tuple[str, ...] | None = None,
    ):
        super().__init__()
        self.obs_shapes = obs_shapes
        if isinstance(obs_shapes, Mapping):
            excluded = {"is_first", "is_last", "is_terminal", "reward"}
            self.keys = tuple(keys) if keys is not None else tuple(
                k for k in obs_shapes if k not in excluded and not k.startswith("log_")
            )
            self.in_dim = sum(math.prod(obs_shapes[key]) for key in self.keys)
        else:
            self.keys = None
            self.in_dim = math.prod(obs_shapes)
        self.out_dim = int(embedding_dim)
        self.backbone = _build_mlp(self.in_dim, self.out_dim, hidden_dim, layers, act=act, norm=norm)

    def _flatten_obs(self, obs) -> torch.Tensor:
        if _is_mapping_like(obs):
            parts = []
            for key in self.keys:
                value = obs[key]
                obs_rank = len(self.obs_shapes[key])
                prefix = value.shape[:-obs_rank] if obs_rank else value.shape
                parts.append(value.reshape(*prefix, -1))
            return torch.cat(parts, dim=-1)
        obs_rank = len(self.obs_shapes)
        prefix = obs.shape[:-obs_rank] if obs_rank else obs.shape
        return obs.reshape(*prefix, -1)

    def forward(self, obs) -> torch.Tensor:
        # dict/tensor of (B, L, ...) -> (B, L, E), or (B, ...) -> (B, E)
        x = self._flatten_obs(obs)
        prefix = x.shape[:-1]
        x = self.backbone(x.reshape(-1, x.shape[-1]))
        return x.reshape(*prefix, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, max_length: int, embed_dim: int):
        super().__init__()
        self.max_length = int(max_length)
        self.embed_dim = int(embed_dim)
        self.pos_emb = nn.Embedding(self.max_length, self.embed_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.shape[1] > self.max_length:
            raise ValueError(
                f"STORM sequence length {feat.shape[1]} exceeds transformer_max_length={self.max_length}."
            )
        pos = self.pos_emb(torch.arange(self.max_length, device=feat.device))
        return feat + pos[None, : feat.shape[1]]

    def forward_with_position(self, feat: torch.Tensor, position: int | torch.Tensor) -> torch.Tensor:
        assert feat.shape[1] == 1
        position = torch.as_tensor(position, device=feat.device, dtype=torch.long)
        if torch.any(position >= self.max_length):
            max_position = int(position.max().item())
            raise ValueError(
                f"STORM cache position {max_position} exceeds transformer_max_length={self.max_length}."
            )
        pos = self.pos_emb(position.reshape(-1))
        return feat + pos.reshape(feat.shape[0], 1, self.embed_dim)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, torch.finfo(attn.dtype).min)
        attn = self.dropout(F.softmax(attn, dim=-1))
        return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1):
        super().__init__()
        self.n_head = int(n_head)
        self.d_k = int(d_k)
        self.d_v = int(d_v)
        self.w_qs = nn.Linear(d_model, self.n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, self.n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, self.n_head * self.d_v, bias=False)
        self.fc = nn.Linear(self.n_head * self.d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=self.d_k**0.5, attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(batch, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(batch, len_q, -1)
        q = self.dropout(self.fc(q))
        q = self.layer_norm(q + residual)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        return self.layer_norm(x + residual)


class AttentionBlockKVCache(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            num_heads,
            feat_dim,
            feat_dim // num_heads,
            feat_dim // num_heads,
            dropout=dropout,
        )
        self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
        output, attn = self.slf_attn(q, k, v, mask=mask)
        output = self.pos_ffn(output)
        return output, attn


class StochasticTransformerKVCache(nn.Module):
    """STORM Transformer, with continuous DMC actions used directly."""

    def __init__(self, stoch_dim: int, action_dim: int, feat_dim: int, num_layers: int, num_heads: int, max_length: int, dropout: float):
        super().__init__()
        self.action_dim = int(action_dim)
        self.feat_dim = int(feat_dim)
        self.stem = nn.Sequential(
            nn.Linear(int(stoch_dim) + self.action_dim, self.feat_dim, bias=False),
            nn.LayerNorm(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim, bias=False),
            nn.LayerNorm(self.feat_dim),
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=self.feat_dim)
        self.layer_stack = nn.ModuleList(
            [
                AttentionBlockKVCache(
                    feat_dim=self.feat_dim,
                    hidden_dim=self.feat_dim * 2,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(self.feat_dim, eps=1e-6)

    def _prepare_action(self, action: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if action.dim() == 2:
            action = action.unsqueeze(1)
        return action.to(device=device, dtype=dtype)

    def forward(self, samples: torch.Tensor, action: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if samples.shape[1] > self.position_encoding.max_length:
            raise ValueError(
                f"STORM sequence length {samples.shape[1]} exceeds "
                f"transformer_max_length={self.position_encoding.max_length}."
            )
        action = self._prepare_action(action, dtype=samples.dtype, device=samples.device)
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)
        for layer in self.layer_stack:
            feats, _ = layer(feats, feats, feats, mask)
        return feats

    def initial_cache(self, batch_size: int, dtype: torch.dtype, device=None) -> tuple[torch.Tensor, ...]:
        device = device or next(self.parameters()).device
        position = torch.zeros((batch_size,), dtype=torch.long, device=device)
        layer_cache = tuple(
            torch.zeros((batch_size, 0, self.feat_dim), dtype=dtype, device=device) for _ in self.layer_stack
        )
        return (position, *layer_cache)

    def reset_cache(self, cache: tuple[torch.Tensor, ...], reset: torch.Tensor) -> tuple[torch.Tensor, ...]:
        position, *layer_cache = cache
        reset_pos = reset.reshape(reset.shape[0]).to(device=position.device, dtype=torch.bool)
        position = torch.where(reset_pos, torch.zeros_like(position), position)
        reset_cache = reset.reshape(reset.shape[0], 1, 1).to(device=layer_cache[0].device, dtype=torch.bool)
        layer_cache = tuple(torch.where(reset_cache, torch.zeros_like(tensor), tensor) for tensor in layer_cache)
        return (position, *layer_cache)

    def forward_step_with_cache(
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
        cache_length = layer_cache[0].shape[1] if layer_cache else 0
        next_length = max(cache_length, int(position.max().item()) + 1 if position.numel() else 1)
        if next_length > self.position_encoding.max_length:
            raise ValueError(
                f"STORM cache length {next_length} exceeds transformer_max_length={self.position_encoding.max_length}."
            )
        action = self._prepare_action(action, dtype=samples.dtype, device=samples.device)
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding.forward_with_position(feats, position=position)
        feats = self.layer_norm(feats)
        layer_cache = tuple(tensor.to(device=feats.device, dtype=feats.dtype) for tensor in layer_cache)
        mask = torch.arange(next_length, device=samples.device).reshape(1, 1, next_length) <= position.reshape(-1, 1, 1)
        batch_index = torch.arange(samples.shape[0], device=samples.device)
        next_cache = []
        for idx, layer in enumerate(self.layer_stack):
            current_cache = layer_cache[idx]
            if current_cache.shape[1] < next_length:
                pad = current_cache.new_zeros(current_cache.shape[0], next_length - current_cache.shape[1], current_cache.shape[2])
                current_cache = torch.cat([current_cache, pad], dim=1)
            current_cache = current_cache.clone()
            current_cache[batch_index, position] = feats[:, 0]
            feats, _ = layer(feats, current_cache, current_cache, mask)
            next_cache.append(current_cache)
        return feats, (position + 1, *next_cache)


class DistHead(nn.Module):
    def __init__(self, encoder_feat_dim: int, transformer_hidden_dim: int, stoch_dim: int):
        super().__init__()
        self.stoch_dim = int(stoch_dim)
        self.post_head = nn.Linear(encoder_feat_dim, self.stoch_dim * self.stoch_dim)
        self.prior_head = nn.Linear(transformer_hidden_dim, self.stoch_dim * self.stoch_dim)

    def unimix(self, logits: torch.Tensor, mixing_ratio: float = 0.01) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        mixed = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1 - mixing_ratio) * probs
        return torch.log(mixed)

    def forward_post(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.post_head(x)
        logits = logits.reshape(*logits.shape[:-1], self.stoch_dim, self.stoch_dim)
        return self.unimix(logits)

    def forward_prior(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.prior_head(x)
        logits = logits.reshape(*logits.shape[:-1], self.stoch_dim, self.stoch_dim)
        return self.unimix(logits)


class StormWorldModel(nn.Module):
    """STORM world model with MLP observation edges for DMC."""

    def __init__(
        self,
        obs_shapes: Mapping[str, tuple[int, ...]] | tuple[int, ...],
        action_dim: int,
        *,
        transformer_max_length: int | None = None,
        batch_length: int | None = None,
        transformer_hidden_dim: int = 512,
        transformer_num_layers: int = 2,
        transformer_num_heads: int = 8,
        stoch_dim: int = 32,
        encoder_hidden_dim: int = 512,
        encoder_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if transformer_max_length is None:
            transformer_max_length = int(batch_length) if batch_length is not None else 64
        elif batch_length is not None and int(transformer_max_length) < int(batch_length):
            raise ValueError(
                f"transformer_max_length={transformer_max_length} must be >= batch_length={batch_length}."
            )
        self.batch_length = int(batch_length) if batch_length is not None else None
        self.transformer_max_length = int(transformer_max_length)
        self.transformer_hidden_dim = int(transformer_hidden_dim)
        self.stoch_dim = int(stoch_dim)
        self.stoch_flattened_dim = self.stoch_dim * self.stoch_dim
        self.encoder = MLPObservationEncoder(
            obs_shapes,
            embedding_dim=self.transformer_hidden_dim,
            hidden_dim=encoder_hidden_dim,
            layers=encoder_layers,
        )
        self.storm_transformer = StochasticTransformerKVCache(
            stoch_dim=self.stoch_flattened_dim,
            action_dim=action_dim,
            feat_dim=self.transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=self.transformer_max_length,
            dropout=dropout,
        )
        self.dist_head = DistHead(
            encoder_feat_dim=self.encoder.out_dim,
            transformer_hidden_dim=self.transformer_hidden_dim,
            stoch_dim=self.stoch_dim,
        )

    def straight_through_gradient(self, logits: torch.Tensor, sample_mode: str = "random_sample") -> torch.Tensor:
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            return dist.sample() + dist.probs - dist.probs.detach()
        if sample_mode == "mode":
            return F.one_hot(torch.argmax(logits, dim=-1), logits.shape[-1]).to(logits.dtype)
        if sample_mode == "probs":
            return dist.probs
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    def flatten_sample(self, sample: torch.Tensor) -> torch.Tensor:
        return sample.reshape(*sample.shape[:-2], self.stoch_flattened_dim)
