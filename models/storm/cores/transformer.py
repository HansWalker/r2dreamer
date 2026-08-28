"""Transformer recurrent core for STORM."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def causal_mask(length: int, device) -> torch.Tensor:
    """Return STORM's causal attention mask where True means visible."""
    return torch.ones((1, length, length), dtype=torch.bool, device=device).tril()


class PositionalEncoding1D(nn.Module):
    def __init__(self, max_length: int, embed_dim: int):
        super().__init__()
        self.max_length = int(max_length)
        self.embed_dim = int(embed_dim)
        self.pos_emb = nn.Embedding(self.max_length, self.embed_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.shape[1] > self.max_length:
            raise ValueError(f"STORM sequence length {feat.shape[1]} exceeds transformer_max_length={self.max_length}.")
        pos = self.pos_emb(torch.arange(self.max_length, device=feat.device))
        return feat + pos[None, : feat.shape[1]]

    def forward_with_position(self, feat: torch.Tensor, position: int | torch.Tensor) -> torch.Tensor:
        assert feat.shape[1] == 1
        position = torch.as_tensor(position, device=feat.device, dtype=torch.long)
        if torch.any(position >= self.max_length):
            max_position = int(position.max().item())
            raise ValueError(f"STORM cache position {max_position} exceeds transformer_max_length={self.max_length}.")
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
        batch, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(batch, len_q, self.n_head, self.d_k)
        k = self.w_ks(k).view(batch, len_k, self.n_head, self.d_k)
        v = self.w_vs(v).view(batch, len_v, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(batch, len_q, -1)
        q = self.dropout(self.fc(q))
        return self.layer_norm(q + residual), attn


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
        return self.layer_norm(self.dropout(x) + residual)


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
        return self.pos_ffn(output), attn


class TransformerSequenceCore(nn.Module):
    """STORM Transformer recurrent core with continuous DMC actions."""

    streaming = False

    def __init__(self, stem: nn.Module, feat_dim: int, config):
        super().__init__()
        transformer = config.transformer
        self.feat_dim = int(feat_dim)
        self.stem = stem
        self.position_encoding = PositionalEncoding1D(
            max_length=int(transformer.max_length),
            embed_dim=self.feat_dim,
        )
        self.layer_stack = nn.ModuleList([
            AttentionBlockKVCache(
                feat_dim=self.feat_dim,
                hidden_dim=int(transformer.ffn_dim),
                num_heads=int(transformer.num_heads),
                dropout=float(config.dropout),
            )
            for _ in range(int(config.layers))
        ])
        self.layer_norm = nn.LayerNorm(self.feat_dim, eps=1e-6)

    def _prepare_action(self, action: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if action.dim() == 2:
            action = action.unsqueeze(1)
        return action.to(device=device, dtype=dtype)

    def forward(self, samples: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = self._prepare_action(action, dtype=samples.dtype, device=samples.device)
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.layer_norm(self.position_encoding(feats))
        mask = causal_mask(feats.shape[1], feats.device)
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
        cache_length = layer_cache[0].shape[1] if layer_cache else 0
        next_length = max(cache_length, int(position.max().item()) + 1 if position.numel() else 1)
        if next_length > self.position_encoding.max_length:
            raise ValueError(
                f"STORM cache length {next_length} exceeds transformer_max_length={self.position_encoding.max_length}."
            )
        action = self._prepare_action(action, dtype=samples.dtype, device=samples.device)
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.layer_norm(self.position_encoding.forward_with_position(feats, position=position))
        layer_cache = tuple(tensor.to(device=feats.device, dtype=feats.dtype) for tensor in layer_cache)
        mask = torch.arange(next_length, device=samples.device).reshape(1, 1, next_length) <= position.reshape(-1, 1, 1)
        batch_index = torch.arange(samples.shape[0], device=samples.device)
        next_cache = []
        for index, layer in enumerate(self.layer_stack):
            current_cache = layer_cache[index]
            if current_cache.shape[1] < next_length:
                padding = current_cache.new_zeros(
                    current_cache.shape[0],
                    next_length - current_cache.shape[1],
                    current_cache.shape[2],
                )
                current_cache = torch.cat([current_cache, padding], dim=1)
            current_cache = current_cache.clone()
            current_cache[batch_index, position] = feats[:, 0]
            feats, _ = layer(feats, current_cache, current_cache, mask)
            next_cache.append(current_cache)
        return feats, (position + 1, *next_cache)
