"""Small transformer primitives shared by the JEPA-style model families."""

import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, value):
        return self.net(value)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        inner_dim = self.heads * self.dim_head
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.dropout = float(dropout)

    def forward(self, value, mask=None):
        batch, length, _ = value.shape
        q, k, v = self.to_qkv(self.norm(value)).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.reshape(batch, length, self.heads, self.dim_head).transpose(1, 2)

        output = F.scaled_dot_product_attention(
            split_heads(q),
            split_heads(k),
            split_heads(v),
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=mask is None,
        )
        return self.to_out(output.transpose(1, 2).reshape(batch, length, -1))
