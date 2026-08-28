"""Sliding-window attention deterministic core for Dreamer's RSSM."""

import torch
import torch.nn.functional as F
from torch import nn

from models.shared.utils import normalize_action, weight_init_


class DreamerSlidingWindowCore(nn.Module):
    """Causal attention over a fixed-size rolling episode history."""

    cache_keys = ("sliding_position", "sliding_kv_state")

    def __init__(self, deter, stoch, act_dim, config):
        super().__init__()
        self.width = int(config.width)
        self.num_heads = int(config.num_heads)
        self.window_size = int(config.window_size)
        if self.width % self.num_heads:
            raise ValueError("Dreamer sliding-window width must be divisible by num_heads.")
        if self.window_size < 1:
            raise ValueError("Dreamer sliding-window window_size must be positive.")

        self.head_dim = self.width // self.num_heads
        self.scale = self.head_dim**-0.5
        self.token = nn.Linear(int(stoch) + int(act_dim), self.width)
        self.input_norm = nn.RMSNorm(self.width, eps=1e-4, dtype=torch.float32)
        self.query = nn.Linear(self.width, self.width, bias=False)
        self.key = nn.Linear(self.width, self.width, bias=False)
        self.value = nn.Linear(self.width, self.width, bias=False)
        self.attention_output = nn.Linear(self.width, self.width, bias=False)
        self.relative_bias = nn.Parameter(torch.zeros(self.num_heads, self.window_size))
        self.ffn_norm = nn.RMSNorm(self.width, eps=1e-4, dtype=torch.float32)
        self.ffn = nn.Sequential(
            nn.Linear(self.width, int(config.ffn_dim)),
            getattr(nn, str(config.act))(),
            nn.Linear(int(config.ffn_dim), self.width),
        )
        self.dropout = nn.Dropout(float(config.dropout))
        self.output = nn.Linear(self.width, int(deter))
        self.output_norm = nn.RMSNorm(int(deter), eps=1e-4, dtype=torch.float32)
        self.apply(weight_init_)
        nn.init.zeros_(self.relative_bias)

    def initial_context(self, batch_size, device=None, dtype=None):
        device = device or self.token.weight.device
        dtype = dtype or self.token.weight.dtype
        return (
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.zeros(
                batch_size,
                self.window_size,
                2,
                self.num_heads,
                self.head_dim,
                dtype=dtype,
                device=device,
            ),
        )

    def forward(self, stoch, action, position=None, key_value=None):
        batch_size = action.shape[0]
        stoch = stoch.reshape(batch_size, -1)
        action = normalize_action(action)
        token = self.input_norm(self.token(torch.cat((stoch, action), dim=-1)))

        query = self.query(token).view(batch_size, self.num_heads, 1, self.head_dim)
        current_key = self.key(token).view(batch_size, self.num_heads, self.head_dim)
        current_value = self.value(token).view(batch_size, self.num_heads, self.head_dim)
        if position is None or key_value is None:
            position, key_value = self.initial_context(batch_size, current_key.device, current_key.dtype)
        else:
            position = position.to(device=current_key.device, dtype=torch.long)
            key_value = key_value.to(device=current_key.device, dtype=current_key.dtype)

        batch = torch.arange(batch_size, device=token.device)
        slots = torch.arange(self.window_size, device=token.device)
        current_slot = position % self.window_size
        key_value = key_value.clone()
        key_value[batch, current_slot, 0] = current_key
        key_value[batch, current_slot, 1] = current_value
        key = key_value[:, :, 0].transpose(1, 2)
        value = key_value[:, :, 1].transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        age = (current_slot[:, None] - slots[None]) % self.window_size
        bias = self.relative_bias[:, age].permute(1, 0, 2).unsqueeze(2).to(logits.dtype)
        logits = logits + bias
        valid = slots[None] < (position + 1).clamp(max=self.window_size)[:, None]
        logits = logits.masked_fill(~valid[:, None, None], torch.finfo(logits.dtype).min)
        weights = self.dropout(F.softmax(logits.float(), dim=-1).to(value.dtype))
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch_size, self.width)

        token = token + self.dropout(self.attention_output(attended))
        token = token + self.dropout(self.ffn(self.ffn_norm(token)))
        return self.output_norm(self.output(token)), position + 1, key_value
