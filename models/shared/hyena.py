"""Hyena long-convolution operator adapted from HazyResearch Safari.

The sequence path uses causal FFT convolution. The recurrent path computes the
same bounded filter from a rolling history, which is required for environment
interaction and latent imagination.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def fft_convolution(value: torch.Tensor, kernel: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Apply a depthwise causal convolution to tensors shaped (B, D, T)."""
    length = value.shape[-1]
    fft_size = 2 * length
    value_dtype = value.dtype
    value_fft = torch.fft.rfft(value.float(), n=fft_size)
    kernel_fft = torch.fft.rfft(kernel.float(), n=fft_size)
    output = torch.fft.irfft(value_fft * kernel_fft.unsqueeze(0), n=fft_size)[..., :length]
    return (output + value.float() * skip[None, :, None]).to(value_dtype)


class Sin(nn.Module):
    def __init__(self, width: int, frequency: float):
        super().__init__()
        self.frequency = nn.Parameter(torch.full((1, int(width)), float(frequency)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.frequency * value)


class HyenaFilter(nn.Module):
    """Implicit sinusoidal filter with the exponential modulation used by Hyena."""

    def __init__(self, channels: int, max_length: int, config):
        super().__init__()
        channels = int(channels)
        self.max_length = int(max_length)
        embedding_dim = int(config.embedding_dim)
        if embedding_dim < 3 or embedding_dim % 2 == 0:
            raise ValueError("Hyena embedding_dim must be an odd integer of at least 3.")
        width = int(config.filter_order)
        bands = (embedding_dim - 1) // 2
        time = torch.linspace(0, 1, self.max_length).unsqueeze(-1)
        position = torch.arange(self.max_length).float().unsqueeze(-1)
        frequency = torch.linspace(1e-4, max(1, bands - 1), bands).unsqueeze(0)
        angle = 2 * math.pi * position * frequency / self.max_length
        self.register_buffer("position", torch.cat((time, torch.cos(angle), -torch.sin(angle)), dim=-1))
        self.register_buffer("time", time)

        self.input = nn.Linear(embedding_dim, width)
        self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(int(config.filter_layers)))
        self.activation = Sin(width, float(config.frequency))
        self.output = nn.Linear(width, channels, bias=False)
        self.bias = nn.Parameter(torch.randn(channels))

        target = float(config.modulation_target)
        fast = float(config.fast_decay)
        slow = float(config.slow_decay)
        decay = torch.linspace(math.log(target) / slow, math.log(target) / fast, channels)
        self.register_buffer("decay", decay.reshape(1, channels))

    def forward(self, length: int) -> torch.Tensor:
        value = self.activation(self.input(self.position[:length]))
        for layer in self.hidden:
            value = self.activation(layer(value))
        value = self.output(value)
        return value * torch.exp(-self.time[:length] * self.decay.abs())


class HyenaOperator(nn.Module):
    """Order-two Hyena operator with an exact finite-history step interface."""

    def __init__(self, d_model: int, config):
        super().__init__()
        if int(config.order) != 2:
            raise ValueError("The recurrent Hyena implementation currently supports order=2 only.")
        self.d_model = int(d_model)
        self.max_length = int(config.max_length)
        inner = 3 * self.d_model
        self.in_proj = nn.Linear(self.d_model, inner)
        self.short_filter = nn.Conv1d(inner, inner, 3, padding=2, groups=inner)
        self.filter = HyenaFilter(self.d_model, self.max_length, config)
        self.dropout = nn.Dropout(float(config.dropout))
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self._inference_kernel = None
        self._inference_version = None

    def _kernel(self, length: int, reference: torch.Tensor) -> torch.Tensor:
        versions = tuple(parameter._version for parameter in self.filter.parameters())
        if not torch.is_grad_enabled() and self._inference_version == versions:
            kernel = self._inference_kernel
            if kernel is not None and kernel.device == reference.device and kernel.shape[0] >= length:
                return kernel[:length].to(reference.dtype)
        kernel = self.filter(length).to(device=reference.device, dtype=reference.dtype)
        if not torch.is_grad_enabled():
            self._inference_kernel = kernel.detach()
            self._inference_version = versions
        return kernel

    def initial_state(self, batch_size: int, dtype: torch.dtype, device=None) -> tuple[torch.Tensor, ...]:
        device = device or self.in_proj.weight.device
        return (
            torch.zeros(batch_size, 2, 3 * self.d_model, dtype=dtype, device=device),
            torch.zeros(batch_size, self.max_length, self.d_model, dtype=dtype, device=device),
            torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        length = value.shape[1]
        if length > self.max_length:
            raise ValueError(f"Hyena sequence length {length} exceeds max_length={self.max_length}.")
        projected = self.in_proj(value).transpose(1, 2)
        short = self.short_filter(projected)[..., :length]
        x0, x1, content = short.chunk(3, dim=1)
        content = self.dropout(content * x1)
        kernel = self._kernel(length, content).transpose(0, 1)
        content = fft_convolution(content, kernel, self.filter.bias)
        return self.out_proj((content * x0).transpose(1, 2))

    def step(
        self,
        value: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
        kernel: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if state is None:
            state = self.initial_state(value.shape[0], value.dtype, value.device)
        short_history, long_history, length = state
        projected = self.in_proj(value)
        short_history = short_history.to(device=projected.device, dtype=projected.dtype)
        short_input = torch.cat((short_history, projected.unsqueeze(1)), dim=1)
        short = F.conv1d(
            short_input.transpose(1, 2),
            self.short_filter.weight,
            self.short_filter.bias,
            groups=3 * self.d_model,
        )[:, :, 0]
        x0, x1, content = short.chunk(3, dim=-1)
        content = self.dropout(content * x1)
        long_history = long_history.to(device=content.device, dtype=content.dtype)
        long_history = torch.cat((content.unsqueeze(1), long_history[:, :-1]), dim=1)
        length = (length.to(value.device) + 1).clamp(max=self.max_length)
        kernel = (
            self._kernel(self.max_length, content)
            if kernel is None
            else kernel.to(device=content.device, dtype=content.dtype)
        )
        valid = torch.arange(self.max_length, device=value.device)[None, :] < length[:, None]
        mixed = (long_history * kernel.unsqueeze(0) * valid.unsqueeze(-1)).sum(1)
        mixed = mixed + self.filter.bias * content
        output = self.out_proj(mixed * x0)
        return output, (short_input[:, 1:], long_history, length)


class HyenaBlock(nn.Module):
    """Pre-normalized Hyena operator and gated feed-forward block."""

    def __init__(self, d_model: int, config):
        super().__init__()
        d_model = int(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.operator = HyenaOperator(d_model, config)
        self.dropout = nn.Dropout(float(config.dropout))
        hidden = int(config.ffn_dim)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_in = nn.Linear(d_model, 2 * hidden)
        self.ff_out = nn.Linear(hidden, d_model)

    def initial_state(self, batch_size: int, dtype: torch.dtype, device=None) -> tuple[torch.Tensor, ...]:
        return self.operator.initial_state(batch_size, dtype, device)

    def kernel(self, reference: torch.Tensor) -> torch.Tensor:
        return self.operator._kernel(self.operator.max_length, reference)

    def _feed_forward(self, value: torch.Tensor) -> torch.Tensor:
        content, gate = self.ff_in(self.ff_norm(value)).chunk(2, dim=-1)
        return value + self.dropout(self.ff_out(content * F.gelu(gate)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.dropout(self.operator(self.norm(value)))
        return self._feed_forward(value)

    def step(self, value: torch.Tensor, state=None, kernel=None) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        mixed, state = self.operator.step(self.norm(value), state, kernel)
        return self._feed_forward(value + self.dropout(mixed)), state
