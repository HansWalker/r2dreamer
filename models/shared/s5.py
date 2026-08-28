"""Compact S5 layer with parallel-sequence and recurrent interfaces."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _combine(left_a, left_b, right_a, right_b):
    return right_a * left_a, right_a * left_b + right_b


def _associative_scan(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive tree scan for the affine recurrence ``x = a * x + b``."""
    length = a.shape[1]
    if length < 2:
        return a, b

    pair_a, pair_b = _combine(a[:, 0:-1:2], b[:, 0:-1:2], a[:, 1::2], b[:, 1::2])
    odd_a, odd_b = _associative_scan(pair_a, pair_b)
    prior_a = odd_a[:, :-1] if length % 2 == 0 else odd_a
    prior_b = odd_b[:, :-1] if length % 2 == 0 else odd_b
    tail_a, tail_b = _combine(prior_a, prior_b, a[:, 2::2], b[:, 2::2])
    even_a = torch.cat((a[:, :1], tail_a), dim=1)
    even_b = torch.cat((b[:, :1], tail_b), dim=1)

    paired = odd_a.shape[1]
    out_a = torch.stack((even_a[:, :paired], odd_a), dim=2).flatten(1, 2)
    out_b = torch.stack((even_b[:, :paired], odd_b), dim=2).flatten(1, 2)
    if length % 2:
        out_a = torch.cat((out_a, even_a[:, -1:]), dim=1)
        out_b = torch.cat((out_b, even_b[:, -1:]), dim=1)
    return out_a, out_b


def _hippo_dplr(state_dim: int, blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the S5 HiPPO eigenvalues and eigenvectors used at initialization."""
    if state_dim % blocks:
        raise ValueError(f"S5 state_dim={state_dim} must be divisible by blocks={blocks}.")
    size = state_dim // blocks
    index = np.arange(size, dtype=np.float64)
    scale = np.sqrt(1 + 2 * index)
    hippo = -(np.tril(scale[:, None] * scale[None, :]) - np.diag(index))
    rank_one = np.sqrt(index + 0.5)
    normal = hippo + rank_one[:, None] * rank_one[None, :]
    decay = np.diag(normal).mean()
    skew = normal - decay * np.eye(size)
    frequency, vectors = np.linalg.eigh(skew * -1j)
    eigenvalues = np.tile(decay + 1j * frequency, blocks)
    eigenvectors = np.kron(np.eye(blocks), vectors)
    return (
        torch.from_numpy(eigenvalues.astype(np.complex64)),
        torch.from_numpy(eigenvectors.astype(np.complex64)),
    )


class S5StateSpace(nn.Module):
    """One multi-input, multi-output diagonal S5 state-space system."""

    def __init__(
        self,
        d_model: int,
        state_dim: int,
        *,
        blocks: int = 1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        discretization: str = "zoh",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.state_dim = int(state_dim)
        self.discretization = str(discretization)
        if self.discretization not in {"zoh", "bilinear"}:
            raise ValueError(f"Unsupported S5 discretization: {self.discretization}")

        eigenvalues, vectors = _hippo_dplr(self.state_dim, int(blocks))
        self.log_decay = nn.Parameter(torch.log((-eigenvalues.real).clamp_min(1e-4)))
        self.frequency = nn.Parameter(eigenvalues.imag)

        raw_b = torch.empty(self.state_dim, self.d_model)
        nn.init.normal_(raw_b, std=1 / math.sqrt(self.d_model))
        b = vectors.conj().transpose(0, 1) @ raw_b.to(torch.complex64)
        raw_c = torch.complex(
            torch.randn(self.d_model, self.state_dim),
            torch.randn(self.d_model, self.state_dim),
        ) / math.sqrt(2 * self.state_dim)
        c = raw_c @ vectors
        self.b_real = nn.Parameter(b.real)
        self.b_imag = nn.Parameter(b.imag)
        self.c_real = nn.Parameter(c.real)
        self.c_imag = nn.Parameter(c.imag)
        self.skip = nn.Parameter(torch.rand(self.d_model))
        self.log_step = nn.Parameter(torch.empty(self.state_dim).uniform_(math.log(dt_min), math.log(dt_max)))

    def initial_state(self, batch_size: int, device=None) -> torch.Tensor:
        device = device or self.log_decay.device
        return torch.zeros(batch_size, self.state_dim, dtype=torch.complex64, device=device)

    def _discrete(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eigenvalues = torch.complex(-self.log_decay.exp(), self.frequency)
        b = torch.complex(self.b_real, self.b_imag)
        step = self.log_step.exp()
        if self.discretization == "zoh":
            transition = torch.exp(eigenvalues * step)
            drive = ((transition - 1) / eigenvalues).unsqueeze(-1) * b
        else:
            inverse = 1 / (1 - 0.5 * step * eigenvalues)
            transition = inverse * (1 + 0.5 * step * eigenvalues)
            drive = (inverse * step).unsqueeze(-1) * b
        return transition, drive, torch.complex(self.c_real, self.c_imag)

    def step(self, value: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = self.initial_state(value.shape[0], value.device)
        transition, drive, readout = self._discrete()
        source = value.float().to(torch.complex64) @ drive.transpose(0, 1)
        state = transition * state.to(device=value.device, dtype=torch.complex64) + source
        output = (state @ readout.transpose(0, 1)).real + self.skip * value.float()
        return output.to(value.dtype), state

    def forward(
        self,
        value: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transition, drive, readout = self._discrete()
        source = value.float().to(torch.complex64) @ drive.transpose(0, 1)
        transitions = transition.reshape(1, 1, -1).expand_as(source)
        if state is not None:
            first = source[:, :1] + transition * state.to(device=value.device, dtype=torch.complex64).unsqueeze(1)
            source = torch.cat((first, source[:, 1:]), dim=1)
        _, states = _associative_scan(transitions, source)
        output = (states @ readout.transpose(0, 1)).real + self.skip * value.float()
        return output.to(value.dtype), states[:, -1]


class S5Block(nn.Module):
    """S5 state-space layer followed by the standard gated feed-forward block."""

    def __init__(self, d_model: int, config):
        super().__init__()
        d_model = int(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.ssm = S5StateSpace(
            d_model,
            int(config.state_dim),
            blocks=int(config.blocks),
            dt_min=float(config.dt_min),
            dt_max=float(config.dt_max),
            discretization=str(config.discretization),
        )
        self.dropout = nn.Dropout(float(config.dropout))
        hidden = int(config.ffn_dim)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_in = nn.Linear(d_model, 2 * hidden)
        self.ff_out = nn.Linear(hidden, d_model)

    def initial_state(self, batch_size: int, device=None) -> torch.Tensor:
        return self.ssm.initial_state(batch_size, device)

    def _feed_forward(self, value: torch.Tensor) -> torch.Tensor:
        content, gate = self.ff_in(self.ff_norm(value)).chunk(2, dim=-1)
        return value + self.dropout(self.ff_out(content * F.gelu(gate)))

    def forward(self, value: torch.Tensor, state=None, return_state=False):
        mixed, state = self.ssm(self.norm(value), state)
        output = self._feed_forward(value + self.dropout(F.gelu(mixed)))
        return (output, state) if return_state else output

    def step(self, value: torch.Tensor, state=None) -> tuple[torch.Tensor, torch.Tensor]:
        mixed, state = self.ssm.step(self.norm(value), state)
        return self._feed_forward(value + self.dropout(F.gelu(mixed))), state
