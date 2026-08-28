"""Hyena deterministic core for Dreamer's RSSM."""

import torch
from torch import nn

from models.shared.hyena import HyenaBlock
from models.shared.utils import normalize_action, weight_init_


class DreamerHyenaCore(nn.Module):
    cache_keys = ("hyena_short_state", "hyena_long_state", "hyena_length")

    def __init__(self, deter, stoch, act_dim, config):
        super().__init__()
        self.width = int(config.width)
        self.token = nn.Linear(int(stoch) + int(act_dim), self.width)
        self.input_norm = nn.RMSNorm(self.width, eps=1e-4, dtype=torch.float32)
        self.layer = HyenaBlock(self.width, config)
        self.output = nn.Linear(self.width, int(deter))
        self.output_norm = nn.RMSNorm(int(deter), eps=1e-4, dtype=torch.float32)
        self._kernel = None
        weight_init_(self.token)
        weight_init_(self.output)

    def initial_context(self, batch_size, device=None, dtype=None):
        dtype = dtype or self.token.weight.dtype
        return self.layer.initial_state(batch_size, dtype, device)

    def prepare_sequence(self, reference):
        self._kernel = self.layer.kernel(reference)

    def clear_sequence(self):
        self._kernel = None

    def forward(self, stoch, action, short_state=None, long_state=None, length=None):
        batch_size = action.shape[0]
        stoch = stoch.reshape(batch_size, -1)
        action = normalize_action(action)
        token = self.input_norm(self.token(torch.cat((stoch, action), dim=-1)))
        state = None if short_state is None else (short_state, long_state, length)
        output, state = self.layer.step(token, state, self._kernel)
        return self.output_norm(self.output(output)), *state
