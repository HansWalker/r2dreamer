"""S5 deterministic core for Dreamer's RSSM."""

import torch
from torch import nn

from models.shared.s5 import S5Block
from models.shared.utils import normalize_action, weight_init_


class DreamerS5Core(nn.Module):
    cache_keys = ("s5_state",)

    def __init__(self, deter, stoch, act_dim, config):
        super().__init__()
        self.deter = int(deter)
        self.token = nn.Linear(int(stoch) + int(act_dim), self.deter)
        self.input_norm = nn.RMSNorm(self.deter, eps=1e-4, dtype=torch.float32)
        self.layer = S5Block(self.deter, config)
        self.output_norm = nn.RMSNorm(self.deter, eps=1e-4, dtype=torch.float32)
        weight_init_(self.token)

    def initial_context(self, batch_size, device=None, dtype=None):
        return (self.layer.initial_state(batch_size, device),)

    def forward(self, stoch, action, state=None):
        batch_size = action.shape[0]
        stoch = stoch.reshape(batch_size, -1)
        action = normalize_action(action)
        token = self.input_norm(self.token(torch.cat((stoch, action), dim=-1)))
        output, state = self.layer.step(token, state)
        return self.output_norm(output), state
