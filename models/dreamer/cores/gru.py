"""Block-GRU deterministic core for Dreamer."""

import torch
from torch import nn

from models.shared.utils import normalize_action

from ..networks import BlockLinear


class DreamerGRUCore(nn.Module):
    """Dreamer's block-GRU deterministic state transition."""

    def __init__(self, deter, stoch, act_dim, config):
        super().__init__()
        hidden = int(config.hidden_dim)
        self.blocks = int(config.blocks)
        self.dynlayers = int(config.dyn_layers)
        activation = getattr(nn, str(config.act))
        self._dyn_in0 = nn.Sequential(
            nn.Linear(deter, hidden),
            nn.RMSNorm(hidden, eps=1e-4, dtype=torch.float32),
            activation(),
        )
        self._dyn_in1 = nn.Sequential(
            nn.Linear(stoch, hidden),
            nn.RMSNorm(hidden, eps=1e-4, dtype=torch.float32),
            activation(),
        )
        self._dyn_in2 = nn.Sequential(
            nn.Linear(act_dim, hidden),
            nn.RMSNorm(hidden, eps=1e-4, dtype=torch.float32),
            activation(),
        )
        self._dyn_hid = nn.Sequential()
        in_dim = (3 * hidden + deter // self.blocks) * self.blocks
        for index in range(self.dynlayers):
            self._dyn_hid.add_module(f"dyn_hid_{index}", BlockLinear(in_dim, deter, self.blocks))
            self._dyn_hid.add_module(f"norm_{index}", nn.RMSNorm(deter, eps=1e-4, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{index}", activation())
            in_dim = deter
        self._dyn_gru = BlockLinear(in_dim, 3 * deter, self.blocks)

    def forward(self, stoch, deter, action):
        batch_size = action.shape[0]
        stoch = stoch.reshape(batch_size, -1)
        action = normalize_action(action)
        inputs = torch.cat([self._dyn_in0(deter), self._dyn_in1(stoch), self._dyn_in2(action)], dim=-1)
        inputs = inputs.unsqueeze(-2).expand(-1, self.blocks, -1)
        grouped_deter = deter.reshape(*deter.shape[:-1], self.blocks, -1)
        hidden = torch.cat([grouped_deter, inputs], dim=-1).reshape(*deter.shape[:-1], -1)
        gates = self._dyn_gru(self._dyn_hid(hidden)).reshape(*deter.shape[:-1], self.blocks, -1)
        reset, candidate, update = (chunk.reshape(*deter.shape[:-1], -1) for chunk in torch.chunk(gates, 3, dim=-1))
        reset = torch.sigmoid(reset)
        candidate = torch.tanh(reset * candidate)
        update = torch.sigmoid(update - 1)
        return update * candidate + (1 - update) * deter
