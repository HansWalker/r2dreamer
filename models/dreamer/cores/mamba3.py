"""Mamba3 deterministic core for Dreamer's RSSM."""

import torch
from torch import nn

from models.shared.mamba3 import Mamba3Layer, active_compute_dtype
from models.shared.utils import normalize_action, weight_init_


class DreamerMambaCore(nn.Module):
    """Mamba3 replacement for Dreamer's deterministic recurrent core."""

    cache_keys = ("mamba_angle_state", "mamba_ssm_state", "mamba_k_state", "mamba_v_state")

    def __init__(self, deter, stoch, act_dim, config):
        super().__init__()
        if int(config.n_layers) != 1:
            raise ValueError("The Mamba3 RSSM currently supports n_layers=1 only.")
        self.deter = int(deter)
        self.token = nn.Linear(int(stoch) + int(act_dim), self.deter)
        self.input_norm = nn.RMSNorm(self.deter, eps=1e-4, dtype=torch.float32)
        self.output_norm = nn.RMSNorm(self.deter, eps=1e-4, dtype=torch.float32)
        self.layer = Mamba3Layer(
            d_model=self.deter,
            config=config,
            layer_idx=0,
        )
        weight_init_(self.token)

    def initial_context(self, batch_size, device=None, dtype=None):
        return self.layer.initial_context(batch_size, device=device, dtype=dtype)

    def forward(self, stoch, action, angle_state=None, ssm_state=None, k_state=None, v_state=None):
        batch_size = action.shape[0]
        stoch = stoch.reshape(batch_size, -1)
        action = normalize_action(action)
        token = self.token(torch.cat([stoch, action], dim=-1)).contiguous()
        mixer_input = self.input_norm(token).contiguous()
        if angle_state is None:
            angle_state, ssm_state, k_state, v_state = self.initial_context(
                batch_size,
                device=mixer_input.device,
                dtype=active_compute_dtype(mixer_input),
            )
        mixed, angle_state, ssm_state, k_state, v_state = self.layer.step(
            mixer_input,
            angle_state,
            ssm_state,
            k_state,
            v_state,
        )
        deter = self.output_norm(token + mixed)
        return deter, angle_state, ssm_state, k_state, v_state
