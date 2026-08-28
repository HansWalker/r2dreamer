"""Shared Mamba3 recurrent integration."""

import math

import torch
from torch import nn
from torch.nn import functional as F

try:
    from mamba_ssm.modules.mamba3 import Mamba3
except Exception as exc:  # pragma: no cover - depends on the optional CUDA stack
    Mamba3 = None
    MAMBA3_IMPORT_ERROR = exc
else:
    MAMBA3_IMPORT_ERROR = None


def active_compute_dtype(token):
    if torch.is_autocast_enabled(token.device.type):
        return torch.get_autocast_dtype(token.device.type)
    return token.dtype


def _rms_norm(value, norm):
    normalized = value.float() * torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + norm.eps)
    return (normalized * norm.weight.float()).to(value.dtype)


def _rotary_step(q, k, angle_state, angle_projection, dt, q_bias, k_bias):
    """Differentiable SISO equivalent of Mamba3's rotary inference kernel."""
    q_dtype, k_dtype = q.dtype, k.dtype
    rotary_dim = 2 * angle_state.shape[-1]
    angle = angle_state + torch.tanh(angle_projection.float()) * dt.float().unsqueeze(-1) * math.pi
    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)

    q = q.float() + q_bias.float().unsqueeze(0)
    k = k.float() + k_bias.float().unsqueeze(0)
    q_rot = q[..., :rotary_dim].reshape(*q.shape[:-1], rotary_dim // 2, 2)
    k_rot = k[..., :rotary_dim].reshape(*k.shape[:-1], rotary_dim // 2, 2)
    q0, q1 = q_rot.unbind(dim=-1)
    k0, k1 = k_rot.unbind(dim=-1)
    q_rot = torch.stack((q0 * cos - q1 * sin, q0 * sin + q1 * cos), dim=-1).flatten(-2)
    k_rot = torch.stack((k0 * cos - k1 * sin, k0 * sin + k1 * cos), dim=-1).flatten(-2)

    if rotary_dim < q.shape[-1]:
        q_rot = torch.cat((q_rot, q[..., rotary_dim:]), dim=-1)
        k_rot = torch.cat((k_rot, k[..., rotary_dim:]), dim=-1)
    return q_rot.to(q_dtype), k_rot.to(k_dtype), angle


def _state_step(state, a, b, c, x, z, dt, previous_b, previous_x, trap, d):
    """Differentiable equivalent of the official fused Mamba3 state update."""
    state_dtype, output_dtype = state.dtype, x.dtype
    state = state.float()
    a = a.float()
    b = b.float()
    c = c.float()
    x = x.float()
    z = z.float()
    dt = dt.float()
    previous_b = previous_b.float()
    previous_x = previous_x.float()
    trap = trap.float()
    d = d.float()

    decay = torch.exp(a * dt)
    previous_weight = (1 - trap) * dt * decay
    current_weight = trap * dt
    current = torch.einsum("brnh,brns->bnhs", x.unsqueeze(1) * current_weight[:, None, :, None], b)
    previous = torch.einsum(
        "brnh,brns->bnhs",
        previous_x.unsqueeze(1) * previous_weight[:, None, :, None],
        previous_b,
    )
    next_state = state * decay[:, :, None, None] + current + previous

    output = torch.einsum("bnhs,brns->brnh", next_state, c)
    output = output + x.unsqueeze(1) * d[None, None, :, None]
    gate = z.unsqueeze(1)
    output = (output * F.silu(gate)).sum(dim=1)
    return output.to(output_dtype), next_state.to(state_dtype)


def _validate_step_config(d_model, config):
    d_state = int(config.d_state)
    expand = int(config.expand)
    headdim = int(config.headdim)
    is_mimo = bool(config.is_mimo)
    mimo_rank = int(config.mimo_rank)
    is_outproj_norm = bool(config.is_outproj_norm)
    if is_mimo:
        raise ValueError("Mamba3 step mode currently supports is_mimo=False only.")
    if mimo_rank != 1:
        raise ValueError("Mamba3 step mode currently supports mimo_rank=1 only.")
    if is_outproj_norm:
        raise ValueError("Mamba3 step mode currently supports is_outproj_norm=False only.")
    if d_state not in (32, 64, 128):
        raise ValueError(f"Mamba3 step mode requires d_state in [32, 64, 128], got d_state={d_state}.")
    if expand <= 0 or headdim < 64 or headdim % 64:
        raise ValueError(
            "Mamba3 step mode requires a positive expand and headdim that is a multiple of 64, "
            f"got expand={expand}, headdim={headdim}."
        )

    inner = d_model * expand
    if inner % headdim:
        raise ValueError(
            "Mamba3 requires d_model * expand to be divisible by headdim, "
            f"got d_model={d_model}, expand={expand}, headdim={headdim}."
        )
    nheads = inner // headdim
    if nheads % 8:
        raise ValueError(f"Mamba3 step mode requires the number of heads to be divisible by 8, got nheads={nheads}.")

    num_rope_angles = (d_state // 2) // 2
    in_proj_width = 2 * inner + 2 * d_state + 3 * nheads + num_rope_angles
    if in_proj_width % 8:
        raise ValueError(
            f"Mamba3 step mode requires its internal projection width to be divisible by 8, got width={in_proj_width}."
        )
    if Mamba3 is None:
        raise ImportError(
            "Mamba3 requires mamba_ssm.modules.mamba3.Mamba3 and its CUDA runtime dependencies."
        ) from MAMBA3_IMPORT_ERROR


class Mamba3Layer(nn.Module):
    """Mamba3 mixer with differentiable training and fused inference steps."""

    def __init__(self, d_model, config, layer_idx=0):
        super().__init__()
        d_model = int(d_model)
        _validate_step_config(d_model, config)
        self.mamba = Mamba3(
            d_model=d_model,
            d_state=int(config.d_state),
            expand=int(config.expand),
            headdim=int(config.headdim),
            is_mimo=bool(config.is_mimo),
            mimo_rank=int(config.mimo_rank),
            chunk_size=max(1, int(config.chunk_size)),
            is_outproj_norm=bool(config.is_outproj_norm),
            layer_idx=int(layer_idx),
        )

    def initial_context(self, batch_size, device=None, dtype=None):
        return self.mamba.allocate_inference_cache(
            batch_size,
            max_seqlen=0,
            device=device,
            dtype=dtype,
        )

    def forward(self, token):
        return self.mamba(token)

    def _project(self, token):
        projection = self.mamba.in_proj(token)
        z, x, b, c, delta, a, trap, angles = torch.split(
            projection,
            [
                self.mamba.d_inner,
                self.mamba.d_inner,
                self.mamba.d_state * self.mamba.num_bc_heads,
                self.mamba.d_state * self.mamba.num_bc_heads,
                self.mamba.nheads,
                self.mamba.nheads,
                self.mamba.nheads,
                self.mamba.num_rope_angles,
            ],
            dim=-1,
        )

        batch_size = token.shape[0]
        a = -F.softplus(a.float())
        a = torch.clamp(a, max=-self.mamba.A_floor)
        delta = F.softplus(delta.float() + self.mamba.dt_bias)
        trap = torch.sigmoid(trap.float())
        b = b.reshape(batch_size, 1, self.mamba.num_bc_heads, self.mamba.d_state)
        c = c.reshape(batch_size, 1, self.mamba.num_bc_heads, self.mamba.d_state)
        b = _rms_norm(b, self.mamba.B_norm).expand(-1, -1, self.mamba.nheads, -1)
        c = _rms_norm(c, self.mamba.C_norm).expand(-1, -1, self.mamba.nheads, -1)
        x = x.reshape(batch_size, self.mamba.nheads, self.mamba.headdim)
        z = z.reshape(batch_size, self.mamba.nheads, self.mamba.headdim)
        angles = angles.unsqueeze(1).expand(-1, self.mamba.nheads, -1)
        return z, x, b, c, delta, a, trap, angles

    def recurrent_step(self, token, angle_state, ssm_state, k_state, v_state):
        """Out-of-place PyTorch step used whenever gradients are enabled."""
        z, x, b, c, delta, a, trap, angles = self._project(token)
        c, b, next_angle = _rotary_step(
            c,
            b,
            angle_state,
            angles,
            delta,
            self.mamba.C_bias.permute(1, 0, 2),
            self.mamba.B_bias.permute(1, 0, 2),
        )
        output, next_ssm = _state_step(
            ssm_state,
            a,
            b,
            c,
            x,
            z,
            delta,
            k_state,
            v_state,
            trap,
            self.mamba.D,
        )
        output = self.mamba.out_proj(output.flatten(1).to(x.dtype))
        return output, next_angle.contiguous(), next_ssm.contiguous(), b.contiguous(), x.contiguous()

    def fast_step(self, token, angle_state, ssm_state, k_state, v_state):
        """Official in-place CUDA decode step for no-gradient inference."""
        cache_dtype = active_compute_dtype(token)
        angle_state = angle_state.to(device=token.device, dtype=torch.float32).contiguous()
        ssm_state = ssm_state.to(device=token.device, dtype=torch.float32).contiguous()
        k_state = k_state.to(device=token.device, dtype=cache_dtype).contiguous()
        v_state = v_state.to(device=token.device, dtype=cache_dtype).contiguous()
        return self.mamba.step(token, angle_state, ssm_state, k_state, v_state)

    def step(self, token, angle_state, ssm_state, k_state, v_state):
        if torch.is_grad_enabled():
            return self.recurrent_step(token, angle_state, ssm_state, k_state, v_state)
        return self.fast_step(token, angle_state, ssm_state, k_state, v_state)
