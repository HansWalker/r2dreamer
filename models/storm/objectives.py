"""Losses, returns, and policy distributions for native STORM training."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, OneHotCategorical

from models.shared.distributions import symexp, symlog


def optimize(module, optimizer, scaler, loss, grad_clip):
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(module.parameters(), max_norm=grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


class SymLogTwoHotLoss(nn.Module):
    def __init__(self, num_classes: int = 255, lower_bound: float = -20.0, upper_bound: float = 20.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.register_buffer(
            "bins",
            torch.linspace(self.lower_bound, self.upper_bound, self.num_classes),
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = symlog(target.to(dtype=torch.float32)).squeeze(-1)
        target = target.clamp(self.lower_bound, self.upper_bound)
        below = torch.sum(self.bins <= target.unsqueeze(-1), dim=-1) - 1
        above = torch.sum(self.bins < target.unsqueeze(-1), dim=-1)
        below = below.clamp(0, self.num_classes - 1)
        above = above.clamp(0, self.num_classes - 1)
        equal = below == above
        dist_below = torch.where(equal, torch.ones_like(target), (self.bins[below] - target).abs())
        dist_above = torch.where(equal, torch.ones_like(target), (self.bins[above] - target).abs())
        total = dist_below + dist_above
        weight_below = dist_above / total
        weight_above = dist_below / total
        target_prob = F.one_hot(below, self.num_classes).to(logits.dtype) * weight_below.unsqueeze(-1)
        target_prob = target_prob + F.one_hot(above, self.num_classes).to(logits.dtype) * weight_above.unsqueeze(-1)
        return -(target_prob * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        return symexp(F.softmax(logits.to(dtype=torch.float32), dim=-1) @ self.bins).unsqueeze(-1)


class CategoricalKLLossWithFreeBits(nn.Module):
    def __init__(self, free_bits: float = 1.0):
        super().__init__()
        self.free_bits = float(free_bits)

    def forward(self, p_logits: torch.Tensor, q_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).sum(dim=-1).mean()
        return torch.maximum(kl, torch.ones_like(kl) * self.free_bits), kl


class TanhNormal:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std
        self.normal = Normal(mean, std)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.normal.rsample()
        action = torch.tanh(raw)
        return action, self.log_prob(action, raw)

    def mode(self) -> torch.Tensor:
        return torch.tanh(self.mean)

    def log_prob(self, action: torch.Tensor, raw: torch.Tensor | None = None) -> torch.Tensor:
        eps = torch.finfo(action.dtype).eps
        action = action.clamp(-1 + 1e-6, 1 - 1e-6)
        if raw is None:
            raw = 0.5 * (torch.log1p(action) - torch.log1p(-action))
        log_prob = self.normal.log_prob(raw) - torch.log(1 - action.pow(2) + eps)
        return log_prob.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self.normal.entropy().sum(dim=-1)


def percentile(x: torch.Tensor, percentage: float) -> torch.Tensor:
    flat = x.detach().reshape(-1)
    kth = max(1, int(float(percentage) * flat.numel()))
    return flat.sort().values[kth - 1]


def lambda_return(reward, value, termination, gamma: float, lambd: float) -> torch.Tensor:
    termination = termination.squeeze(-1) if termination.shape[-1] == 1 else termination
    reward = reward.squeeze(-1) if reward.shape[-1] == 1 else reward
    value = value.squeeze(-1) if value.shape[-1] == 1 else value
    cont = 1.0 - termination.to(value.dtype)
    returns = torch.zeros_like(value)
    returns[:, -1] = value[:, -1]
    for idx in reversed(range(reward.shape[1])):
        returns[:, idx] = reward[:, idx] + gamma * cont[:, idx] * (
            (1 - lambd) * value[:, idx] + lambd * returns[:, idx + 1]
        )
    return returns[:, :-1].unsqueeze(-1)


class EMAScalar:
    def __init__(self, decay: float = 0.99):
        self.decay = float(decay)
        self.value = None

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        value = value.detach()
        self.value = value if self.value is None else self.decay * self.value + (1 - self.decay) * value
        return self.value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state.get("value")
