"""STORM actor, critic, and behavior-learning updates."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
from torch import nn

from .objectives import (
    EMAScalar,
    SymLogTwoHotLoss,
    TanhNormal,
    lambda_return,
    optimize,
    percentile,
)


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    layers: int,
    act: str,
) -> nn.Sequential:
    activation = getattr(nn, act)
    modules: list[nn.Module] = []
    for _ in range(int(layers)):
        modules.extend([
            nn.Linear(int(input_dim), int(hidden_dim), bias=False),
            nn.LayerNorm(int(hidden_dim)),
            activation(inplace=True) if activation is nn.ReLU else activation(),
        ])
        input_dim = int(hidden_dim)
    modules.append(nn.Linear(int(input_dim), int(output_dim), bias=True))
    return nn.Sequential(*modules)


class ActorCriticAgent(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, config) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.gamma = float(config.gamma)
        self.lambd = float(config.lambd)
        self.entropy_coef = float(config.entropy_coef)
        self.slow_critic_decay = float(config.slow_critic_decay)
        self.use_amp = bool(config.use_amp)
        self.amp_dtype = torch.bfloat16 if str(config.amp_dtype) == "bfloat16" else torch.float16
        self.grad_clip = float(config.grad_clip)

        hidden_dim = int(config.hidden_dim)
        layers = int(config.layers)
        act = str(config.act)
        value_bins = int(config.value_bins)
        self.actor = _build_mlp(feat_dim, 2 * self.action_dim, hidden_dim, layers, act)
        self.critic = _build_mlp(feat_dim, value_bins, hidden_dim, layers, act)
        self.slow_critic = copy.deepcopy(self.critic)
        self.slow_critic.requires_grad_(False)
        self.slow_critic.eval()
        self.twohot = SymLogTwoHotLoss(value_bins, -20, 20)
        self.lowerbound_ema = EMAScalar(0.99)
        self.upperbound_ema = EMAScalar(0.99)

        self.optimizer = torch.optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters()],
            lr=float(config.lr),
            eps=float(config.eps),
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and torch.cuda.is_available(),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def train(self, mode: bool = True):
        super().train(mode)
        self.slow_critic.eval()
        return self

    def _amp(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp and self.device.type == "cuda",
        )

    def training_state_dict(self):
        return {
            "lowerbound_ema": self.lowerbound_ema.state_dict(),
            "upperbound_ema": self.upperbound_ema.state_dict(),
        }

    def load_training_state_dict(self, state):
        if not state:
            return
        self.lowerbound_ema.load_state_dict(state["lowerbound_ema"])
        self.upperbound_ema.load_state_dict(state["upperbound_ema"])
        for ema in (self.lowerbound_ema, self.upperbound_ema):
            if ema.value is not None:
                ema.value = ema.value.to(self.device)

    def dist(self, feature: torch.Tensor) -> TanhNormal:
        mean, log_std = torch.chunk(self.actor(feature), 2, dim=-1)
        log_std = log_std.clamp(-5.0, 2.0)
        return TanhNormal(mean.to(torch.float32), torch.exp(log_std).to(torch.float32))

    @torch.no_grad()
    def slow_value(self, feature: torch.Tensor) -> torch.Tensor:
        return self.twohot.decode(self.slow_critic(feature))

    @torch.no_grad()
    def sample(self, feature: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.eval()
        with self._amp():
            dist = self.dist(feature)
            if deterministic:
                return dist.mode(), None
            return dist.sample()

    @torch.no_grad()
    def update_slow_critic(self, decay: float | None = None):
        decay = float(decay if decay is not None else self.slow_critic_decay)
        for slow, parameter in zip(self.slow_critic.parameters(), self.critic.parameters(), strict=True):
            slow.data.copy_(slow.data * decay + parameter.data * (1 - decay))

    def update(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        termination: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        self.train()
        with self._amp():
            dist = self.dist(latent[:, :-1])
            raw_value = self.critic(latent)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            value = self.twohot.decode(raw_value)
            slow_value = self.slow_value(latent)
            returns = lambda_return(reward, value, termination, self.gamma, self.lambd)
            slow_returns = lambda_return(reward, slow_value, termination, self.gamma, self.lambd)

            value_loss = self.twohot(raw_value[:, :-1], returns.detach())
            slow_value_loss = self.twohot(raw_value[:, :-1], slow_returns.detach())
            low = self.lowerbound_ema(percentile(returns, 0.05))
            high = self.upperbound_ema(percentile(returns, 0.95))
            scale = torch.maximum(torch.ones_like(high), high - low)
            advantage = (returns - value[:, :-1]) / scale
            policy_loss = -(log_prob * advantage.detach().squeeze(-1)).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + value_loss + slow_value_loss - self.entropy_coef * entropy_loss

        optimize(self, self.optimizer, self.scaler, loss, self.grad_clip)
        self.update_slow_critic()
        return {
            "ac/loss": loss.detach(),
            "ac/policy": policy_loss.detach(),
            "ac/value": value_loss.detach(),
            "ac/slow_value": slow_value_loss.detach(),
            "ac/entropy": entropy_loss.detach(),
            "ac/return": returns.mean().detach(),
        }

    def update_expert(
        self,
        feature: torch.Tensor,
        action: torch.Tensor,
        returns: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        self.train()
        with self._amp():
            feature = feature.detach()
            dist = self.dist(feature)
            bc_loss = -dist.log_prob(action.to(torch.float32)).mean()
            value_loss = self.twohot(self.critic(feature), returns)
            loss = bc_loss + value_loss

        optimize(self, self.optimizer, self.scaler, loss, self.grad_clip)
        self.update_slow_critic()
        return {
            "expert/bc": bc_loss.detach(),
            "expert/value": value_loss.detach(),
            "expert/ac_loss": loss.detach(),
        }
