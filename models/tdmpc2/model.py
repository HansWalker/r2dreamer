"""Single-task TD-MPC2 for the shared DMC trainer."""

from __future__ import annotations

import math
from copy import deepcopy
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn

from models.shared.utils import parse_model_io
from models.shared.vision import channel_first, image_spec


def symlog(value):
    return torch.sign(value) * torch.log1p(value.abs())


def symexp(value):
    return torch.sign(value) * torch.expm1(value.abs())


class SimNorm(nn.Module):
    def __init__(self, group_dim):
        super().__init__()
        self.group_dim = int(group_dim)

    def forward(self, value):
        shape = value.shape
        value = value.reshape(*shape[:-1], -1, self.group_dim)
        return value.softmax(dim=-1).reshape(shape)


class NormedLinear(nn.Linear):
    def __init__(self, in_dim, out_dim, dropout=0.0, activation=None):
        super().__init__(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = activation or nn.Mish()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, value):
        return self.activation(self.norm(self.dropout(F.linear(value, self.weight, self.bias))))


def td_mlp(in_dim, hidden_dims, out_dim, *, final=None, dropout=0.0):
    dims = [in_dim, *hidden_dims, out_dim]
    layers = []
    for index, (left, right) in enumerate(pairwise(dims)):
        if index == len(dims) - 2:
            layers.append(NormedLinear(left, right, activation=final) if final else nn.Linear(left, right))
        else:
            layers.append(NormedLinear(left, right, dropout=dropout if index == 0 else 0.0))
    return nn.Sequential(*layers)


def _weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class RandomShift(nn.Module):
    def __init__(self, pad=3):
        super().__init__()
        self.pad = int(pad)

    def forward(self, value):
        if not self.training:
            return value
        batch, _, height, _ = value.shape
        value = F.pad(value, (self.pad,) * 4, mode="replicate")
        padded = height + 2 * self.pad
        eps = 1.0 / padded
        axis = torch.linspace(-1 + eps, 1 - eps, padded, device=value.device)[:height]
        x, y = torch.meshgrid(axis, axis, indexing="xy")
        grid = torch.stack((x, y), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, (batch, 1, 1, 2), device=value.device)
        grid = grid + shift * (2.0 / padded)
        return F.grid_sample(value, grid, padding_mode="zeros", align_corners=False)


class PixelEncoder(nn.Module):
    """TD-MPC2's original 64x64 pixel encoder."""

    def __init__(self, model_io, config, final):
        super().__init__()
        self.key, (height, width, channels) = image_spec(model_io)
        if (height, width) != (64, 64):
            raise ValueError(f"TD-MPC2 expects 64x64 images, got {height}x{width}.")
        feature_channels = int(config.num_channels)
        self.net = nn.Sequential(
            RandomShift(),
            nn.Conv2d(channels, feature_channels, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 3),
            nn.Flatten(),
            final,
        )
        self.out_dim = feature_channels * 4 * 4
        if self.out_dim != int(config.latent_dim):
            raise ValueError(
                f"TD-MPC2 pixel encoder outputs {self.out_dim} values, but latent_dim={config.latent_dim}."
            )

    def forward(self, obs):
        pixels, prefix = channel_first(obs[self.key])
        if obs[self.key].dtype == torch.uint8:
            pixels = pixels / 255.0
        output = self.net(pixels - 0.5)
        return output.reshape(*prefix, output.shape[-1])


class TDMPC2(nn.Module):
    def __init__(self, config, model_io):
        super().__init__()
        settings = config.tdmpc2_model
        _, action_shape, action_kind = parse_model_io(model_io)
        if action_kind != "continuous":
            raise ValueError("TD-MPC2 requires continuous actions.")
        self.action_dim = math.prod(action_shape)
        self.history_size = 1
        self.sequence_length = int(settings.horizon) + 1
        self.horizon = int(settings.horizon)
        self.latent_dim = int(settings.latent_dim)
        self.num_q = int(settings.num_q)
        self.num_bins = int(settings.num_bins)
        self.vmin = float(settings.vmin)
        self.vmax = float(settings.vmax)
        self.bin_size = (self.vmax - self.vmin) / (self.num_bins - 1)
        self.rho = float(settings.rho)
        self.gamma = float(settings.gamma)
        self.tau = float(settings.tau)
        self.entropy_coef = float(settings.entropy_coef)
        self.grad_clip = float(settings.grad_clip)
        self.consistency_coef = float(settings.consistency_coef)
        self.reward_coef = float(settings.reward_coef)
        self.value_coef = float(settings.value_coef)
        self.planner = settings.planner

        hidden = [int(settings.mlp_dim)] * 2
        latent_action = self.latent_dim + self.action_dim
        self.encoder = PixelEncoder(model_io, settings, SimNorm(int(settings.simnorm_dim)))
        self.dynamics = td_mlp(
            latent_action,
            hidden,
            self.latent_dim,
            final=SimNorm(int(settings.simnorm_dim)),
        )
        self.reward = td_mlp(latent_action, hidden, self.num_bins)
        self.policy = td_mlp(self.latent_dim, hidden, 2 * self.action_dim)
        self.qs = nn.ModuleList(
            td_mlp(latent_action, hidden, self.num_bins, dropout=float(settings.dropout)) for _ in range(self.num_q)
        )
        self.apply(_weight_init)
        nn.init.zeros_(self.reward[-1].weight)
        for q in self.qs:
            nn.init.zeros_(q[-1].weight)
        self.target_qs = deepcopy(self.qs).requires_grad_(False)
        self.target_qs.eval()
        self.register_buffer("log_std_min", torch.tensor(float(settings.log_std_min)))
        self.register_buffer(
            "log_std_range",
            torch.tensor(float(settings.log_std_max) - float(settings.log_std_min)),
        )
        self.register_buffer("q_scale", torch.ones(1))
        self._previous_mean = None

        self.model_optimizer = torch.optim.Adam(
            [
                {"params": self.encoder.parameters(), "lr": float(settings.lr) * float(settings.encoder_lr_scale)},
                {"params": self.dynamics.parameters()},
                {"params": self.reward.parameters()},
                {"params": self.qs.parameters()},
            ],
            lr=float(settings.lr),
        )
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(settings.lr), eps=1e-5)

    @property
    def device(self):
        return next(self.parameters()).device

    def train(self, mode=True):
        super().train(mode)
        self.target_qs.eval()
        return self

    def optimizer_state_dict(self):
        return {
            "model": self.model_optimizer.state_dict(),
            "policy": self.policy_optimizer.state_dict(),
        }

    def load_optimizer_state_dict(self, state):
        self.model_optimizer.load_state_dict(state["model"])
        self.policy_optimizer.load_state_dict(state["policy"])

    def _two_hot(self, target):
        target = symlog(target).clamp(self.vmin, self.vmax)
        position = (target - self.vmin) / self.bin_size
        lower = position.floor().long().clamp(0, self.num_bins - 1)
        upper = (lower + 1).clamp(max=self.num_bins - 1)
        upper_weight = position - lower
        target_prob = torch.zeros(*target.shape[:-1], self.num_bins, device=target.device, dtype=target.dtype)
        target_prob.scatter_add_(-1, lower, 1 - upper_weight)
        target_prob.scatter_add_(-1, upper, upper_weight)
        return target_prob

    def _soft_ce(self, prediction, target):
        return -(self._two_hot(target) * prediction.log_softmax(dim=-1)).sum(dim=-1)

    def _decode(self, prediction):
        bins = torch.linspace(
            self.vmin,
            self.vmax,
            self.num_bins,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        return symexp((prediction.softmax(dim=-1) * bins).sum(dim=-1, keepdim=True))

    def _policy(self, latent, deterministic=False):
        mean, log_std = self.policy(latent).chunk(2, dim=-1)
        log_std = self.log_std_min + 0.5 * self.log_std_range * (log_std.tanh() + 1)
        noise = torch.zeros_like(mean) if deterministic else torch.randn_like(mean)
        raw_action = mean + noise * log_std.exp()
        action = raw_action.tanh()
        log_prob = (-0.5 * noise.square() - log_std - 0.9189385332).sum(-1, keepdim=True)
        scaled_entropy = -log_prob * self.action_dim
        log_prob -= torch.log(F.relu(1 - action.square()) + 1e-6).sum(-1, keepdim=True)
        return action, {
            "mean": mean.tanh(),
            "entropy": -log_prob,
            "scaled_entropy": scaled_entropy,
        }

    def _q_logits(self, latent, action, *, target=False):
        value = torch.cat((latent, action), dim=-1)
        networks = self.target_qs if target else self.qs
        return torch.stack([network(value) for network in networks], dim=0)

    def _q_value(self, latent, action, *, target=False, average=False):
        values = self._decode(self._q_logits(latent, action, target=target))
        index = torch.randperm(self.num_q, device=latent.device)[:2]
        selected = values[index]
        return selected.mean(0) if average else selected.min(0).values

    @torch.no_grad()
    def _td_target(self, next_latent, reward, terminal):
        next_action, _ = self._policy(next_latent)
        return reward + self.gamma * (1 - terminal) * self._q_value(next_latent, next_action, target=True)

    def _update_policy(self, latent):
        self.qs.requires_grad_(False)
        try:
            action, info = self._policy(latent)
            value = self._q_value(latent, action, average=True)
            with torch.no_grad():
                percentiles = torch.quantile(value[:, 0].flatten(), torch.tensor([0.05, 0.95], device=value.device))
                self.q_scale.lerp_((percentiles[1] - percentiles[0]).clamp(min=1), self.tau)
            weight = self.rho ** torch.arange(value.shape[1], device=value.device)
            objective = self.entropy_coef * info["scaled_entropy"] + value / self.q_scale
            policy_loss = -(objective.mean(dim=(0, 2)) * weight).mean()
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            self.policy_optimizer.step()
            return policy_loss.detach(), info["entropy"].mean().detach()
        finally:
            self.qs.requires_grad_(True)

    def update(self, batch):
        obs, action, reward, terminal, *_ = batch
        obs = {key: value.to(self.device, non_blocking=True) for key, value in obs.items()}
        action = action.to(self.device, non_blocking=True)
        reward = reward.to(self.device, non_blocking=True)
        terminal = terminal.to(self.device, non_blocking=True)
        horizon = action.shape[1]
        weight = self.rho ** torch.arange(horizon, device=self.device)

        with torch.no_grad():
            next_latent = self.encoder({key: value[:, 1:] for key, value in obs.items()})
            target = self._td_target(next_latent, reward, terminal)

        latent = self.encoder({key: value[:, 0] for key, value in obs.items()})
        rollout = [latent]
        consistency = []
        for step in range(horizon):
            latent = self.dynamics(torch.cat((latent, action[:, step]), dim=-1))
            consistency.append(F.mse_loss(latent, next_latent[:, step]))
            rollout.append(latent)
        rollout = torch.stack(rollout, dim=1)
        current = rollout[:, :-1]
        reward_logits = self.reward(torch.cat((current, action), dim=-1))
        q_logits = self._q_logits(current, action)
        consistency_loss = (torch.stack(consistency) * weight).sum() / horizon
        reward_loss = (self._soft_ce(reward_logits, reward) * weight).mean()
        value_loss = (self._soft_ce(q_logits, target.unsqueeze(0)) * weight).mean()
        loss = self.consistency_coef * consistency_loss + self.reward_coef * reward_loss + self.value_coef * value_loss

        self.model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [*self.encoder.parameters(), *self.dynamics.parameters(), *self.reward.parameters(), *self.qs.parameters()],
            self.grad_clip,
        )
        self.model_optimizer.step()
        policy_loss, entropy = self._update_policy(rollout.detach())
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_qs.parameters(), self.qs.parameters(), strict=True):
                target_parameter.lerp_(parameter, self.tau)

        return {
            "loss": float(loss.detach()),
            "consistency_loss": float(consistency_loss.detach()),
            "reward_loss": float(reward_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "policy_loss": float(policy_loss),
            "entropy": float(entropy),
            "grad_norm": float(grad_norm),
        }

    @torch.no_grad()
    def _estimate(self, latent, actions):
        batch, samples, horizon, _ = actions.shape
        state = latent[:, None].expand(-1, samples, -1).reshape(batch * samples, self.latent_dim)
        actions = actions.reshape(batch * samples, horizon, self.action_dim)
        value = torch.zeros(batch * samples, 1, device=self.device)
        discount = 1.0
        for step in range(horizon):
            action = actions[:, step]
            value += discount * self._decode(self.reward(torch.cat((state, action), dim=-1)))
            state = self.dynamics(torch.cat((state, action), dim=-1))
            discount *= self.gamma
        action, _ = self._policy(state)
        value += discount * self._q_value(state, action, average=True)
        return value.reshape(batch, samples)

    @torch.no_grad()
    def act(self, history, past_action, deterministic=False, first=None):
        was_training = self.training
        self.eval()
        try:
            history = {key: value.to(self.device) for key, value in history.items()}
            latent = self.encoder({key: value[:, -1] for key, value in history.items()})
            batch = latent.shape[0]
            horizon = int(self.planner.horizon)
            samples = int(self.planner.samples)
            policy_samples = min(int(self.planner.policy_samples), samples)
            mean = torch.zeros(batch, horizon, self.action_dim, device=self.device)
            std = torch.full_like(mean, float(self.planner.max_std))
            if self._previous_mean is not None and self._previous_mean.shape == mean.shape:
                shifted = torch.cat((self._previous_mean[:, 1:], torch.zeros_like(mean[:, :1])), dim=1)
                if first is None:
                    mean.copy_(shifted)
                else:
                    mean.copy_(torch.where(first[:, None, None].to(self.device), mean, shifted))

            policy_actions = torch.empty(batch, policy_samples, horizon, self.action_dim, device=self.device)
            state = latent[:, None].expand(-1, policy_samples, -1).reshape(-1, self.latent_dim)
            for step in range(horizon):
                action, _ = self._policy(state)
                policy_actions[:, :, step] = action.reshape(batch, policy_samples, self.action_dim)
                state = self.dynamics(torch.cat((state, action), dim=-1))

            for _ in range(int(self.planner.iterations)):
                noise = torch.randn(batch, samples, horizon, self.action_dim, device=self.device)
                actions = (mean[:, None] + std[:, None] * noise).clamp(-1, 1)
                actions[:, :policy_samples] = policy_actions
                score = self._estimate(latent, actions)
                elite_score, elite_index = score.topk(int(self.planner.elites), dim=1)
                elite = actions.gather(
                    1,
                    elite_index[:, :, None, None].expand(-1, -1, horizon, self.action_dim),
                )
                weights = torch.softmax(float(self.planner.temperature) * elite_score, dim=1)
                mean = (weights[:, :, None, None] * elite).sum(dim=1)
                variance = (weights[:, :, None, None] * (elite - mean[:, None]).square()).sum(dim=1)
                std = variance.sqrt().clamp(float(self.planner.min_std), float(self.planner.max_std))

            selected = torch.multinomial(weights, 1)
            trajectory = elite[torch.arange(batch, device=self.device), selected[:, 0]]
            action = trajectory[:, 0]
            if not deterministic:
                action = action + std[:, 0] * torch.randn_like(action)
            self._previous_mean = mean.detach()
            return action.clamp(-1, 1)
        finally:
            self.train(was_training)
