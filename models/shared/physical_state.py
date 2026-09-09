"""DMC physical targets and a readout trained only on detached world-model features."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

STATE_KEY = "physical_state"
TARGET_VERSION = "dmc_physical_state_v2"


class PhysicalStateTargets:
    """One label layout for stored observations, live observations, and decoded geometry."""

    def __init__(self, task, fields):
        self.task = str(task)
        position_dim = {
            "dmc_cartpole_balance_sparse": 3,
            "dmc_reacher_easy": 2,
            "dmc_ball_in_cup_catch": 4,
            "dmc_point_mass_easy": 2,
        }[self.task]
        self.raw_coordinates = [f"{key}[{index}]" for key, indices in fields.items() for index in indices]
        self.angles = {
            index
            for index, name in enumerate(self.raw_coordinates)
            if self.task == "dmc_reacher_easy" and name.startswith("position[")
        }
        self.coordinates = [
            coordinate
            for index, name in enumerate(self.raw_coordinates)
            for coordinate in ([f"cos({name})", f"sin({name})"] if index in self.angles else [name])
        ]
        positions = [f"position[{index}]" for index in range(position_dim)]
        if self.angles:
            positions = [coordinate for name in positions for coordinate in (f"cos({name})", f"sin({name})")]
        self.positions = [self.coordinates.index(name) for name in positions]
        velocity_dim = 2 if self.task == "dmc_cartpole_balance_sparse" else position_dim
        self.velocities = [self.coordinates.index(f"velocity[{index}]") for index in range(velocity_dim)]
        self.to_target = [self.coordinates.index(f"to_target[{index}]") for index in range(2)] if self.angles else []
        self.derived_coordinates = {
            "dmc_reacher_easy": ["fingertip_x", "fingertip_y", "fingertip_vx", "fingertip_vy"],
            "dmc_ball_in_cup_catch": ["ball_to_target_x", "ball_to_target_z", "ball_relative_vx", "ball_relative_vz"],
        }.get(self.task, [])

    def encode(self, state):
        state = np.asarray(state, dtype=np.float32)
        if not self.angles:
            return state
        return np.stack([
            value
            for index in range(len(self.raw_coordinates))
            for value in (
                (np.cos(state[..., index]), np.sin(state[..., index]))
                if index in self.angles else (state[..., index],)
            )
        ], axis=-1)

    def goal_relation(self, state):
        position = state[..., self.positions]
        if self.task == "dmc_cartpole_balance_sparse":
            cart, cosine, sine = position.unbind(-1)
            # A zero orientation vector is undefined, not a successful upright pole.
            valid = cosine.square() + sine.square() > 1e-12
            angle = torch.atan2(torch.where(valid, sine, 0), torch.where(valid, cosine, -1))
            return torch.stack((cart, angle), dim=-1)
        if self.task == "dmc_reacher_easy":
            return state[..., self.to_target]
        if self.task == "dmc_ball_in_cup_catch":
            # DMC qpos are offsets: cup target z=0.55+cup_z; ball z=0.2+ball_z.
            return position[..., :2] - position[..., 2:] + position.new_tensor([0, 0.35])
        return -position  # Point-mass target is fixed at the world origin.

    def derived(self, state):
        """Diagnostics in metres and metres/second, with no extra fitted outputs."""
        if self.task == "dmc_reacher_easy":
            c1, s1, c2, s2 = state[..., self.positions].unbind(-1)
            v1, v2 = state[..., self.velocities].unbind(-1)
            c12, s12 = c1 * c2 - s1 * s2, s1 * c2 + c1 * s2
            # Both joint-to-joint/fingertip offsets in reacher.xml are 0.12 m.
            return 0.12 * torch.stack(
                (c1 + c12, s1 + s12, -s1 * v1 - s12 * (v1 + v2), c1 * v1 + c12 * (v1 + v2)), dim=-1
            )
        if self.task == "dmc_ball_in_cup_catch":
            velocity = state[..., self.velocities]
            return torch.cat((self.goal_relation(state), velocity[..., 2:] - velocity[..., :2]), dim=-1)
        return state[..., :0]


class PhysicalStateHead(nn.Module):
    def __init__(self, feature_dim, config, *, history=1, tokens=1):
        super().__init__()
        self.history = int(history)
        if config.target_version != TARGET_VERSION:
            raise ValueError(f"Physical targets require {TARGET_VERSION}; start a fresh run with the current config.")
        self.targets = PhysicalStateTargets(config.task, config.fields)
        self.coordinates = self.targets.coordinates
        self.samples_per_update = int(config.samples_per_update)
        self.grad_clip = float(config.grad_clip)
        output_dim = len(self.coordinates)
        projection = int(config.projection_dim)
        hidden = int(config.hidden_dim)
        # Auxiliary initialization must not change the native model's RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.project = nn.Sequential(nn.Linear(feature_dim, projection), nn.SiLU())
            self.readout = nn.Sequential(
                nn.Linear(self.history * tokens * projection, hidden), nn.SiLU(), nn.Linear(hidden, output_dim)
            )
        self.register_buffer("mean", torch.zeros(output_dim))
        self.register_buffer("std", torch.ones(output_dim))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("examples", torch.zeros((), dtype=torch.long))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(config.lr))

    @torch.no_grad()
    def set_stats(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, device=self.mean.device))
        self.std.copy_(torch.as_tensor(std, device=self.std.device).clamp_min(1e-3))

    def forward(self, features):
        # [B,T,D] or [B,T,P,D]; preserve ordered patches and only complete causal histories.
        if features.ndim == 3:
            features = features.unsqueeze(-2)
        with torch.autocast(device_type=features.device.type, enabled=False):
            value = self.project(features.float()).flatten(-2)
            value = value.unfold(1, self.history, 1).transpose(-1, -2).flatten(-2)
            return self.readout(value) * self.std + self.mean

    def fit(self, features, targets):
        features = features.detach()
        targets = targets[:, self.history - 1 :].to(device=features.device, dtype=torch.float32)
        prediction = self(features)
        if prediction.shape != targets.shape:
            raise ValueError(f"Physical-state prediction {prediction.shape} does not match targets {targets.shape}.")
        prediction, targets = prediction.flatten(0, 1), targets.flatten(0, 1)
        count = min(self.samples_per_update, len(targets))
        indices = torch.linspace(0, len(targets) - 1, count, device=features.device).long()
        loss = F.mse_loss((prediction[indices] - self.mean) / self.std, (targets[indices] - self.mean) / self.std)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.updates.add_(1)
        self.examples.add_(count)
        return {
            "state/loss": loss.detach(),
            "state/updates": self.updates.item(),
            "state/examples": self.examples.item(),
        }
