"""Task-relative latent planning shared by planning model families."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from models.shared.utils import parse_model_io


class LatentPlanner(nn.Module):
    goal_conditioned = True

    def __init__(
        self,
        config,
        model_io,
        predictor,
        encoder,
        action_encoder,
        *,
        goal_readout,
        projector=None,
        pred_projector=None,
        decoder=None,
    ):
        super().__init__()
        settings = config.jepa_model
        _, action_shape, action_kind = parse_model_io(model_io)
        if action_kind != "continuous":
            raise ValueError("Latent planning requires continuous actions.")
        self.action_dim = math.prod(action_shape)
        self.history_size = int(settings.history_size)
        self.sequence_length = self.history_size + 1
        self.grad_clip = float(settings.optim.grad_clip)
        self.planner = settings.planner
        goal = settings.goal
        self.goal_geometry = str(goal.geometry)
        if self.goal_geometry not in {"radial", "box"}:
            raise ValueError(f"Unknown goal geometry: {self.goal_geometry}")
        tolerance = torch.as_tensor(list(goal.tolerance), dtype=torch.float32).flatten()
        if tolerance.numel() not in (1, 2) or not torch.all(tolerance > 0):
            raise ValueError(f"Goal tolerance must contain one or two positive values, got {tolerance.tolist()}.")
        if self.goal_geometry == "box" and tolerance.numel() != 2:
            raise ValueError("Box goal geometry requires one tolerance per relation coordinate.")
        self.register_buffer("goal_tolerance", tolerance)
        self.goal_relation_weight = float(goal.relation_weight)
        self.goal_stable_steps = int(goal.stable_steps)
        self.goal_action_weight = float(goal.action_weight)
        if not 1 <= self.goal_stable_steps <= int(self.planner.horizon):
            raise ValueError("Goal stable_steps must be between one and the planning horizon.")

        self.encoder = encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.goal_readout = goal_readout
        self.projector = projector or nn.Identity()
        self.pred_projector = pred_projector or nn.Identity()
        self.decoder = decoder
        self._cem_mean = None
        self._gradient_actions = None

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, obs):
        return self.projector(self.encoder(obs))

    def predict(self, state, action):
        return self.pred_projector(self.predictor(state, self.action_encoder(action)))

    def pool(self, latent):
        return self.encoder.pool(latent)

    @staticmethod
    def replay_observation(history):
        return {key: value[:, -1] for key, value in history.items()}

    def representation_loss(self, obs, latent, action):
        raise NotImplementedError

    def optimizer_state_dict(self):
        return {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}

    def load_optimizer_state_dict(self, state):
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state[name])

    def update(self, batch):
        obs, action, *_ = batch
        obs = {key: value.to(self.device, non_blocking=True) for key, value in obs.items()}
        action = action.to(self.device, non_blocking=True)
        latent = self.encode(obs)
        loss, metrics = self.representation_loss(obs, latent, action)
        relation = batch[4] if len(batch) > 4 else None
        if relation is not None:
            target = relation.to(self.device, non_blocking=True) / self.goal_tolerance
            prediction = self.goal_readout(latent.detach())
            if prediction.shape != target.shape:
                raise ValueError(f"Goal readout has shape {prediction.shape}, expected {target.shape}.")
            relation_loss = F.mse_loss(prediction, target)
            loss = loss + self.goal_relation_weight * relation_loss
            metrics["goal_relation_loss"] = relation_loss

        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        goal_parameters = list(self.goal_readout.parameters())
        goal_ids = {id(parameter) for parameter in goal_parameters}
        model_parameters = [parameter for parameter in self.parameters() if id(parameter) not in goal_ids]
        grad_norm = torch.nn.utils.clip_grad_norm_(model_parameters, self.grad_clip)
        goal_grad_norm = torch.nn.utils.clip_grad_norm_(goal_parameters, self.grad_clip)
        for optimizer in self.optimizers.values():
            optimizer.step()

        output = {
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            **{name: float(value.detach()) for name, value in metrics.items()},
        }
        if relation is not None:
            output["goal_grad_norm"] = float(goal_grad_norm)
        return output

    def _rollout(self, history, past_action, candidates):
        batch, samples, horizon, _ = candidates.shape
        latent_shape = history.shape[2:]
        state = (
            history[:, None]
            .expand(-1, samples, -1, *latent_shape)
            .reshape(batch * samples, self.history_size, *latent_shape)
        )
        action_history = (
            past_action[:, None]
            .expand(-1, samples, -1, -1)
            .reshape(batch * samples, self.history_size - 1, self.action_dim)
        )
        candidates = candidates.reshape(batch * samples, horizon, self.action_dim)
        prediction = []
        for step in range(horizon):
            conditioned_action = torch.cat((action_history, candidates[:, step, None]), dim=1)
            next_state = self.predict(state, conditioned_action)[:, -1]
            prediction.append(next_state)
            state = torch.cat((state[:, 1:], next_state[:, None]), dim=1)
            action_history = conditioned_action[:, 1:]
        return torch.stack(prediction, dim=1).reshape(batch, samples, horizon, *latent_shape)

    def _goal_cost(self, history, past_action, candidates):
        prediction = self._rollout(history, past_action, candidates)
        relation = self.goal_readout(prediction[:, :, -self.goal_stable_steps :])
        if self.goal_geometry == "radial":
            outside = F.relu(relation.norm(dim=-1) - 1)
            cost = outside.square()
        else:
            outside = F.relu(relation.abs() - 1)
            cost = outside.square().sum(dim=-1)
        action_cost = candidates.square().mean(dim=(-1, -2))
        return cost.mean(dim=-1) + self.goal_action_weight * action_cost

    def _first_mask(self, first, batch):
        return (
            torch.ones(batch, dtype=torch.bool, device=self.device)
            if first is None
            else first.to(device=self.device, dtype=torch.bool)
        )

    @torch.no_grad()
    def _cem(self, history, past_action, deterministic, first):
        batch = next(iter(history.values())).shape[0]
        horizon = int(self.planner.horizon)
        samples = int(self.planner.samples)
        mean = torch.zeros(batch, horizon, self.action_dim, device=self.device)
        first = self._first_mask(first, batch)
        if self._cem_mean is not None and self._cem_mean.shape == mean.shape:
            shifted = torch.cat((self._cem_mean[:, 1:], mean[:, -1:]), dim=1)
            mean = torch.where(first[:, None, None], mean, shifted)
        std = torch.full_like(mean, float(self.planner.initial_std))
        latent = self.encode(history)
        for _ in range(int(self.planner.iterations)):
            noise = torch.randn(batch, samples, horizon, self.action_dim, device=self.device)
            actions = (mean[:, None] + std[:, None] * noise).clamp(-1, 1)
            cost = self._goal_cost(latent, past_action, actions)
            elite_index = cost.topk(int(self.planner.elites), dim=1, largest=False).indices
            elite = actions.gather(
                1,
                elite_index[:, :, None, None].expand(-1, -1, horizon, self.action_dim),
            )
            mean = elite.mean(dim=1)
            std = elite.std(dim=1).clamp(float(self.planner.min_std), float(self.planner.max_std))
        self._cem_mean = mean.detach()
        action = mean[:, 0]
        if not deterministic:
            action = action + std[:, 0] * torch.randn_like(action)
        return action.clamp(-1, 1)

    def _gradient_plan(self, history, past_action, deterministic, first):
        batch = next(iter(history.values())).shape[0]
        restarts = int(self.planner.samples)
        horizon = int(self.planner.horizon)
        with torch.no_grad():
            latent = self.encode(history)
            first = self._first_mask(first, batch)
            actions = torch.tanh(torch.randn(batch, restarts, horizon, self.action_dim, device=self.device))
            if self._gradient_actions is not None and self._gradient_actions.shape == actions.shape:
                shifted = torch.cat((self._gradient_actions[:, :, 1:], actions[:, :, -1:]), dim=2)
                actions = torch.where(first[:, None, None, None], actions, shifted)
            logits = nn.Parameter(torch.atanh(actions.clamp(-0.999, 0.999)))
        optimizer = torch.optim.Adam((logits,), lr=float(self.planner.lr))
        iterations = int(self.planner.iterations)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
        action_noise = float(self.planner.action_noise)
        with torch.enable_grad():
            for _ in range(iterations):
                cost = self._goal_cost(latent, past_action, logits.tanh())
                optimizer.zero_grad(set_to_none=True)
                logits.grad = torch.autograd.grad(cost.mean(), logits)[0]
                optimizer.step()
                scheduler.step()
                if action_noise:
                    with torch.no_grad():
                        noisy = (logits.tanh() + action_noise * torch.randn_like(logits)).clamp(-0.999, 0.999)
                        logits.copy_(torch.atanh(noisy))
        with torch.no_grad():
            actions = logits.tanh()
            self._gradient_actions = actions.detach()
            cost = self._goal_cost(latent, past_action, actions)
            best = cost.argmin(dim=1)
            action = actions[torch.arange(batch, device=self.device), best, 0]
            if not deterministic:
                action = action + action_noise * torch.randn_like(action)
            return action.clamp(-1, 1)

    def act(self, history, past_action, deterministic=False, first=None):
        was_training = self.training
        self.eval()
        try:
            history = {key: value.to(self.device) for key, value in history.items()}
            past_action = past_action.to(self.device)
            if str(self.planner.type) == "gradient":
                return self._gradient_plan(history, past_action, deterministic, first)
            return self._cem(history, past_action, deterministic, first)
        finally:
            self.train(was_training)
