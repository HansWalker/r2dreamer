"""Native latent-goal planning with an independent physical-state evaluation head."""

import math

import torch
from torch import nn

from models.shared.physical_state import STATE_KEY, PhysicalStateHead, readout_mode
from models.shared.latent_goal import latent_goal_cost
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
        projector=None,
        pred_projector=None,
    ):
        super().__init__()
        settings = config.jepa_model
        _, action_shape, action_kind = parse_model_io(model_io)
        if action_kind != "continuous":
            raise ValueError("Latent planning requires continuous actions.")
        self.action_dim = math.prod(action_shape)
        self.history_size = int(settings.history_size)
        self.training_horizon = int(settings.get("training_horizon", 1))
        if self.training_horizon < 1:
            raise ValueError("training_horizon must be positive.")
        self.sequence_length = self.history_size + self.training_horizon
        self.grad_clip = float(settings.optim.grad_clip)
        self.planner = settings.planner
        self.use_amp = bool(settings.use_amp)
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
        self.goal_reduction = "sum" if str(config.model_family) == "leworldmodel" else "mean"

        self.encoder = encoder
        self.model_family = str(config.model_family)
        self._aggregate_goal_weight()  # Reject invalid opt-in configs before fitting.
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.state_head = PhysicalStateHead(
            encoder.out_dim, config.state_head, history=self.history_size, tokens=getattr(encoder, "num_tokens", 1)
        )
        self.projector = projector or nn.Identity()
        self.pred_projector = pred_projector or nn.Identity()
        self.decoder = None
        self._cem_mean = None
        self._gradient_actions = None
        self._adaptation_mode = "native"
        self._frozen_parameters = []
        self._gradient_updates = 0
        self._clipped_updates = 0

    def set_adaptation_mode(self, mode):
        """Diagnostic ablations only; production retains native parameter/statistic updates."""
        if mode not in {"native", "frozen_bn", "frozen_encoder"}:
            raise ValueError(f"Unknown adaptation mode: {mode}")
        for parameter, trainable in self._frozen_parameters:
            parameter.requires_grad_(trainable)
        self._frozen_parameters = []
        self._adaptation_mode = mode
        if mode == "frozen_encoder":
            for module in (self.encoder, self.projector):
                for parameter in module.parameters():
                    self._frozen_parameters.append((parameter, parameter.requires_grad))
                    parameter.requires_grad_(False)
                    parameter.grad = None
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "_adaptation_mode", "native") != "native":
            if self._adaptation_mode == "frozen_encoder":
                self.encoder.eval()
                self.projector.eval()
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, obs):
        with self.amp_context():
            return self.projector(self.encoder(obs)).float()

    def predict(self, state, action):
        with self.amp_context():
            return self.pred_projector(self.predictor(state, self.action_encoder(action))).float()

    def amp_context(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp and self.device.type == "cuda")

    @staticmethod
    def replay_observation(history):
        return {"image": history["image"][:, -1]}

    def representation_loss(self, obs, latent, action):
        raise NotImplementedError

    def training_predictions(self, latent, action):
        """Keep the native one-step recipe, or train the planner's recursive rollout.

        Future real encodings supply targets only. Predictions remain attached so
        later errors teach earlier steps, with the same fixed context as planning.
        """
        if latent.shape[1] != self.sequence_length or action.shape[1] != self.sequence_length - 1:
            raise ValueError("Training windows must contain history_size + training_horizon frames and aligned actions.")
        if self.training_horizon == 1:
            return self.predict(latent[:, :-1], action), latent[:, 1:]
        history = self.history_size
        prediction = self.rollout(
            latent[:, :history], action[:, :history - 1], action[:, None, history - 1:],
        )[:, 0]
        return prediction, latent[:, history:]

    def prediction_metrics(self, prediction, target):
        if self.training_horizon == 1:
            return {}
        errors = (prediction.detach() - target.detach()).square().flatten(2).mean(dim=(0, 2))
        return {f"prediction_step_{step + 1}": value for step, value in enumerate(errors)}

    def optimizer_state_dict(self):
        return {
            **{name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
            "state_head": self.state_head.optimizer.state_dict(),
        }

    def load_optimizer_state_dict(self, state):
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state[name])
        self.state_head.load_optimizer_state_dict(state["state_head"])

    def update(self, batch, *, readout_batch=None):
        obs, action, *_ = batch
        obs = {key: value.to(self.device, non_blocking=True) for key, value in obs.items()}
        labels = obs.pop(STATE_KEY)
        action = action.to(self.device, non_blocking=True)
        latent = self.encode(obs)
        loss, metrics = self.representation_loss(obs, latent, action)
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        decoder_parameters = list(self.decoder.parameters()) if self.decoder is not None else []
        readout_ids = {id(parameter) for parameter in [*self.state_head.parameters(), *decoder_parameters]}
        model_parameters = [parameter for parameter in self.parameters() if id(parameter) not in readout_ids]
        grad_norm = torch.nn.utils.clip_grad_norm_(model_parameters, self.grad_clip, error_if_nonfinite=True)
        grad_norm = float(grad_norm)
        clipped = grad_norm > self.grad_clip
        # Detached visualization losses must not rescale the representation gradients.
        if decoder_parameters:
            torch.nn.utils.clip_grad_norm_(decoder_parameters, self.grad_clip, error_if_nonfinite=True)
        for optimizer in self.optimizers.values():
            optimizer.step()
        self._gradient_updates += 1
        self._clipped_updates += int(clipped)

        feature, labels = self.readout_features(batch if readout_batch is None else readout_batch)
        return {
            "loss": float(loss.detach()),
            "grad_norm": grad_norm,
            "grad_clipped": float(clipped),
            "grad_clip_fraction_since_load": self._clipped_updates / self._gradient_updates,
            **{name: float(value.detach()) for name, value in metrics.items()},
            **self.state_head.fit(feature, labels),
        }

    def readout_features(self, batch):
        obs = {key: value.to(self.device, non_blocking=True) for key, value in batch[0].items()}
        labels = obs.pop(STATE_KEY)
        with readout_mode(self):
            return self.encode(obs), labels

    def _predict_rollout(self, state, action, conditioning):
        return self.predictor(state, action), None

    def rollout(self, history, past_action, candidates):
        """Predict action candidates [batch, samples, horizon, action] from encoded history."""
        # Autocast only the neural rollout, not goal costs or the action optimizer.
        with self.amp_context():
            return self._rollout(history, past_action, candidates).float()

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
        # Actions are known for the entire rollout; only latent states are recursive.
        encoded_action = self.action_encoder(torch.cat((action_history, candidates), dim=1))
        prediction = []
        conditioning = None
        for step in range(horizon):
            conditioned_action = encoded_action[:, step : step + self.history_size]
            output, conditioning = self._predict_rollout(state, conditioned_action, conditioning)
            next_state = self.pred_projector(output)[:, -1]
            prediction.append(next_state)
            state = torch.cat((state[:, 1:], next_state[:, None]), dim=1)
        return torch.stack(prediction, dim=1).reshape(batch, samples, horizon, *latent_shape)

    def _goal_cost(self, history, past_action, candidates, goal):
        prediction = self.rollout(history, past_action, candidates)
        return self.planning_cost(prediction, goal, history=history)

    def _aggregate_goal_weight(self):
        weight = float(self.planner.get("aggregate_goal_weight", 0.))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("aggregate_goal_weight must be finite and nonnegative.")
        if weight and (self.model_family != "temporal_straightening"
                       or not callable(getattr(self.encoder, "agg", None))
                       or not hasattr(self.encoder, "agg_mlp")
                       or not hasattr(self.encoder, "agg_post_norm")):
            raise ValueError("A nonzero aggregate_goal_weight requires the existing TS aggregation head; load a trained agg checkpoint.")
        return weight

    def planning_cost(self, prediction, goal, *, history=None):
        """Native spatial cost plus an optional existing TS aggregate-head cost.

        The zero/absent weight uses the exact legacy scorer. The opt-in term pools
        each state before applying the same coordinate/time reductions. Multiple
        goal alternatives remain coherent across both terms. This never creates
        or fits a head; callers enabling it must load a trained aggregation head.
        """
        weight = self._aggregate_goal_weight()
        settings = {"reduction": self.goal_reduction, "mode": self.planner.get("objective", "last"),
                    "tail_steps": int(self.planner.get("tail_steps", 3))}
        if not weight:
            return latent_goal_cost(prediction, goal, history=history, **settings)
        spatial = latent_goal_cost(prediction, goal, history=history, return_per_goal=True, **settings)
        if prediction.ndim != 5:
            raise ValueError("Aggregate goal scoring requires TS spatial states [batch, candidates, time, patches, channels].")
        goals = goal[:, None] if goal.ndim == prediction.ndim - 2 else goal

        def aggregate(value):
            pooled = self.encoder.agg(value.float().reshape(-1, *value.shape[-2:]))
            return pooled.reshape(*value.shape[:-2], pooled.shape[-1])

        # Native TS curvature trains this head in FP32; costs and action gradients
        # retain that precision independently of the neural rollout's autocast.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            pooled_prediction = aggregate(prediction)
            with torch.no_grad():
                pooled_goal = aggregate(goals)
                pooled_history = aggregate(history) if history is not None else None
            pooled = latent_goal_cost(pooled_prediction, pooled_goal, history=pooled_history,
                                      return_per_goal=True, **settings)
        return (spatial + weight * pooled).min(-1).values

    def _first_mask(self, first, batch):
        return (
            torch.ones(batch, dtype=torch.bool, device=self.device)
            if first is None
            else first.to(device=self.device, dtype=torch.bool)
        )

    @torch.no_grad()
    def _cem(self, history, past_action, deterministic, first, goal):
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
            cost = self._goal_cost(latent, past_action, actions, goal)
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

    def _gradient_plan(self, history, past_action, deterministic, first, goal):
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
        candidates = batch * restarts
        batch_size = int(self.planner.gradient_batch_size)
        chunks = []
        for start in range(0, candidates, batch_size):
            stop = min(start + batch_size, candidates)
            indices = torch.arange(start, stop, device=self.device) // restarts
            chunks.append((start, stop, latent[indices], past_action[indices], goal[indices]))
        with torch.enable_grad():
            for _ in range(iterations):
                optimizer.zero_grad(set_to_none=True)
                gradient = torch.empty_like(logits).flatten(0, 1)
                for start, stop, chunk_latent, chunk_past, chunk_goal in chunks:
                    chunk = logits.flatten(0, 1)[start:stop].detach().requires_grad_()
                    cost = self._goal_cost(chunk_latent, chunk_past, chunk.tanh()[:, None], chunk_goal)
                    # Each candidate has independent action variables. Averaging across
                    # environments/restarts changes Adam's effective epsilon and step.
                    gradient[start:stop] = torch.autograd.grad(cost.sum(), chunk)[0]
                logits.grad = gradient.view_as(logits)
                optimizer.step()
                scheduler.step()
                if action_noise:
                    with torch.no_grad():
                        noisy = (logits.tanh() + action_noise * torch.randn_like(logits)).clamp(-0.999, 0.999)
                        logits.copy_(torch.atanh(noisy))
        with torch.no_grad():
            actions = logits.tanh()
            self._gradient_actions = actions.detach()
            cost = self._goal_cost(latent, past_action, actions, goal)
            best = cost.argmin(dim=1)
            action = actions[torch.arange(batch, device=self.device), best, 0]
            if not deterministic:
                action = action + action_noise * torch.randn_like(action)
            return action.clamp(-1, 1)

    def act(self, history, past_action, deterministic=False, first=None):
        if "goal_image" not in history and "goal_images" not in history:
            raise ValueError(
                "Latent planning requires a rendered physical goal (goal_image). "
                "Use the physical_render_v1 environment config; the evaluation head is not a controller."
            )
        was_training = self.training
        self.eval()
        gradient_planner = str(self.planner.type) == "gradient"
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad] if gradient_planner else []
        # Retain action gradients without saving tensors for unused weight gradients.
        for parameter in trainable:
            parameter.requires_grad_(False)
        try:
            with torch.no_grad():
                # Re-encode with current weights; never keep stale goals across online updates.
                if "goal_images" in history:
                    goal = self.encode({"image": history["goal_images"].to(self.device)})
                else:
                    goal = self.encode({"image": history["goal_image"].to(self.device)[:, None]})[:, 0]
            history = {key: history[key].to(self.device) for key in self.encoder.keys}
            past_action = past_action.to(self.device)
            if gradient_planner:
                return self._gradient_plan(history, past_action, deterministic, first, goal)
            return self._cem(history, past_action, deterministic, first, goal)
        finally:
            for parameter in trainable:
                parameter.requires_grad_(True)
            self.train(was_training)
