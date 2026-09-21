"""Diagnostic-only adapters; production models, objectives and replay stay unchanged."""

import copy
from contextlib import contextmanager
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dmc_expert.storage import split_episode_indices
from scripts.diagnose_goal_objective import cart_state, set_cart_state


def action_statistics(dataset):
    total = np.zeros(dataset.action_dim, np.float64)
    squared = np.zeros_like(total)
    count = 0
    for offset in range(0, len(dataset.episodes), 64):
        episodes = np.sort(dataset.episodes[offset:offset + 64])
        values = np.asarray(dataset.actions[episodes], dtype=np.float64)
        mask = np.arange(values.shape[1])[None] < dataset.lengths[episodes, None]
        values = values[mask]
        total += values.sum(0)
        squared += (values * values).sum(0)
        count += len(values)
    if not count or not np.isfinite(total).all() or not np.isfinite(squared).all():
        raise ValueError("Training actions have no finite normalization statistics.")
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(squared / count - mean**2, 0)), 1e-3)
    return mean.astype(np.float32), std.astype(np.float32), count


def resize_images(images, size):
    if images.shape[-3:-1] == (size, size):
        return images
    prefix = images.shape[:-3]
    pixels = images.reshape(-1, *images.shape[-3:]).permute(0, 3, 1, 2).float()
    pixels = F.interpolate(pixels, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return pixels.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).reshape(*prefix, size, size, 3)


def block_actions(actions, stride):
    if actions.shape[-2] % stride:
        raise ValueError("Incomplete action block.")
    return actions.reshape(*actions.shape[:-2], actions.shape[-2] // stride, stride * actions.shape[-1])


class BlockReplay:
    def __init__(self, dataset, stride, size, device):
        self.dataset, self.stride, self.size, self.device = dataset, stride, size, device
        self.state_mean, self.state_std = dataset.state_mean, dataset.state_std

    def sample_episode_batch(self):
        obs, actions, rewards, terminals = self.dataset.sample_episode_batch()
        obs = {key: value[:, ::self.stride].to(self.device) for key, value in obs.items()}
        obs["image"] = resize_images(obs["image"], self.size)
        return (obs, block_actions(actions, self.stride).to(self.device),
                block_actions(rewards, self.stride).sum(-1, keepdim=True).to(self.device),
                block_actions(terminals, self.stride).amax(-1, keepdim=True).to(self.device))


class NormalizedActionEncoder(nn.Module):
    def __init__(self, encoder, mean, std, device):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("mean", torch.as_tensor(mean, device=device))
        self.register_buffer("std", torch.as_tensor(std, device=device))

    def forward(self, actions):
        return self.encoder((actions.float() - self.mean) / self.std)


def reference_configs(base, stride):
    """Upstream-sized components, vision-only DMC adaptation, unchanged native losses."""
    raw, model = copy.deepcopy(base), copy.deepcopy(base)
    ts = str(base.model_family) == "temporal_straightening"
    batch = 32 if ts else 128
    for config in (raw, model):
        config.replay.batch_size = config.training.expert.batch_size = batch
        config.replay.episodes_per_batch = batch
    raw.replay.sequence_length = 3 * stride + 1
    model.model_io.observations.image = [224, 224, 3]
    model.model_io.action.shape = [int(base.model_io.action.shape[0]) * stride]
    settings = model.jepa_model
    settings.encoder.embedding_dim = 8 if ts else 192
    settings.encoder.vision.base_channels = 32
    settings.encoder.vision.patch_size = 14
    settings.encoder.vision.layers = 12
    settings.encoder.vision.heads = 3
    settings.encoder.vision.mlp_dim = 768
    settings.encoder.vision.dropout = 0.0
    settings.predictor.layers = 6
    settings.predictor.heads = 16
    settings.predictor.dim_head = 64
    settings.predictor.mlp_dim = 2048
    settings.predictor.action_embedding_dim = 10
    settings.projector_dim = 2048
    settings.planner.samples = 1 if ts else 300
    settings.planner.elites = 30
    settings.planner.iterations = 100 if ts else 10
    settings.planner.horizon = 5
    settings.planner.objective = "last"  # Historical reference experiment, endpoint-only scoring.
    if ts:
        settings.prediction_weight = 1.0  # Visual-only MSE, not the old visual/proprio channel ratio.
    return raw, model


def direct_gradient_plan(model, history, past_action, deterministic, first, goal, *, mean, std, mode):
    """100-step, zero-initialized comparison; only direct mode uses normalized coordinates."""
    del deterministic
    batch = history["image"].shape[0]
    horizon = int(model.planner.horizon)
    with torch.no_grad():
        latent = model.encode(history)
        actions = torch.zeros(batch, 1, horizon, model.action_dim, device=model.device)
        if model._gradient_actions is not None and model._gradient_actions.shape == actions.shape:
            shifted = torch.cat((model._gradient_actions[:, :, 1:], actions[:, :, -1:]), dim=2)
            actions = torch.where(model._first_mask(first, batch)[:, None, None, None], actions, shifted)
        mean = torch.as_tensor(mean, device=model.device)
        std = torch.as_tensor(std, device=model.device)
        value = torch.atanh(actions.clamp(-.999, .999)) if mode == "tanh" else (actions - mean) / std
    value = nn.Parameter(value)
    optimizer = torch.optim.Adam([value], lr=1.0 if mode == "tanh" else .1)
    iterations = int(model.planner.iterations)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    chunk_size = int(model.planner.gradient_batch_size)
    with torch.enable_grad():
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            gradient = torch.empty_like(value)
            for start in range(0, batch, chunk_size):
                stop = min(start + chunk_size, batch)
                chunk = value[start:stop].detach().requires_grad_()
                candidate = chunk.tanh() if mode == "tanh" else mean + std * chunk
                cost = model._goal_cost(latent[start:stop], past_action[start:stop], candidate, goal[start:stop])
                gradient[start:stop] = torch.autograd.grad(cost.sum(), chunk)[0]
            value.grad = gradient
            optimizer.step()
            scheduler.step()
            # DMC bounds are an explicit adaptation: never score impossible controls.
            if mode == "direct":
                with torch.no_grad():
                    value.copy_(((mean + std * value).clamp(-1, 1) - mean) / std)
    with torch.no_grad():
        actions = value.tanh() if mode == "tanh" else (mean + std * value).clamp(-1, 1)
        model._gradient_actions = actions.detach()
        return actions[:, 0, 0]


@contextmanager
def optimizer_arm(model, name, mean, std, iterations=100):
    original_planner = model.planner
    original_method = model.__dict__.get("_gradient_plan")
    caches = model._cem_mean, model._gradient_actions
    model.planner = copy.deepcopy(original_planner)
    model._cem_mean = model._gradient_actions = None
    try:
        if name != "current":
            if name not in {"tanh", "direct"}:
                raise ValueError(f"Unknown optimizer arm: {name}")
            model.planner.iterations = iterations
            model.planner.samples = 1
            model.planner.action_noise = 0.0
            def plan(self, history, past_action, deterministic, first, goal):
                return direct_gradient_plan(self, history, past_action, deterministic, first, goal,
                                            mean=mean, std=std, mode=name)
            model._gradient_plan = MethodType(plan, model)
        yield
    finally:
        model.planner = original_planner
        model._cem_mean, model._gradient_actions = caches
        if original_method is None:
            model.__dict__.pop("_gradient_plan", None)
        else:
            model._gradient_plan = original_method


def physical_cart_state(observation):
    # The stored DMC observation is [x, cos(theta), sin(theta), dx, dtheta].
    return np.asarray([observation[0], np.arctan2(observation[2], observation[1]),
                       observation[3], observation[4]], dtype=np.float64)


def pose_distance(states, goal, tolerance):
    difference = np.asarray(states)[..., :2] - np.asarray(goal)[..., :2]
    difference = difference.copy()
    difference[..., 1] = np.arctan2(np.sin(difference[..., 1]), np.cos(difference[..., 1]))
    return np.max(np.abs(difference) / np.asarray(tolerance), axis=-1)


def heldout_pairs(dataset, env, *, stride, horizon, count, seed, tolerance):
    """Disjoint heldout episodes; goals are known reachable, not hand-rendered setpoints."""
    episodes = split_episode_indices(dataset.metadata, "heldout", len(dataset.lengths))
    if set(episodes) & set(dataset.episodes):
        raise ValueError("Reference validation overlaps the training split.")
    if list(dataset.metadata["observation_keys"]) != ["position", "velocity"] or dataset.metadata["obs_dim"] != 5:
        raise ValueError("Reference state restoration requires the cartpole observation layout.")
    rng = np.random.default_rng(seed)
    cases = []
    attempted = 0
    for episode in rng.permutation(episodes):
        length = (2 + horizon) * stride
        choices = int(dataset.lengths[episode]) - length + 1
        if choices < 1:
            continue
        for start in rng.permutation(choices)[:32]:
            observations = np.asarray(dataset.h5["observations"][episode, start:start + length + 1])
            anchor = physical_cart_state(observations[2 * stride])
            goal = physical_cart_state(observations[-1])
            if pose_distance(anchor, goal, tolerance) <= 2:
                continue
            actions = np.asarray(dataset.actions[episode, start:start + length], np.float32)
            env.reset()
            set_cart_state(env, physical_cart_state(observations[0]))
            images = [env.render()]
            states = [cart_state(env)]
            for action in actions:
                obs, _, done, _ = env.step(action)
                if done:
                    raise ValueError("Reference replay crossed the episode boundary.")
                images.append(obs["image"])
                states.append(cart_state(env))
            attempted += 1
            errors = np.asarray(states) - np.stack([physical_cart_state(o) for o in observations])
            errors[:, 1] = np.arctan2(np.sin(errors[:, 1]), np.cos(errors[:, 1]))
            # Stored observations omit simulator solver buffers. Check restoration once,
            # instead of silently assuming the expert action sequence still reaches its goal.
            if np.max(np.abs(errors)) > 1e-3:
                continue
            cases.append({"id": f"heldout_{episode}_{start}", "episode": int(episode), "start": int(start),
                          "initial_state": physical_cart_state(observations[0]).tolist(),
                          "prefix": torch.from_numpy(np.stack(images[:2 * stride + 1:stride])),
                          "prefix_actions": actions[:2 * stride], "expert_actions": actions[2 * stride:],
                          "anchor_state": states[2 * stride].tolist(), "goal_state": goal.tolist(),
                          "goal_image": torch.from_numpy(np.ascontiguousarray(images[-1])), "restoration_max_error": float(np.abs(errors).max()),
                          "initial_pose_distance": float(pose_distance(anchor, goal, tolerance))})
            break
        if len(cases) >= count:
            break
    return cases, attempted
