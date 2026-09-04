"""Common DMC world-model evaluation on fixed expert trajectories."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import singledispatch
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from dmc_expert.storage import DATA_FORMAT, observation_indices, split_episode_indices
from models.dreamer import Dreamer
from models.planning import LatentPlanner
from models.shared.utils import parse_model_io
from models.storm import StormModel
from models.tdmpc2 import TDMPC2


@dataclass(frozen=True)
class Window:
    episode: int
    start: int


def _mode(logits):
    return F.one_hot(logits.argmax(dim=-1), logits.shape[-1]).to(logits.dtype).flatten(-2)


def _clone_cache(cache):
    return tuple(value.detach().clone() for value in cache)


@singledispatch
def latent_rollout(model, observation, action, context_length):
    raise TypeError(f"Physical-state evaluation does not support {type(model).__name__}.")


@latent_rollout.register
@torch.no_grad()
def _(model: TDMPC2, observation, action, context_length):
    latent = model.encoder(observation)
    state = latent[:, context_length - 1]
    predictions = []
    for current_action in action[:, context_length - 1 :].unbind(1):
        state = model.dynamics(torch.cat((state, current_action), dim=-1))
        predictions.append(state)
    return latent.float(), torch.stack(predictions, dim=1).float()


@latent_rollout.register
@torch.no_grad()
def _(model: LatentPlanner, observation, action, context_length):
    latent = model.encode(observation)
    history_size = model.history_size
    start = context_length - 1
    if context_length < history_size:
        raise ValueError(f"context_length={context_length} must be at least history_size={history_size}.")

    state = latent[:, context_length - history_size : context_length]
    past_action = action[:, context_length - history_size : start]
    predictions = []
    for current_action in action[:, start:].unbind(1):
        conditioned_action = torch.cat((past_action, current_action[:, None]), dim=1)
        next_state = model.predict(state, conditioned_action)[:, -1]
        predictions.append(model.pool(next_state))
        state = torch.cat((state[:, 1:], next_state[:, None]), dim=1)
        past_action = conditioned_action[:, 1:]
    return model.pool(latent).float(), torch.stack(predictions, dim=1).float()


@latent_rollout.register
@torch.no_grad()
def _(model: Dreamer, observation, action, context_length):
    observation = model.preprocess(dict(observation))
    embed = model.encoder(observation)
    batch, length = embed.shape[:2]
    previous_action = torch.cat((action.new_zeros(batch, 1, action.shape[-1]), action), dim=1)
    reset = torch.zeros(batch, length, dtype=torch.bool, device=embed.device)
    reset[:, 0] = True

    stoch, deter = model.rssm.initial(batch)
    cache = tuple(model.rssm.initial_context(batch) or ())
    features = []
    start_state = None
    for index in range(length):
        _, deter, logits, *cache = model.rssm.obs_step(
            stoch,
            deter,
            previous_action[:, index],
            embed[:, index],
            reset[:, index],
            *cache,
        )
        cache = tuple(cache)
        stoch = model.rssm.get_dist(logits).mode
        features.append(model.rssm.get_feat(stoch, deter))
        if index == context_length - 1:
            start_state = (stoch.detach().clone(), deter.detach().clone(), _clone_cache(cache))

    stoch, deter, cache = start_state
    predictions = []
    for current_action in action[:, context_length - 1 :].unbind(1):
        deter, cache = model.rssm._deter_step(stoch, deter, current_action, cache)
        stoch = model.rssm.get_dist(model.rssm._prior_net(deter)).mode
        predictions.append(model.rssm.get_feat(stoch, deter))
    return torch.stack(features, dim=1).float(), torch.stack(predictions, dim=1).float()


@latent_rollout.register
@torch.no_grad()
def _(model: StormModel, observation, action, context_length):
    world_model = model.world_model
    with world_model._amp():
        stoch = _mode(world_model.posterior(world_model.encoder(observation)))
        if not world_model.sequence_core.streaming:
            return _fixed_storm_rollout(world_model, stoch, action, context_length)

        batch, length = stoch.shape[:2]
        hidden_dim = world_model.feat_size - world_model.stoch_flattened_dim
        zero_deter = stoch.new_zeros(batch, hidden_dim)
        features = [torch.cat((stoch[:, 0], zero_deter), dim=-1)]
        cache = world_model.sequence_core.initial_cache(batch, dtype=stoch.dtype, device=stoch.device)
        start_cache = _clone_cache(cache) if context_length == 1 else None

        for index in range(length - 1):
            deter, cache = world_model.sequence_core.step(
                stoch[:, index : index + 1],
                action[:, index : index + 1],
                cache,
            )
            features.append(torch.cat((stoch[:, index + 1], deter[:, 0]), dim=-1))
            if index + 1 == context_length - 1:
                start_cache = _clone_cache(cache)

        current = stoch[:, context_length - 1]
        cache = start_cache
        predictions = []
        for current_action in action[:, context_length - 1 :].unbind(1):
            deter, cache = world_model.sequence_core.step(
                current[:, None],
                current_action[:, None],
                cache,
            )
            deter = deter[:, 0]
            current = _mode(world_model.prior(deter))
            predictions.append(torch.cat((current, deter), dim=-1))
    return torch.stack(features, dim=1).float(), torch.stack(predictions, dim=1).float()


def _fixed_storm_rollout(world_model, stoch, action, context_length):
    """Roll a fixed-context STORM core by rebuilding its latest context window."""
    core = world_model.sequence_core
    max_length = core.position_encoding.max_length
    batch = stoch.shape[0]
    hidden_dim = world_model.feat_size - world_model.stoch_flattened_dim
    features = [torch.cat((stoch[:, 0], stoch.new_zeros(batch, hidden_dim)), dim=-1)]

    first_window = min(action.shape[1], max_length)
    if first_window:
        deter = core(stoch[:, :first_window], action[:, :first_window])
        features.extend(torch.cat((stoch[:, index + 1], deter[:, index]), dim=-1) for index in range(first_window))
    for index in range(first_window, action.shape[1]):
        start = index + 1 - max_length
        deter = core(stoch[:, start : index + 1], action[:, start : index + 1])[:, -1]
        features.append(torch.cat((stoch[:, index + 1], deter), dim=-1))

    history = stoch[:, :context_length]
    action_history = action[:, : context_length - 1]
    predictions = []
    for current_action in action[:, context_length - 1 :].unbind(1):
        action_history = torch.cat((action_history, current_action[:, None]), dim=1)
        start = max(0, history.shape[1] - max_length)
        deter = core(history[:, start:], action_history[:, start:])[:, -1]
        current = _mode(world_model.prior(deter))
        predictions.append(torch.cat((current, deter), dim=-1))
        history = torch.cat((history, current[:, None]), dim=1)
    return torch.stack(features, dim=1).float(), torch.stack(predictions, dim=1).float()


class ProbeDataset:
    """Read fixed image/action windows and their privileged DMC state labels."""

    def __init__(self, path, model_io, proprio=None):
        self.path = Path(path).expanduser()
        self.metadata = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("format") != DATA_FORMAT:
            raise ValueError(f"{self.path} uses {self.metadata.get('format')!r}, expected {DATA_FORMAT!r}.")
        self.observation_shapes, action_shape, _ = parse_model_io(model_io)
        if "image" not in self.observation_shapes or set(self.observation_shapes) - {"image", "proprio"}:
            raise ValueError("The shared physical-state metric expects image and optional proprioception.")
        self.proprio_indices = None
        if "proprio" in self.observation_shapes:
            self.proprio_indices = observation_indices(self.metadata, proprio)
            if self.observation_shapes["proprio"] != (len(self.proprio_indices),):
                raise ValueError("Configured proprioception does not match the held-out dataset metadata.")
        self.h5 = h5py.File(self.path / "data.hdf5", "r")
        self.episodes = split_episode_indices(self.metadata, "heldout", self.h5["complete"].shape[0])
        self.action_dim = math.prod(action_shape)
        if self.h5["actions"].shape[-1] != self.action_dim:
            raise ValueError(
                f"Dataset action_dim={self.h5['actions'].shape[-1]} does not match model action_dim={self.action_dim}."
            )
        if "images" not in self.h5:
            raise ValueError(f"{self.path} has no images for image-model evaluation.")
        if self.h5["images"].shape[2:] != self.observation_shapes["image"]:
            raise ValueError(
                f"Dataset images have shape {self.h5['images'].shape[2:]}, expected {self.observation_shapes['image']}."
            )

    def close(self):
        self.h5.close()

    def split_windows(self, train_count, test_count, length, seed):
        lengths = np.asarray(self.h5["lengths"], dtype=np.int64)
        complete = np.asarray(self.h5["complete"], dtype=bool)
        incomplete = self.episodes[~complete[self.episodes]]
        if len(incomplete):
            raise ValueError(f"{self.path} is missing {len(incomplete)} held-out episodes.")
        episodes = self.episodes[lengths[self.episodes] >= length - 1]
        if len(episodes) < 2:
            raise ValueError(f"{self.path} needs at least two complete episodes with {length - 1} transitions.")
        rng = np.random.default_rng(seed)
        rng.shuffle(episodes)
        split = max(1, len(episodes) // 2)
        train_episodes, test_episodes = episodes[:split], episodes[split:]

        def sample(pool, count):
            windows = []
            for index in range(count):
                episode = int(pool[index % len(pool)])
                max_start = int(lengths[episode] - (length - 1))
                windows.append(Window(episode, int(rng.integers(max_start + 1))))
            return windows

        return sample(train_episodes, train_count), sample(test_episodes, test_count)

    def batches(self, windows, length, batch_size):
        for offset in range(0, len(windows), batch_size):
            batch = windows[offset : offset + batch_size]
            states = np.stack([
                np.asarray(
                    self.h5["observations"][window.episode, window.start : window.start + length],
                    dtype=np.float32,
                )
                for window in batch
            ])
            actions = np.stack([
                np.asarray(
                    self.h5["actions"][window.episode, window.start : window.start + length - 1],
                    dtype=np.float32,
                )
                for window in batch
            ])
            images = np.stack([
                np.asarray(self.h5["images"][window.episode, window.start : window.start + length]) for window in batch
            ])
            observation = {"image": torch.as_tensor(images)}
            if self.proprio_indices is not None:
                observation["proprio"] = torch.as_tensor(states[..., self.proprio_indices])
            yield observation, torch.as_tensor(actions), torch.as_tensor(states)


def _probe_weights(feature, state, ridge, device):
    feature = feature.to(device=device, dtype=torch.float32)
    state = state.to(device=device, dtype=torch.float32)
    feature_mean = feature.mean(0)
    feature_std = feature.std(0).clamp_min(1e-6)
    state_mean = state.mean(0)
    state_std = state.std(0)
    varying = state_std > 1e-6
    if not varying.any():
        raise ValueError("The selected probe windows contain no varying physical-state dimensions.")

    x = (feature - feature_mean) / feature_std
    y = (state[:, varying] - state_mean[varying]) / state_std[varying]
    scale = math.sqrt(x.shape[0])
    x, y = x / scale, y / scale
    if x.shape[0] <= x.shape[1]:
        system = x @ x.T + float(ridge) * torch.eye(x.shape[0], device=device)
        weight = x.T @ torch.linalg.solve(system, y)
    else:
        system = x.T @ x + float(ridge) * torch.eye(x.shape[1], device=device)
        weight = torch.linalg.solve(system, x.T @ y)
    return feature_mean, feature_std, state_mean[varying], state_std[varying], varying, weight


@torch.no_grad()
def evaluate_state_prediction(
    model,
    model_io,
    dataset_path,
    *,
    proprio=None,
    context_length=64,
    horizons=(1, 5, 10, 25),
    train_windows=32,
    test_windows=32,
    batch_size=8,
    ridge=1e-3,
    seed=2_000_000,
):
    """Fit one frozen linear state probe and score action-conditioned latent rollouts."""

    horizons = tuple(sorted(set(map(int, horizons))))
    if not horizons or horizons[0] < 1:
        raise ValueError("Prediction horizons must be positive integers.")
    context_length = int(context_length)
    if context_length < 1:
        raise ValueError("context_length must be positive.")
    total_length = context_length + horizons[-1]
    device = model.world_model.device if isinstance(model, StormModel) else next(model.parameters()).device
    was_training = model.training
    model.eval()
    dataset = ProbeDataset(dataset_path, model_io, proprio)
    try:
        train_specs, test_specs = dataset.split_windows(train_windows, test_windows, total_length, seed)
        train_feature, train_state = [], []
        for observation, action, state in dataset.batches(train_specs, total_length, batch_size):
            observation = {key: value.to(device, non_blocking=True) for key, value in observation.items()}
            action = action.to(device, non_blocking=True)
            feature, _ = latent_rollout(model, observation, action, context_length)
            train_feature.append(feature[:, context_length - 1 :].cpu())
            train_state.append(state[:, context_length - 1 :])
        train_feature = torch.cat(train_feature).flatten(0, 1)
        train_state = torch.cat(train_state).flatten(0, 1)
        probe = _probe_weights(train_feature, train_state, ridge, device)

        feature_mean, feature_std, state_mean, state_std, varying, weight = probe
        errors = {horizon: [] for horizon in horizons}
        for observation, action, state in dataset.batches(test_specs, total_length, batch_size):
            observation = {key: value.to(device, non_blocking=True) for key, value in observation.items()}
            action = action.to(device, non_blocking=True)
            _, prediction = latent_rollout(model, observation, action, context_length)
            prediction = (prediction - feature_mean) / feature_std
            predicted_state = prediction @ weight
            target = state[:, context_length:].to(device)[:, :, varying]
            target = (target - state_mean) / state_std
            for horizon in horizons:
                squared_error = (predicted_state[:, horizon - 1] - target[:, horizon - 1]).square().mean(-1)
                errors[horizon].append(squared_error.cpu())

        nrmse = {str(horizon): float(torch.cat(values).mean().sqrt()) for horizon, values in errors.items()}
        return {
            "nrmse": nrmse,
            "mean_nrmse": float(np.mean(list(nrmse.values()))),
            "metric": "RMSE after per-dimension physical-state standardization",
            "probe": "ridge linear",
            "ridge": float(ridge),
            "episode_split": "disjoint probe-fit and probe-test episodes",
            "dataset_split": "heldout",
            "dataset_episode_range": [int(dataset.episodes[0]), int(dataset.episodes[-1]) + 1],
            "horizons": list(horizons),
            "context_length": context_length,
            "probe_train_windows": int(train_windows),
            "probe_test_windows": int(test_windows),
            "physical_state_dim": int(varying.sum()),
        }
    finally:
        dataset.close()
        model.train(was_training)
