"""Common DMC world-model evaluation on fixed expert trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import singledispatch
from pathlib import Path

import h5py
import numpy as np
import torch

from dmc_expert.storage import (
    observation_indices,
    read_image_window,
    read_physical_state,
    split_episode_indices,
)
from models.dreamer import Dreamer
from models.planning import LatentPlanner
from models.shared.utils import parse_model_io
from models.storm import StormModel
from models.storm.world_model import categorical_sample
from models.tdmpc2 import TDMPC2
from training.progress import Progress


class EpisodeMetrics:
    """Common scoring for one completed evaluation episode per environment."""

    def __init__(self, batch_size, device, config):
        self.settings = config.evaluation
        self.action_repeat = int(config.env.action_repeat)
        self.returns = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.reached = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.sustained = torch.zeros_like(self.reached)
        self.streak = torch.zeros_like(self.lengths)
        self.success_history = []
        self.active_history = []
        limit = int(config.env.get("time_limit", 0))
        self.progress = Progress("Evaluation", math.ceil(limit / self.action_repeat)) if limit else None

    def update(self, reward, active):
        self.returns += reward[:, 0] * active
        self.lengths += active
        qualifies = (reward[:, 0] / self.action_repeat >= float(self.settings.success_threshold)) & active
        self.reached |= qualifies
        self.streak = torch.where(qualifies, self.streak + 1, torch.where(active, 0, self.streak))
        self.sustained |= self.streak >= int(self.settings.sustained_success_steps)
        self.success_history.append(qualifies)
        self.active_history.append(active)
        if self.progress and self.progress.due():
            self.progress.update(int(self.lengths.max()), f"episodes={self.returns.numel()}")

    def result(self):
        if self.progress:
            self.progress.update(int(self.lengths.max()), f"episodes={self.returns.numel()}", force=True)
        active = torch.stack(self.active_history)
        qualifies = torch.stack(self.success_history)
        tail_length = (self.lengths * float(self.settings.maintenance_fraction)).ceil().long().clamp_min(1)
        tail = active & (active.cumsum(0) > self.lengths - tail_length)
        occupancy = (qualifies & tail).sum(0) / tail_length
        success = (self.lengths > 0) & (occupancy >= float(self.settings.maintenance_occupancy))
        std = self.returns.std(unbiased=self.returns.numel() > 1)
        return (
            float(self.returns.mean()),
            float(self.lengths.float().mean()),
            {
                "success": success.float().mean(),
                "reached_success": self.reached.float().mean(),
                "sustained_success": self.sustained.float().mean(),
                "return_std": std,
                "return_stderr": std / self.returns.numel() ** 0.5,
            },
        )


@dataclass(frozen=True)
class Window:
    episode: int
    start: int
    cohort: str = "uniform"
    motion: float = 0.0


def _clone_cache(cache):
    return tuple(value.detach().clone() for value in cache)


@singledispatch
def latent_rollout(model, observation, action, context_length):
    raise TypeError(f"Physical-state evaluation does not support {type(model).__name__}.")


@latent_rollout.register
@torch.no_grad()
def _tdmpc2_rollout(model: TDMPC2, observation, action, context_length):
    stacked = model.stack_sequence(observation)
    latent = model._forward(model.encoder, {key: value[:, :context_length] for key, value in stacked.items()})
    state = latent[:, context_length - 1]
    predictions = []
    for current_action in action[:, context_length - 1 :].unbind(1):
        state = model._forward(model.dynamics, torch.cat((state, current_action), dim=-1))
        predictions.append(state)
    future = model._forward(model.encoder, {key: value[:, context_length:] for key, value in stacked.items()})
    return torch.cat((latent, future), dim=1).float(), torch.stack(predictions, dim=1).float()


@latent_rollout.register
@torch.no_grad()
def _planning_rollout(model: LatentPlanner, observation, action, context_length):
    latent = model.encode({key: value[:, :context_length] for key, value in observation.items()})
    history_size = model.history_size
    start = context_length - 1
    if context_length < history_size:
        raise ValueError(f"context_length={context_length} must be at least history_size={history_size}.")

    state = latent[:, context_length - history_size : context_length]
    past_action = action[:, context_length - history_size : start]
    prediction = model.rollout(state, past_action, action[:, None, start:])[:, 0]
    future = model.encode({key: value[:, context_length:] for key, value in observation.items()})
    return torch.cat((latent, future), dim=1).float(), prediction.float()


@latent_rollout.register
@torch.no_grad()
def _dreamer_rollout(model: Dreamer, observation, action, context_length):
    with model.amp_context():
        return _dreamer_predictions(model, observation, action, context_length)


def _dreamer_predictions(model, observation, action, context_length):
    observation = model.preprocess(dict(observation))
    embed = model.encoder({key: value[:, :context_length] for key, value in observation.items()})
    batch, length = next(iter(observation.values())).shape[:2]
    previous_action = torch.cat((action.new_zeros(batch, 1, action.shape[-1]), action), dim=1)
    reset = torch.zeros(batch, length, dtype=torch.bool, device=embed.device)
    reset[:, 0] = True

    stoch, deter = model.rssm.initial(batch)
    cache = tuple(model.rssm.initial_context(batch) or ())
    features = []
    predictions = []
    with model.rssm.sequence_context(embed):
        for index in range(context_length):
            stoch, deter, _, *cache = model.rssm.obs_step(
                stoch, deter, previous_action[:, index], embed[:, index], reset[:, index], *cache
            )
            features.append(model.rssm.get_feat(stoch, deter))

        # Fork at the anchor before either branch can mutate the recurrent cache.
        observed_state = (stoch.clone(), deter.clone(), _clone_cache(cache))
        for current_action in action[:, context_length - 1 :].unbind(1):
            stoch, deter, *cache = model.rssm.img_step(stoch, deter, current_action, *cache)
            predictions.append(model.rssm.get_feat(stoch, deter))

        # The decoding diagnostic sees real future images, but only after forecasting.
        stoch, deter, cache = observed_state
        future_embed = model.encoder({key: value[:, context_length:] for key, value in observation.items()})
        for index in range(context_length, length):
            stoch, deter, _, *cache = model.rssm.obs_step(
                stoch, deter, previous_action[:, index],
                future_embed[:, index - context_length], reset[:, index], *cache,
            )
            features.append(model.rssm.get_feat(stoch, deter))
    return torch.stack(features, dim=1).float(), torch.stack(predictions, dim=1).float()


@latent_rollout.register
@torch.no_grad()
def _storm_rollout(model: StormModel, observation, action, context_length, *, storm_context_length=None):
    world_model = model.world_model
    with world_model._amp():
        stoch = world_model.encode_obs({key: value[:, :context_length] for key, value in observation.items()})
        if not world_model.sequence_core.streaming:
            if storm_context_length is None:
                raise ValueError("Fixed-context STORM evaluation requires storm_train.context_length.")
            return _fixed_storm_rollout(world_model, stoch, observation, action, storm_context_length)

        batch, length = stoch.shape[:2]
        hidden_dim = world_model.feat_size - world_model.stoch_flattened_dim
        zero_deter = stoch.new_zeros(batch, hidden_dim)
        features = [torch.cat((stoch[:, 0], zero_deter), dim=-1)]
        cache = world_model.sequence_core.initial_cache(batch, dtype=stoch.dtype, device=stoch.device)
        for index in range(length - 1):
            deter, cache = world_model.sequence_core.step(
                stoch[:, index : index + 1],
                action[:, index : index + 1],
                cache,
            )
            features.append(torch.cat((stoch[:, index + 1], deter[:, 0]), dim=-1))
        observed_cache = _clone_cache(cache)
        current = stoch[:, context_length - 1]
        predictions = []
        for current_action in action[:, context_length - 1 :].unbind(1):
            deter, cache = world_model.sequence_core.step(
                current[:, None],
                current_action[:, None],
                cache,
            )
            deter = deter[:, 0]
            current = categorical_sample(world_model.prior(deter)).flatten(-2)
            predictions.append(torch.cat((current, deter), dim=-1))

        future = world_model.encode_obs({key: value[:, context_length:] for key, value in observation.items()})
        observed = torch.cat((stoch[:, -1:], future), dim=1)
        cache = observed_cache
        for index, current_action in enumerate(action[:, context_length - 1 :].unbind(1)):
            deter, cache = world_model.sequence_core.step(
                observed[:, index : index + 1], current_action[:, None], cache
            )
            features.append(torch.cat((future[:, index], deter[:, 0]), dim=-1))
    return torch.stack(features, dim=1).float(), torch.stack(predictions, dim=1).float()


def _fixed_storm_rollout(world_model, stoch, observation, action, max_length):
    """Roll a fixed-context STORM core by rebuilding its latest context window."""
    core = world_model.sequence_core
    context_length = stoch.shape[1]
    batch = stoch.shape[0]
    hidden_dim = world_model.feat_size - world_model.stoch_flattened_dim
    features = [torch.cat((stoch[:, 0], stoch.new_zeros(batch, hidden_dim)), dim=-1)]

    first_window = min(context_length - 1, max_length)
    if first_window:
        deter = core(stoch[:, :first_window], action[:, :first_window])
        features.extend(torch.cat((stoch[:, index + 1], deter[:, index]), dim=-1) for index in range(first_window))
    for index in range(first_window, context_length - 1):
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
        current = categorical_sample(world_model.prior(deter)).flatten(-2)
        predictions.append(torch.cat((current, deter), dim=-1))
        history = torch.cat((history, current[:, None]), dim=1)

    future = world_model.encode_obs({key: value[:, context_length:] for key, value in observation.items()})
    observed = torch.cat((stoch, future), dim=1)
    for index in range(context_length - 1, action.shape[1]):
        start = max(0, index + 1 - max_length)
        deter = core(observed[:, start : index + 1], action[:, start : index + 1])[:, -1]
        features.append(torch.cat((observed[:, index + 1], deter), dim=-1))
    return torch.stack(features, dim=1).float(), torch.stack(predictions, dim=1).float()


class StateDataset:
    """Read the same fixed observation/action windows for every model, from held-out episodes only."""

    def __init__(self, h5, metadata, model_io, fields, targets):
        self.h5 = h5
        self.path = Path(h5.filename).parent
        observation_shapes, action_shape, _ = parse_model_io(model_io)
        if set(observation_shapes) != {"image"}:
            raise ValueError("Physical-state evaluation expects image-only model inputs.")
        self.state_indices = observation_indices(metadata, fields)
        self.state_targets = targets
        self.episodes = split_episode_indices(metadata, "heldout", self.h5["complete"].shape[0])
        training = split_episode_indices(metadata, "train", self.h5["complete"].shape[0])
        if np.intersect1d(training, self.episodes).size:
            raise ValueError("Physical-state evaluation requires disjoint training and held-out episodes.")
        action_dim = math.prod(action_shape)
        if self.h5["actions"].shape[-1] != action_dim:
            raise ValueError(
                f"Dataset action_dim={self.h5['actions'].shape[-1]} does not match model action_dim={action_dim}."
            )
        if "images" not in self.h5:
            raise ValueError(f"{self.path} has no images for image-model evaluation.")
        if self.h5["images"].shape[2:] != observation_shapes["image"]:
            raise ValueError(
                f"Dataset images have shape {self.h5['images'].shape[2:]}, expected {observation_shapes['image']}."
            )

    def sample_windows(self, count, length, seed, context_length, motion_fraction, motion_candidates):
        lengths = np.asarray(self.h5["lengths"], dtype=np.int64)
        episodes = self.episodes[lengths[self.episodes] >= length - 1]
        if not len(episodes) or not np.asarray(self.h5["complete"])[episodes].all():
            raise ValueError(f"{self.path} needs complete held-out episodes with {length - 1} transitions.")
        rng = np.random.default_rng(seed)
        rng.shuffle(episodes)
        windows = []
        motion_count = int(count * motion_fraction)
        for index in range(count):
            episode = int(episodes[index % len(episodes)])
            max_start = int(lengths[episode] - (length - 1))
            cohort = "motion" if index < motion_count else "uniform"
            candidates = min(motion_candidates if cohort == "motion" else 1, max_start + 1)
            starts = rng.choice(max_start + 1, size=candidates, replace=False)
            state = read_physical_state(
                self.h5, episode, slice(0, lengths[episode] + 1), self.state_indices, self.state_targets
            )
            # Rank physical motion, not rewards or model error. Scale coordinates only for window selection.
            position = state[:, self.state_targets.positions].astype(np.float64)
            scale = np.maximum(np.ptp(position, axis=0), 1e-3)
            change = np.square(np.diff(position, axis=0) / scale).mean(axis=-1)
            scores = [float(change[start + context_length - 1 : start + length - 1].mean()) for start in starts]
            selected = int(np.argmax(scores))
            windows.append(Window(episode, int(starts[selected]), cohort, scores[selected]))
        return windows

    def batches(self, windows, length, batch_size):
        for offset in range(0, len(windows), batch_size):
            yield self.read_batch(windows[offset : offset + batch_size], length)

    def read_batch(self, batch, length):
        if not np.isin([window.episode for window in batch], self.episodes).all():
            raise ValueError("Forecast windows must belong to the held-out split, never training episodes.")
        targets = np.stack([
            read_physical_state(
                self.h5, window.episode, slice(window.start, window.start + length),
                self.state_indices, self.state_targets,
            )
            for window in batch
        ])
        actions = np.stack([
            np.asarray(self.h5["actions"][window.episode, window.start : window.start + length - 1], dtype=np.float32)
            for window in batch
        ])
        images = np.stack([
            read_image_window(self.h5["images"], window.episode, window.start, length)
            for window in batch
        ])
        return {"image": torch.as_tensor(images)}, torch.as_tensor(actions), torch.as_tensor(targets)


@torch.no_grad()
def evaluate_state_prediction(model, config, dataset_path, metadata):
    """Score the checkpoint's physical readout without fitting on held-out episodes."""
    settings = config.evaluation.final
    horizons = tuple(sorted(set(map(int, settings.horizons))))
    context_length = int(settings.context_length)
    head = model.state_head
    coordinates = head.coordinates + head.targets.derived_coordinates
    if not head.updates.item():
        raise ValueError("The checkpoint has no trained physical-state head.")
    frame_stack = model.frame_stack if isinstance(model, TDMPC2) else 1
    if context_length < max(head.history, frame_stack):
        raise ValueError("Evaluation context is shorter than the physical head's native history.")
    total_length = context_length + horizons[-1]
    device = head.mean.device
    stochastic = isinstance(model, (Dreamer, StormModel))
    samples = int(settings.state_samples) if stochastic else 1
    batch_size = int(settings.state_batch_size)
    rollout_options = {}
    if isinstance(model, StormModel):
        rollout_options["storm_context_length"] = int(config.storm_train.context_length)
    with h5py.File(Path(dataset_path).expanduser() / "data.hdf5", "r") as h5:
        dataset = StateDataset(h5, metadata, config.model_io, config.state_head.fields, head.targets)
        was_training = model.training
        model.eval()
        try:
            windows = dataset.sample_windows(
                int(settings.state_windows), total_length, int(settings.state_seed), context_length,
                float(settings.motion_fraction), int(settings.motion_candidates),
            )
            errors = {name: [] for name in ("rmse", "observed_rmse", "persistence_rmse")}
            progress = Progress("Prediction", len(windows))
            completed = 0
            progress.update(completed, "held-out windows")
            horizon_indices = torch.tensor(horizons, device=device) - 1
            devices = [device.index] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(settings.state_seed))
                batches = dataset.batches(windows, total_length, max(1, batch_size // samples))
                for observation, action, target in batches:
                    observation = {key: value.to(device, non_blocking=True) for key, value in observation.items()}
                    action = action.to(device, non_blocking=True)
                    batch = action.shape[0]
                    parallel_samples = max(1, batch_size // batch)
                    forecast, observed, persistence = 0, 0, 0
                    for offset in range(0, samples, parallel_samples):
                        count = min(parallel_samples, samples - offset)
                        feature, prediction = latent_rollout(
                            model,
                            {key: value.repeat_interleave(count, dim=0) for key, value in observation.items()},
                            action.repeat_interleave(count, dim=0),
                            context_length,
                            **rollout_options,
                        )
                        prefix = feature[:, context_length - head.history + 1 : context_length]
                        decoded = head(torch.cat((prefix, prediction), dim=1))
                        diagnostic = head(feature[:, context_length - head.history + 1 :])
                        anchor = head(feature[:, context_length - head.history : context_length])
                        # Derive kinematics per draw before averaging, not from averaged angles/velocities.
                        decoded = torch.cat((decoded, head.targets.derived(decoded)), dim=-1)
                        diagnostic = torch.cat((diagnostic, head.targets.derived(diagnostic)), dim=-1)
                        anchor = torch.cat((anchor, head.targets.derived(anchor)), dim=-1)
                        # Average physical predictions, not categorical latents or per-sample errors.
                        forecast = forecast + decoded.reshape(batch, count, horizons[-1], -1).sum(1)
                        observed = observed + diagnostic.reshape(batch, count, horizons[-1], -1).sum(1)
                        persistence = persistence + anchor.reshape(batch, count, 1, -1).sum(1)
                    target = target[:, context_length:].to(device)
                    target = torch.cat((target, head.targets.derived(target)), dim=-1)
                    estimates = {"rmse": forecast, "observed_rmse": observed, "persistence_rmse": persistence}
                    for name, estimate in estimates.items():
                        squared_error = (estimate / samples - target).square()
                        errors[name].append(squared_error.index_select(1, horizon_indices).cpu())
                    completed += batch
                    progress.update(completed, "held-out windows", force=completed == len(windows))
            errors = {name: torch.cat(rows) for name, rows in errors.items()}

            def summarize(indices):
                result = {}
                for name, values in errors.items():
                    scores = values[indices].mean(0).sqrt()
                    for prefix, names in (("", head.coordinates), ("derived_", head.targets.derived_coordinates)):
                        columns = [coordinates.index(coordinate) for coordinate in names]
                        result[prefix + name] = {
                            str(horizon): dict(zip(names, scores[index, columns].tolist(), strict=True))
                            for index, horizon in enumerate(horizons)
                        }
                return result

            cohorts = {}
            for cohort in ("uniform", "motion"):
                indices = [index for index, window in enumerate(windows) if window.cohort == cohort]
                if indices:
                    cohorts[cohort] = {"windows": len(indices), **summarize(indices)}

            return {
                **summarize(slice(None)),
                "cohorts": cohorts,
                "windows": [
                    {
                        "episode": window.episode, "start": window.start,
                        "forecast_start": window.start + context_length,
                        "cohort": window.cohort, "motion_score": window.motion,
                    }
                    for window in windows
                ],
                "window_sampling": "uniform_and_motion_stratified",
                "motion_definition": (
                    "mean squared position increment during forecast, "
                    "scaled by held-out episode coordinate ranges (floor 1e-3)"
                ),
                "persistence_definition": "hold the last decoded observed state fixed; no simulator-state input",
                "rollout": "open_loop_recorded_actions",
                "derived_coordinates": head.targets.derived_coordinates,
                "derived_units": "position: m; velocity: m/s",
                "target_version": str(config.state_head.target_version),
                "metric": "RMSE per physical coordinate in original units; fixed checkpoint readout",
                "state_coordinates": head.coordinates,
                "readout": "detached supervised physical-state head",
                "readout_history_length": head.history,
                "readout_updates": int(head.updates),
                "readout_examples": int(head.examples),
                "evaluation_fitting": False,
                "dataset_split": "heldout",
                "dataset_episode_range": [int(dataset.episodes[0]), int(dataset.episodes[-1]) + 1],
                "horizons": list(horizons),
                "context_length": context_length,
                "context_length_role": "common_observed_prefix",
                "history_policy": "common_prefix_native_memory",
                "latent_sampling": "native" if stochastic else "deterministic",
                "state_samples": samples,
                "state_seed": int(settings.state_seed),
                "state_windows": len(windows),
                "input_frame_stack": frame_stack,
                "physical_state_dim": len(head.coordinates),
            }
        finally:
            model.train(was_training)
