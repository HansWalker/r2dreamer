"""Replay views over dense DMC expert datasets."""

import math
from pathlib import Path

import h5py
import numpy as np
import torch
from tensordict import TensorDict

from models.shared.physical_state import STATE_KEY, PhysicalStateTargets

from .storage import (
    observation_indices,
    read_image_window,
    read_physical_state,
    split_episode_indices,
    validate_dataset,
)


class DMCExpertDataset:
    """Shared HDF5 loading and tensor formatting for DMC expert data."""

    frame_stack = 1

    def __init__(self, config):
        settings = config.training.expert
        self.path = Path(settings.data_path).expanduser()
        self.batch_size = int(settings.batch_size)
        self.episodes_per_batch = int(config.replay.episodes_per_batch)
        self.shuffle = bool(settings.shuffle)
        seed = int(config.replay.seed)
        self.episode_rng = np.random.default_rng(seed)
        self.window_rng = np.random.default_rng(seed + 1_000_003)

        self.metadata = validate_dataset(self.path, config, splits=("train",))
        self.action_dim = int(self.metadata["action_dim"])
        requested = {str(key): tuple(map(int, shape)) for key, shape in config.model_io.observations.items()}
        size = int(self.metadata["image_size"])
        stored_shapes = {"image": (size, size, 3)}
        if requested != stored_shapes:
            raise ValueError(f"Configured observations {requested} do not match dataset images {stored_shapes}.")
        action = config.model_io.action
        if str(action.kind) != "continuous" or math.prod(action.shape) != self.action_dim:
            raise ValueError(
                f"Configured action {action.kind}{tuple(action.shape)} does not match "
                f"continuous dataset action_dim={self.action_dim}."
            )
        self.state_indices = observation_indices(self.metadata, config.state_head.fields)
        self.state_targets = PhysicalStateTargets(config.state_head.task, config.state_head.fields)
        self.obs_keys = ["image", STATE_KEY]

        self.h5 = h5py.File(self.path / "data.hdf5", "r")
        self.images = self.h5["images"]
        self.actions = self.h5["actions"]
        self.rewards = self.h5["rewards"]
        self.terminations = self.h5["terminations"]
        self.truncations = self.h5["truncations"]
        self.lengths = np.asarray(self.h5["lengths"], dtype=np.int64)

        train_episodes = split_episode_indices(self.metadata, "train", len(self.lengths))
        self.episodes = train_episodes[self.lengths[train_episodes] > 0]
        self._validate_episode_count()
        self.state_mean, self.state_std = self._state_stats()
        self._episode_order = np.array([], dtype=np.int64)
        self._episode_pos = 0

    def close(self):
        self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def state_dict(self):
        return {
            "episode_rng_state": self.episode_rng.bit_generator.state,
            "window_rng_state": self.window_rng.bit_generator.state,
            "episode_order": self._episode_order.copy(),
            "episode_pos": self._episode_pos,
        }

    def load_state_dict(self, state):
        self.episode_rng.bit_generator.state = state["episode_rng_state"]
        self.window_rng.bit_generator.state = state["window_rng_state"]
        self._episode_order = np.asarray(state["episode_order"], dtype=np.int64)
        self._episode_pos = int(state["episode_pos"])

    def _validate_episode_count(self):
        self.num_episodes = len(self.episodes)
        if self.num_episodes < self.episodes_per_batch:
            raise ValueError(
                f"{self.path} has {self.num_episodes} usable episodes, but this run requires "
                f"{self.episodes_per_batch} source episodes per update."
            )

    def _next_episode_indices(self, count):
        target = int(count)
        indices = []
        while len(indices) < target:
            if self._episode_pos >= len(self._episode_order):
                self._episode_order = (
                    self.episode_rng.permutation(self.episodes) if self.shuffle else self.episodes.copy()
                )
                self._episode_pos = 0
            take = min(
                target - len(indices),
                len(self._episode_order) - self._episode_pos,
            )
            indices.extend(self._episode_order[self._episode_pos : self._episode_pos + take])
            self._episode_pos += take
        return np.asarray(indices, dtype=np.int64)

    def _sample_episode_groups(self, valid_starts):
        indices = self._next_episode_indices(self.episodes_per_batch)
        windows_per_episode = self.batch_size // len(indices)
        groups = []
        for ep_idx in indices:
            choices = int(valid_starts(int(ep_idx)))
            starts = self.window_rng.choice(
                choices,
                size=windows_per_episode,
                replace=windows_per_episode > choices,
            )
            groups.append((int(ep_idx), tuple(sorted(map(int, starts)))))
        return groups

    def _read_observations(self, ep_idx, start, length):
        ep_idx, start, length = int(ep_idx), int(start), int(length)
        return {
            "image": read_image_window(self.images, ep_idx, start, length, self.frame_stack),
            STATE_KEY: read_physical_state(
                self.h5, ep_idx, slice(start, start + length), self.state_indices, self.state_targets
            ),
        }

    def _transition_batch(self, transitions):
        """Keep T observation frames, with T actions for STORM or T-1 for planners."""
        groups = self._sample_episode_groups(lambda ep_idx: int(self.lengths[ep_idx]) - transitions + 1)
        rows = []
        for ep_idx, starts in groups:
            for start in starts:
                row = self._read_observations(ep_idx, start, self.sequence_length)
                window = slice(start, start + transitions)
                row.update({
                    "action": np.asarray(self.actions[ep_idx, window], dtype=np.float32),
                    "reward": np.asarray(self.rewards[ep_idx, window], dtype=np.float32),
                    "terminal": np.asarray(self.terminations[ep_idx, window], dtype=np.float32),
                })
                rows.append(row)
        data = {key: torch.as_tensor(np.stack([row[key] for row in rows])) for key in rows[0]}
        batch = ({key: data[key] for key in self.obs_keys}, data["action"], data["reward"], data["terminal"])
        return groups, batch

    def _state_stats(self):
        total = np.zeros(len(self.state_targets.coordinates), dtype=np.float64)
        squared = np.zeros_like(total)
        count = 0
        steps = np.arange(self.images.shape[1])
        for offset in range(0, len(self.episodes), 64):
            episodes = self.episodes[offset : offset + 64]
            state = read_physical_state(self.h5, episodes, slice(None), self.state_indices, self.state_targets)
            state = state[steps[None] <= self.lengths[episodes, None]].astype(np.float64)
            total += state.sum(axis=0)
            squared += (state * state).sum(axis=0)
            count += len(state)
        mean = total / count
        std = np.sqrt(np.maximum(squared / count - np.square(mean), 1e-6))
        return mean.astype(np.float32), std.astype(np.float32)


class DMCExpertEpisodeReplay(DMCExpertDataset):
    """Sample Dreamer windows and rebuild their state from episode prefixes."""

    def __init__(self, config):
        super().__init__(config)
        self.sequence_length = int(config.replay.sequence_length)
        self.episodes = self.episodes[self.lengths[self.episodes] >= self.sequence_length - 1]
        self._validate_episode_count()

    def sample_episode_batch(self):
        groups = self._sample_episode_groups(lambda ep_idx: int(self.lengths[ep_idx]) - self.sequence_length + 2)
        rows = [self._make_window(ep_idx, start, self.sequence_length) for ep_idx, starts in groups for start in starts]
        data = {key: torch.as_tensor(np.stack([row[key] for row in rows], axis=0)) for key in rows[0]}
        batch = TensorDict(data, batch_size=(self.batch_size, self.sequence_length))
        contexts = []
        for ep_idx, starts in groups:
            context_length = max(starts)
            row = self._make_window(ep_idx, 0, context_length)
            context = TensorDict(
                {key: torch.as_tensor(value) for key, value in row.items()},
                batch_size=(context_length,),
            ).unsqueeze(0)
            contexts.append((context, starts))
        return contexts, batch

    def _make_window(self, ep_idx, start, length):
        ep_idx = int(ep_idx)
        start, length = int(start), int(length)
        obs = self._read_observations(ep_idx, start, length)
        # Dreamer pairs each observation with the action and reward that led to it.
        actions = np.zeros((length, self.action_dim), dtype=np.float32)
        rewards = np.zeros((length, 1), dtype=np.float32)
        terminations = np.zeros((length, 1), dtype=bool)
        is_last = np.zeros((length, 1), dtype=bool)
        destination = 0 if start else 1
        source = max(start - 1, 0)
        count = length - destination
        if count > 0:
            transition = slice(source, source + count)
            actions[destination:] = np.asarray(self.actions[ep_idx, transition], dtype=np.float32)
            rewards[destination:] = np.asarray(self.rewards[ep_idx, transition], dtype=np.float32)
            terminations[destination:] = np.asarray(self.terminations[ep_idx, transition], dtype=bool)
            is_last[destination:] = np.logical_or(
                terminations[destination:],
                np.asarray(self.truncations[ep_idx, transition], dtype=bool),
            )

        obs.update({
            "action": actions,
            "reward": rewards,
            "is_first": np.zeros((length, 1), dtype=bool),
            "is_last": is_last,
            "is_terminal": terminations,
        })
        if length and start == 0:
            obs["is_first"][0] = True
        return obs


class DMCExpertSequenceReplay(DMCExpertDataset):
    """Sample fixed STORM sequences in current-observation convention."""

    def __init__(self, config):
        super().__init__(config)
        self.sequence_length = int(config.replay.sequence_length)
        self.episodes = self.episodes[self.lengths[self.episodes] >= self.sequence_length]
        self._validate_episode_count()
        self.reconstruct_context = str(config.storm_model.sequence_core) != "transformer"

    def sample_episode_batch(self):
        groups, batch = self._transition_batch(self.sequence_length)
        if not self.reconstruct_context:
            return batch

        contexts = []
        for ep_idx, starts in groups:
            context_length = max(starts)
            context = self._read_observations(ep_idx, 0, context_length)
            context["action"] = np.asarray(self.actions[ep_idx, :context_length], dtype=np.float32)
            context = TensorDict(
                {key: torch.as_tensor(value) for key, value in context.items()},
                batch_size=(context_length,),
            ).unsqueeze(0)
            contexts.append((context, starts))
        return contexts, batch


class DMCExpertTransitionReplay(DMCExpertDataset):
    """Sample short image-transition sequences for latent planning models."""

    def __init__(self, config):
        super().__init__(config)
        self.sequence_length = int(config.replay.sequence_length)
        transitions = self.sequence_length - 1
        self.episodes = self.episodes[self.lengths[self.episodes] >= transitions]
        self._validate_episode_count()
        if str(config.model_family) in {"leworldmodel", "temporal_straightening"}:
            goal = self.metadata.get("goal_relation")
            requested = config.jepa_model.goal
            if goal is None:
                raise ValueError(f"{self.path} has no goal-relation labels; recollect this expert dataset.")
            stored_tolerance = np.asarray(goal["tolerance"], dtype=np.float32)
            requested_tolerance = np.asarray(list(requested.tolerance), dtype=np.float32)
            if (
                str(goal["geometry"]) != str(requested.geometry)
                or stored_tolerance.shape != requested_tolerance.shape
                or not np.allclose(stored_tolerance, requested_tolerance)
            ):
                raise ValueError(
                    f"Dataset goal geometry {goal} does not match configured geometry "
                    f"{requested.geometry}/{list(requested.tolerance)}."
                )

    def sample_episode_batch(self):
        _, batch = self._transition_batch(self.sequence_length - 1)
        return batch


class DMCExpertFrameStackReplay(DMCExpertTransitionReplay):
    """Present raw expert images as causal frame stacks without changing the HDF5 data."""

    def __init__(self, config):
        self.frame_stack = int(config.tdmpc2_model.frame_stack)
        if self.frame_stack < 1:
            raise ValueError("frame_stack must be positive.")
        super().__init__(config)
