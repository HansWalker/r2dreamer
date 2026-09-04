"""Replay views over dense DMC expert datasets."""

import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from tensordict import TensorDict

from .storage import DATA_FORMAT, observation_indices


class DMCExpertDataset:
    """Shared HDF5 loading and tensor formatting for DMC expert data."""

    def __init__(self, config):
        settings = config.training.expert
        self.path = Path(settings.data_path).expanduser()
        self.batch_size = int(config.replay.batch_size)
        self.shuffle = bool(settings.shuffle)
        self.rng = np.random.default_rng(int(config.replay.seed))

        self.data_path = self.path / "data.hdf5"
        self.metadata_path = self.path / "metadata.json"
        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        if self.metadata.get("format") != DATA_FORMAT:
            raise ValueError(f"{self.path} uses format={self.metadata.get('format')!r}; expected {DATA_FORMAT!r}.")

        self.h5 = h5py.File(self.data_path, "r")
        self.observations = self.h5["observations"]
        self.images = self.h5["images"]
        self.actions = self.h5["actions"]
        self.rewards = self.h5["rewards"]
        self.terminations = self.h5["terminations"]
        self.truncations = self.h5["truncations"]
        self.goal_relations = self.h5.get("goal_relations")
        self.lengths = np.asarray(self.h5["lengths"], dtype=np.int64)
        self.complete = np.asarray(self.h5["complete"], dtype=bool)

        self.action_dim = int(self.metadata["action_dim"])
        requested = {str(key): tuple(map(int, shape)) for key, shape in config.model_io.observations.items()}
        if "image" not in requested or set(requested) - {"image", "proprio"}:
            raise ValueError(f"Expert replay expects image and optional proprioception, got {requested}.")
        self.proprio_indices = None
        if "proprio" in requested:
            self.proprio_indices = observation_indices(self.metadata, config.env.proprio)
            if requested["proprio"] != (len(self.proprio_indices),):
                raise ValueError(
                    f"Configured proprioception has shape {requested['proprio']}, "
                    f"but the selected DMC state has {len(self.proprio_indices)} values."
                )
        self.obs_keys = list(requested)
        self.obs_shapes = requested
        self._check_array_shapes()
        self.episodes = np.flatnonzero(self.complete & (self.lengths > 0))
        if len(self.episodes) == 0:
            raise ValueError(f"{self.path} has no complete episodes.")
        self.num_episodes = len(self.episodes)
        if self.proprio_indices is not None:
            self.proprio_mean, self.proprio_std = self._proprio_stats()
        self._episode_order = np.array([], dtype=np.int64)
        self._episode_pos = 0

    def close(self):
        self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _check_array_shapes(self):
        episode_shape = self.actions.shape[:2]
        expected_action = (*episode_shape, self.action_dim)
        if self.actions.shape != expected_action:
            raise ValueError(f"Action array has shape {self.actions.shape}, expected {expected_action}.")
        expected_step = (*episode_shape, 1)
        for name in ("rewards", "discounts", "terminations", "truncations"):
            shape = self.h5[name].shape
            if shape != expected_step:
                raise ValueError(f"{name} array has shape {shape}, expected {expected_step}.")
        if self.lengths.shape != episode_shape[:1] or self.complete.shape != episode_shape[:1]:
            raise ValueError("Episode metadata arrays do not match the HDF5 episode dimension.")
        expected_images = (*episode_shape[:1], episode_shape[1] + 1, *self.obs_shapes["image"])
        if self.images.shape != expected_images:
            raise ValueError(f"Image array has shape {self.images.shape}, expected {expected_images}.")
        expected_observations = (*episode_shape[:1], episode_shape[1] + 1, int(self.metadata["obs_dim"]))
        if self.observations.shape != expected_observations:
            raise ValueError(
                f"Observation array has shape {self.observations.shape}, expected {expected_observations}."
            )
        goal_relation = self.metadata.get("goal_relation")
        if goal_relation:
            expected = (*episode_shape[:1], episode_shape[1] + 1, *goal_relation["shape"])
            if self.goal_relations is None or self.goal_relations.shape != expected:
                shape = None if self.goal_relations is None else self.goal_relations.shape
                raise ValueError(f"Goal-relation array has shape {shape}, expected {expected}.")

        for name in ("action_min", "action_max"):
            size = np.asarray(self.metadata.get(name, [-1.0])).size
            if size not in (1, self.action_dim):
                raise ValueError(f"{name} must be scalar or action_dim={self.action_dim}, got {size} values.")

    def validate_model_io(self, model_io):
        observations = {str(key): tuple(map(int, shape)) for key, shape in model_io.observations.items()}
        action_dim = math.prod(model_io.action.shape)
        if observations != self.obs_shapes:
            raise ValueError(f"Configured observations {observations} do not match dataset metadata {self.obs_shapes}.")
        if str(model_io.action.kind) != "continuous" or action_dim != self.action_dim:
            raise ValueError(
                f"Configured action {model_io.action.kind}{tuple(model_io.action.shape)} does not match "
                f"continuous dataset action_dim={self.action_dim}."
            )

    def state_dict(self):
        return {
            "rng_state": self.rng.bit_generator.state,
            "episode_order": self._episode_order.copy(),
            "episode_pos": self._episode_pos,
        }

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng_state"]
        self._episode_order = np.asarray(state["episode_order"], dtype=np.int64)
        self._episode_pos = int(state["episode_pos"])

    def _next_episode_indices(self):
        indices = []
        while len(indices) < self.batch_size:
            if self._episode_pos >= len(self._episode_order):
                self._episode_order = self.rng.permutation(self.episodes) if self.shuffle else self.episodes.copy()
                self._episode_pos = 0
            count = min(
                self.batch_size - len(indices),
                len(self._episode_order) - self._episode_pos,
            )
            indices.extend(self._episode_order[self._episode_pos : self._episode_pos + count])
            self._episode_pos += count
        return np.asarray(indices, dtype=np.int64)

    def _read_observations(self, ep_idx, start, length):
        ep_idx, start, length = int(ep_idx), int(start), int(length)
        result = {"image": np.asarray(self.images[ep_idx, start : start + length])}
        if self.proprio_indices is not None:
            state = np.asarray(self.observations[ep_idx, start : start + length], dtype=np.float32)
            result["proprio"] = state[..., self.proprio_indices]
        return result

    def _proprio_stats(self):
        total = np.zeros(len(self.proprio_indices), dtype=np.float64)
        squared = np.zeros_like(total)
        count = 0
        steps = np.arange(self.observations.shape[1])
        for offset in range(0, len(self.episodes), 64):
            episodes = self.episodes[offset : offset + 64]
            state = np.asarray(self.observations[episodes], dtype=np.float32)[..., self.proprio_indices]
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
        if not len(self.episodes):
            raise ValueError(f"{self.path} has no complete episode with {self.sequence_length} observations.")
        self.num_episodes = len(self.episodes)

    def sample_episode_batch(self):
        indices = self._next_episode_indices()
        starts = [
            int(self.rng.integers(int(self.lengths[ep_idx]) - self.sequence_length + 2))
            for ep_idx in indices
        ]
        rows = [
            self._make_window(ep_idx, start, self.sequence_length)
            for ep_idx, start in zip(indices, starts, strict=True)
        ]
        data = {key: torch.as_tensor(np.stack([row[key] for row in rows], axis=0)) for key in rows[0]}
        batch = TensorDict(data, batch_size=(self.batch_size, self.sequence_length))
        contexts = []
        for ep_idx, start in zip(indices, starts, strict=True):
            row = self._make_window(ep_idx, 0, start)
            context = TensorDict(
                {key: torch.as_tensor(value) for key, value in row.items()},
                batch_size=(start,),
            ).unsqueeze(0)
            contexts.append((context, (start,)))
        return contexts, batch

    def _make_window(self, ep_idx, start, length):
        ep_idx = int(ep_idx)
        start, length = int(start), int(length)
        observations = self._read_observations(ep_idx, start, length)
        # Dreamer pairs each observation with the action and reward that led to it.
        actions = np.zeros((length, self.action_dim), dtype=np.float32)
        rewards = np.zeros((length, 1), dtype=np.float32)
        terminations = np.zeros((length, 1), dtype=bool)
        is_last = np.zeros((length, 1), dtype=bool)
        destination = 0 if start else 1
        source = max(start - 1, 0)
        count = length - destination
        if count:
            transition = slice(source, source + count)
            actions[destination:] = np.asarray(self.actions[ep_idx, transition], dtype=np.float32)
            rewards[destination:] = np.asarray(self.rewards[ep_idx, transition], dtype=np.float32)
            terminations[destination:] = np.asarray(self.terminations[ep_idx, transition], dtype=bool)
            is_last[destination:] = np.logical_or(
                terminations[destination:],
                np.asarray(self.truncations[ep_idx, transition], dtype=bool),
            )

        obs = observations
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
        if not len(self.episodes):
            raise ValueError(f"{self.path} has no complete episode of length {self.sequence_length}.")
        self.num_episodes = len(self.episodes)
        self.reconstruct_context = str(config.storm_model.sequence_core) != "transformer"

        reward = np.asarray(self.rewards[..., 0], dtype=np.float32)
        terminal = np.asarray(self.terminations[..., 0], dtype=np.float32)
        self.returns = np.zeros_like(reward)
        running = np.zeros(reward.shape[0], dtype=np.float32)
        gamma = float(config.actor_critic.gamma)
        for step in reversed(range(reward.shape[1])):
            valid = step < self.lengths
            running = np.where(valid, reward[:, step] + gamma * (1.0 - terminal[:, step]) * running, 0.0)
            self.returns[:, step] = running

    def sample_episode_batch(self):
        indices = self._next_episode_indices()
        samples = [self._sample_episode(ep_idx) for ep_idx in indices]
        starts, rows = zip(*samples, strict=True)
        data = {key: torch.as_tensor(np.stack([row[key] for row in rows], axis=0)) for key in rows[0]}
        obs = {key: data[key] for key in self.obs_keys}
        batch = (
            obs,
            data["action"].float(),
            data["reward"].float(),
            data["terminal"].float(),
            data["return"].float(),
        )
        if not self.reconstruct_context:
            return batch

        contexts = []
        for ep_idx, start in zip(indices, starts, strict=True):
            context = self._read_observations(ep_idx, 0, start)
            context["action"] = np.asarray(self.actions[int(ep_idx), :start], dtype=np.float32)
            context = TensorDict(
                {key: torch.as_tensor(value) for key, value in context.items()},
                batch_size=(start,),
            ).unsqueeze(0)
            contexts.append((context, (start,)))
        return contexts, batch

    def _sample_episode(self, ep_idx):
        ep_idx = int(ep_idx)
        length = self.sequence_length
        start = int(self.rng.integers(int(self.lengths[ep_idx]) - length + 1))
        end = start + length
        obs = self._read_observations(ep_idx, start, length)
        obs.update({
            "action": np.asarray(self.actions[ep_idx, start:end], dtype=np.float32),
            "reward": np.asarray(self.rewards[ep_idx, start:end], dtype=np.float32),
            "terminal": np.asarray(self.terminations[ep_idx, start:end], dtype=bool),
            "return": self.returns[ep_idx, start:end, None],
        })
        return start, obs


class DMCExpertTransitionReplay(DMCExpertDataset):
    """Sample short image-transition sequences for latent planning models."""

    def __init__(self, config):
        super().__init__(config)
        self.sequence_length = int(config.replay.sequence_length)
        transitions = self.sequence_length - 1
        self.episodes = self.episodes[self.lengths[self.episodes] >= transitions]
        if not len(self.episodes):
            raise ValueError(f"{self.path} has no complete episode with {transitions} transitions.")
        self.num_episodes = len(self.episodes)
        if str(config.model_family) in {"leworldmodel", "temporal_straightening"}:
            goal = self.metadata.get("goal_relation")
            requested = config.jepa_model.goal
            if self.goal_relations is None or goal is None:
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
        rows = [self._sample_episode(ep_idx) for ep_idx in self._next_episode_indices()]
        observations = {key: torch.as_tensor(np.stack([row[0][key] for row in rows])) for key in self.obs_keys}
        batch = (
            observations,
            torch.as_tensor(np.stack([row[1] for row in rows])).float(),
            torch.as_tensor(np.stack([row[2] for row in rows])).float(),
            torch.as_tensor(np.stack([row[3] for row in rows])).float(),
        )
        if self.goal_relations is None:
            return batch
        return (*batch, torch.as_tensor(np.stack([row[4] for row in rows])).float())

    def _sample_episode(self, ep_idx):
        transitions = self.sequence_length - 1
        episode_length = int(self.lengths[ep_idx])
        start = int(self.rng.integers(episode_length - transitions + 1))
        observations = self._read_observations(ep_idx, start, self.sequence_length)
        end = start + transitions
        row = (
            observations,
            np.asarray(self.actions[ep_idx, start:end], dtype=np.float32),
            np.asarray(self.rewards[ep_idx, start:end], dtype=np.float32),
            np.asarray(self.terminations[ep_idx, start:end], dtype=np.float32),
        )
        if self.goal_relations is None:
            return row
        relation = np.asarray(self.goal_relations[ep_idx, start : start + self.sequence_length], dtype=np.float32)
        return (*row, relation)
