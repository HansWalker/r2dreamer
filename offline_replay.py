import json
import math
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
from tensordict import TensorDict


class _DMCExpertDataset:
    """Shared HDF5 loading and tensor formatting for DMC expert data."""

    def __init__(self, config):
        self.path = Path(config.data_path).expanduser()
        self.batch_size = int(config.batch_size)
        self.shuffle = bool(getattr(config, "shuffle", True))
        self.rng = np.random.default_rng(int(config.seed))

        self.root_path = self.path
        self.data_path = self.root_path / "data.hdf5"
        self.metadata_path = self.root_path / "metadata.json"
        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        if self.metadata.get("format") != "dmc_expert_hdf5_dense_v1":
            raise ValueError(
                f"{self.root_path} uses format={self.metadata.get('format')!r}; "
                "expected 'dmc_expert_hdf5_dense_v1'."
            )

        self.h5 = h5py.File(self.data_path, "r")
        self.observations = self.h5["observations"]
        self.actions = self.h5["actions"]
        self.rewards = self.h5["rewards"]
        self.terminations = self.h5["terminations"]
        self.truncations = self.h5["truncations"]
        self.lengths = np.asarray(self.h5["lengths"], dtype=np.int64)
        self.complete = np.asarray(self.h5["complete"], dtype=bool)

        self.obs_dim = int(self.metadata["obs_dim"])
        self.action_dim = int(self.metadata["action_dim"])
        self.obs_keys = list(self.metadata["observation_keys"])
        self.obs_shapes = {
            key: tuple(self.metadata["observation_shapes"][key])
            for key in self.obs_keys
        }
        self.obs_slices = self._build_obs_slices()
        self.complete_episodes = np.flatnonzero(self.complete & (self.lengths > 0))
        if len(self.complete_episodes) == 0:
            raise ValueError(f"{self.root_path} has no complete episodes.")

    def close(self):
        self.h5.close()

    def _build_obs_slices(self):
        out = {}
        offset = 0
        for key in self.obs_keys:
            shape = self.obs_shapes[key]
            dim = int(math.prod(shape)) if shape else 1
            out[key] = (slice(offset, offset + dim), shape or (1,))
            offset += dim
        if offset != self.obs_dim:
            raise ValueError(f"Observation metadata sums to {offset}, expected obs_dim={self.obs_dim}.")
        return out

    def obs_space(self):
        spaces = {
            key: gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
            for key, (_, shape) in self.obs_slices.items()
        }
        spaces.update(
            {
                "is_first": gym.spaces.Box(0, 1, (1,), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (1,), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=bool),
                "reward": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            }
        )
        return gym.spaces.Dict(spaces)

    def act_space(self):
        low = np.asarray(self.metadata.get("action_min", [-1.0] * self.action_dim), dtype=np.float32)
        high = np.asarray(self.metadata.get("action_max", [1.0] * self.action_dim), dtype=np.float32)
        if low.size == 1:
            low = np.full((self.action_dim,), float(low.reshape(-1)[0]), dtype=np.float32)
        if high.size == 1:
            high = np.full((self.action_dim,), float(high.reshape(-1)[0]), dtype=np.float32)
        if low.size != self.action_dim or high.size != self.action_dim:
            raise ValueError(
                f"Action bounds must be scalar or action_dim={self.action_dim}, "
                f"got low={low.shape}, high={high.shape}."
            )
        return gym.spaces.Box(low.reshape(-1), high.reshape(-1), dtype=np.float32)

    def _make_window(self, ep_idx, start, length):
        ep_idx = int(ep_idx)
        start = int(start)
        end = start + int(length)

        obs = np.asarray(self.observations[ep_idx, start + 1 : end + 1], dtype=np.float32)
        actions = np.asarray(self.actions[ep_idx, start:end], dtype=np.float32)
        rewards = np.asarray(self.rewards[ep_idx, start:end], dtype=np.float32)
        terminations = np.asarray(self.terminations[ep_idx, start:end], dtype=bool)
        truncations = np.asarray(self.truncations[ep_idx, start:end], dtype=bool)
        is_last = np.logical_or(terminations, truncations)

        data = self._split_obs(obs)
        data.update(
            {
                "action": actions,
                "reward": rewards.reshape(length, 1),
                "is_first": np.zeros((length, 1), dtype=bool),
                "is_last": is_last.reshape(length, 1),
                "is_terminal": terminations.reshape(length, 1),
            }
        )
        return data

    def _split_obs(self, obs):
        out = {}
        for key, (slc, shape) in self.obs_slices.items():
            out[key] = obs[:, slc].reshape(obs.shape[0], *shape).astype(np.float32)
        return out

    def _stack(self, rows):
        data = {}
        for key in rows[0]:
            value = np.stack([row[key] for row in rows], axis=0)
            data[key] = torch.as_tensor(value)
        return TensorDict(data, batch_size=(self.batch_size, next(iter(data.values())).shape[1]))


class DMCExpertEpisodeReplay(_DMCExpertDataset):
    """Sample complete expert episodes for teacher-forced pretraining."""

    def __init__(self, config):
        super().__init__(config)
        self.episodes = self.complete_episodes
        self.num_episodes = int(len(self.episodes))
        self.num_windows = 0
        self._episode_order = np.array([], dtype=np.int64)
        self._episode_pos = 0

    def sample_episode_batch(self):
        indices = self._next_episode_indices()
        lengths = self.lengths[indices]
        if not np.all(lengths == lengths[0]):
            raise RuntimeError(
                "Full-episode sampling requires equal-length episodes. "
                "Pad the dataset or use fixed-length DMC expert episodes."
            )
        rows = [self._make_episode(ep_idx) for ep_idx in indices]
        return self._stack(rows)

    def _next_episode_indices(self):
        rows = []
        while len(rows) < self.batch_size:
            if self._episode_pos >= len(self._episode_order):
                self._episode_order = (
                    self.rng.permutation(self.episodes)
                    if self.shuffle
                    else self.episodes.copy()
                )
                self._episode_pos = 0
            take = min(self.batch_size - len(rows), len(self._episode_order) - self._episode_pos)
            rows.extend(self._episode_order[self._episode_pos : self._episode_pos + take])
            self._episode_pos += take
        return np.asarray(rows, dtype=np.int64)

    def _make_episode(self, ep_idx):
        length = int(self.lengths[int(ep_idx)])
        data = self._make_window(ep_idx, 0, length)
        data["is_first"][0] = True
        data["is_last"][-1] = True
        return data
