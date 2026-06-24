import math
import queue
import threading
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import zarr
from tensordict import TensorDict


class DMCExpertReplay:
    """Sample Dreamer training windows from a DMC expert zarr dataset."""

    def __init__(self, config):
        self.path = Path(config.data_path).expanduser()
        self.device = torch.device(config.device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.warmup_length = int(config.warmup_length)
        self.total_length = self.warmup_length + self.batch_length
        self.shuffle = bool(getattr(config, "shuffle", True))
        self.prefetch = bool(getattr(config, "prefetch", True))
        self.prefetch_size = int(getattr(config, "prefetch_size", 2))
        self.rng = np.random.default_rng(int(config.seed))
        self._stop = threading.Event()
        self._queue = None
        self._thread = None

        self.root = zarr.open(str(self.path), mode="r")
        self.obs_dim = int(self.root.attrs["obs_dim"])
        self.action_dim = int(self.root.attrs["action_dim"])
        self.obs_keys = list(self.root.attrs["observation_keys"])
        self.obs_shapes = {
            key: tuple(self.root.attrs["observation_shapes"][key])
            for key in self.obs_keys
        }
        self.obs_slices = self._build_obs_slices()
        self.obs = np.asarray(self.root["obs"][:], dtype=np.float32)
        self.action = np.asarray(self.root["action"][:], dtype=np.float32)
        self.reward = np.asarray(self.root["reward"][:], dtype=np.float32)
        self.is_last = np.asarray(self.root["is_last"][:], dtype=bool)
        self.terminated = np.asarray(self.root["terminated"][:], dtype=bool)
        self.episode_start = np.asarray(self.root["episode_start"][:], dtype=np.int64)
        self.episode_length = np.asarray(self.root["episode_length"][:], dtype=np.int64)
        self.episodes = self._valid_episodes()
        if not self.episodes:
            raise ValueError(
                f"{self.path} has no episodes long enough for "
                f"warmup_length={self.warmup_length}, batch_length={self.batch_length}."
            )
        self.window_starts = self._build_window_starts()
        self._order = np.arange(len(self.window_starts), dtype=np.int64)
        self._pos = len(self._order)
        if self.prefetch:
            self._start_prefetch()

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

    def _valid_episodes(self):
        # Each training item needs obs_{t+1}, so the final raw transition in an episode
        # cannot be sampled because the dataset stores pre-step observations only.
        min_length = self.total_length + 1
        return [
            (int(start), int(length))
            for start, length in zip(self.episode_start, self.episode_length)
            if int(length) >= min_length
        ]

    def _build_window_starts(self):
        starts = []
        for ep_start, ep_length in self.episodes:
            count = ep_length - self.total_length
            starts.append(np.arange(ep_start, ep_start + count, dtype=np.int64))
        return np.concatenate(starts)

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
        low = np.asarray(self.root.attrs.get("action_min", [-1.0] * self.action_dim), dtype=np.float32)
        high = np.asarray(self.root.attrs.get("action_max", [1.0] * self.action_dim), dtype=np.float32)
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

    def sample(self):
        if self._queue is None:
            return self._sample_batch()
        item = self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def _sample_batch(self):
        warmup_rows = []
        train_rows = []
        for raw_start in self._next_starts():
            warmup_rows.append(self._make_window(raw_start, self.warmup_length))
            train_rows.append(self._make_window(raw_start + self.warmup_length, self.batch_length))

        warmup = self._stack(warmup_rows) if self.warmup_length else None
        train = self._stack(train_rows)
        return warmup, train

    def _next_starts(self):
        if self._pos + self.batch_size > len(self._order):
            self._pos = 0
            if self.shuffle:
                self.rng.shuffle(self._order)
        idx = self._order[self._pos : self._pos + self.batch_size]
        self._pos += self.batch_size
        return self.window_starts[idx]

    def _start_prefetch(self):
        self._queue = queue.Queue(maxsize=max(1, self.prefetch_size))
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()

    def _prefetch_loop(self):
        while not self._stop.is_set():
            try:
                item = self._sample_batch()
            except BaseException as exc:
                item = exc
            while not self._stop.is_set():
                try:
                    self._queue.put(item, timeout=0.1)
                    break
                except queue.Full:
                    pass

    def _make_window(self, raw_start, length):
        if length == 0:
            return None
        obs = self.obs[raw_start + 1 : raw_start + length + 1]
        action = self.action[raw_start : raw_start + length]
        reward = self.reward[raw_start : raw_start + length]
        is_last = self.is_last[raw_start : raw_start + length]
        terminated = self.terminated[raw_start : raw_start + length]

        data = self._split_obs(obs)
        data.update(
            {
                "action": action,
                "reward": reward.reshape(length, 1),
                "is_first": np.zeros((length, 1), dtype=bool),
                "is_last": is_last.reshape(length, 1),
                "is_terminal": terminated.reshape(length, 1),
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
            data[key] = torch.as_tensor(value, device=self.device)
        return TensorDict(data, batch_size=(self.batch_size, next(iter(data.values())).shape[1]))
