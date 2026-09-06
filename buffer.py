from collections import deque

import torch
from tensordict import TensorDict


def _to_device(data, device):
    if data.device is not None and data.device.type == "cpu" and device.type == "cuda":
        data = data.pin_memory()
    return data.to(device, non_blocking=True)


class EpisodeReplay:
    """Bounded replay containing raw, episode-aligned transitions."""

    def __init__(self, max_size, storage_device="cpu", seed=0):
        self.max_size = int(max_size)
        self.storage_device = torch.device(storage_device)
        self.completed = deque()
        self.current = None
        self._completed_size = 0
        self._generator = torch.Generator().manual_seed(int(seed))

    def start(self, env_num):
        self.current = [[] for _ in range(int(env_num))]

    def append(self, data, episode_end):
        if self.current is None:
            self.start(data.shape[0])
        data = data.detach().to(self.storage_device)
        episode_end = episode_end.detach().reshape(-1).to("cpu")
        for env_index in range(data.shape[0]):
            self.current[env_index].append(data[env_index].clone())
            if bool(episode_end[env_index]):
                episode = torch.stack(self.current[env_index], dim=0)
                self.completed.append(episode)
                self._completed_size += len(episode)
                self.current[env_index] = []
        self._trim()

    def _trim(self):
        while self.count() > self.max_size and self.completed:
            self._completed_size -= len(self.completed.popleft())

    def count(self):
        current_size = sum(len(episode) for episode in self.current or ())
        return self._completed_size + current_size

    def state_dict(self):
        # A simulator cannot resume mid-episode, so retain each usable prefix as
        # an explicit truncation and begin a new episode after restoration.
        truncated = []
        for rows in self.current or ():
            if not rows:
                continue
            episode = torch.stack(rows, dim=0)
            if "is_last" in episode:
                episode["is_last"][-1] = True
            truncated.append(episode)
        return {
            "completed": list(self.completed),
            "truncated": truncated,
            "generator_state": self._generator.get_state(),
        }

    def load_state_dict(self, state):
        self.completed = deque(
            episode.to(self.storage_device) for episode in (*state.get("completed", ()), *state.get("truncated", ()))
        )
        self._completed_size = sum(len(episode) for episode in self.completed)
        self.current = None
        if "generator_state" in state:
            self._generator.set_state(state["generator_state"].cpu())
        self._trim()

    def ready(self, length, episode_count=1):
        usable = sum(len(episode) >= length for episode in self.completed)
        usable += sum(len(episode) >= length for episode in self.current or ())
        return usable >= int(episode_count)

    def episodes(self, min_length=1):
        episodes = [episode for episode in self.completed if len(episode) >= min_length]
        episodes.extend(torch.stack(episode, dim=0) for episode in self.current or () if len(episode) >= min_length)
        return episodes

    def sample_groups(self, batch_size, sequence_length, episodes_per_batch):
        batch_size = int(batch_size)
        sequence_length = int(sequence_length)
        candidates = self.episodes(sequence_length)
        group_count = int(episodes_per_batch)
        if len(candidates) < group_count:
            raise RuntimeError(f"Replay has {len(candidates)} usable episodes, but this update requires {group_count}.")

        weights = torch.tensor(
            [len(episode) - sequence_length + 1 for episode in candidates],
            dtype=torch.float64,
        )
        selected = torch.multinomial(
            weights,
            group_count,
            replacement=False,
            generator=self._generator,
        )

        groups = []
        windows_per_episode = batch_size // group_count
        for candidate_index in selected.tolist():
            episode = candidates[candidate_index]
            valid_starts = len(episode) - sequence_length + 1
            if valid_starts >= windows_per_episode:
                starts = torch.randperm(valid_starts, generator=self._generator)[:windows_per_episode]
            else:
                starts = torch.randint(valid_starts, (windows_per_episode,), generator=self._generator)
            groups.append((episode, tuple(sorted(starts.tolist()))))
        return groups

    @staticmethod
    def stack_windows(groups, sequence_length):
        return torch.stack(
            [episode[start : start + sequence_length] for episode, starts in groups for start in starts],
            dim=0,
        )


class Buffer(EpisodeReplay):
    """Dreamer replay that rebuilds recurrent state from raw episode prefixes."""

    _transition_keys = ("image", "action", "reward", "is_first", "is_last", "is_terminal")

    def __init__(self, config):
        self.device = torch.device(config.device)
        self.batch_size = int(config.batch_size)
        self.sequence_length = int(config.sequence_length)
        self.episodes_per_batch = int(config.episodes_per_batch)
        super().__init__(
            config.max_size,
            storage_device=config.storage_device,
            seed=config.seed,
        )

    def ready(self):
        return super().ready(self.sequence_length, self.episodes_per_batch)

    def add_transition(self, data):
        raw = data.select(*self._transition_keys)
        self.append(raw, raw["is_last"])

    @staticmethod
    def _previous_actions(episode, start, length):
        actions = episode["action"]
        if length == 0:
            return actions[:0]
        if start:
            return actions[start - 1 : start + length - 1]
        return torch.cat((torch.zeros_like(actions[:1]), actions[: length - 1]), dim=0)

    def _aligned_slice(self, episode, start, length):
        sequence = episode[start : start + length].clone()
        sequence["action"] = self._previous_actions(episode, start, length)
        return sequence

    def sample(self):
        groups = self.sample_groups(
            self.batch_size,
            self.sequence_length,
            self.episodes_per_batch,
        )
        contexts = []
        windows = []
        for episode, starts in groups:
            context_length = max(starts)
            context = self._aligned_slice(episode, 0, context_length).unsqueeze(0)
            contexts.append((_to_device(context, self.device), starts))
            windows.extend(self._aligned_slice(episode, start, self.sequence_length) for start in starts)
        return contexts, _to_device(torch.stack(windows, dim=0), self.device)


class SequenceBuffer(EpisodeReplay):
    """Raw observation-action sequences for STORM and planning models."""

    def __init__(self, config):
        self.batch_size = int(config.batch_size)
        self.sequence_length = int(config.sequence_length)
        self.device = torch.device(config.device)
        self.episodes_per_batch = int(config.episodes_per_batch)
        self._obs_keys = None
        self._has_goal_relation = False
        super().__init__(
            config.max_size,
            storage_device=config.storage_device,
            seed=config.seed,
        )

    def ready(self):
        return super().ready(self.sequence_length, self.episodes_per_batch)

    def append(self, obs, action, reward, terminal, episode_end, goal_relation=None):
        if self._obs_keys is None:
            self._obs_keys = tuple(obs.keys())
            self._has_goal_relation = goal_relation is not None
        data = {
            **{key: value for key, value in obs.items()},
            "action": action,
            "reward": reward,
            "terminal": terminal.reshape(-1, 1),
        }
        if goal_relation is not None:
            data["goal_relation"] = goal_relation
        transition = TensorDict(
            data,
            batch_size=(action.shape[0],),
        )
        super().append(transition, episode_end)

    def state_dict(self):
        return {
            "obs_keys": self._obs_keys,
            "has_goal_relation": self._has_goal_relation,
            "replay": super().state_dict(),
        }

    def load_state_dict(self, state):
        self._obs_keys = tuple(state["obs_keys"]) if state.get("obs_keys") else None
        self._has_goal_relation = bool(state.get("has_goal_relation", False))
        super().load_state_dict(state["replay"])

    def sample(self, batch_size=None, sequence_length=None, with_context=False):
        batch_size = int(batch_size or self.batch_size)
        sequence_length = int(sequence_length or self.sequence_length)
        groups = self.sample_groups(batch_size, sequence_length, self.episodes_per_batch)
        batch = _to_device(self.stack_windows(groups, sequence_length), self.device)
        obs = {key: batch[key] for key in self._obs_keys}
        result = (
            obs,
            batch["action"].float(),
            batch["reward"].float(),
            batch["terminal"].float(),
        )
        if self._has_goal_relation:
            result += (batch["goal_relation"].float(),)
        if not with_context:
            return result
        contexts = [
            (
                _to_device(
                    episode[: max(starts)].select(*self._obs_keys, "action").unsqueeze(0),
                    self.device,
                ),
                starts,
            )
            for episode, starts in groups
        ]
        return contexts, result
