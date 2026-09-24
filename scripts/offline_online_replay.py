"""Gradually hand native training from original data to accumulated experience.

Expert HDF5 and TRAIN action branches are read-only. The union is logical, so
adding online trajectories never copies or rewrites the original dataset.
"""

import copy
import math
from contextlib import contextmanager

import numpy as np
import torch

from dmc_expert.storage import dataset_identity
from models.shared.physical_state import STATE_KEY
from training.planning import OnlineSession


def replay_settings(config, bank, coverage_fraction, updates):
    if float(config.training.online.get("expert_fraction", 0.)):
        raise ValueError("Scheduled sampling cannot also enable the separate fixed expert mixture.")
    if updates < 2:
        raise ValueError("The original-to-online handover requires at least two updates.")
    rows, sources = int(config.replay.batch_size), int(config.replay.episodes_per_batch)
    length, history = int(config.replay.sequence_length), int(config.jepa_model.history_size)
    horizon = int(config.jepa_model.get("training_horizon", 1))
    if rows < 2 or sources < 1 or rows % sources or horizon < 1 or length != history + horizon:
        raise ValueError("Require whole readout source groups and history_size + training_horizon frames.")
    if rows * (length - history + 1) < int(config.state_head.samples_per_update):
        raise ValueError("The full online readout batch cannot supply its unchanged label budget.")
    cases = bank["splits"]["train"]
    if not cases or any(case["split"] != "train" for case in cases):
        raise ValueError("Only original TRAIN anchors may enter native replay.")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Original training anchor IDs must be unique.")
    if any(len(case["prefix"]) + case["image"].shape[1] < length for case in cases):
        raise ValueError("Original branches must contain complete native training windows.")
    return {
        "mode": "offline_plus_online", "sampling": "linear_offline_decay",
        "offline_start_fraction": .5, "offline_end_fraction": 0.,
        "decay_updates": int(updates), "decay_unit": "native_updates_in_this_run_after_warmup",
        "batch_rounding": "nearest_integer_half_up",
        "within_pool_sampling": "uniform_valid_windows_with_replacement",
        "batch_sequences": rows, "sequence_length": length,
        "source_offline_branch_fraction": coverage_fraction,
        "fixed_source_fractions": False, "fixed_native_episode_groups": False,
        "readout_online_sequences": rows, "readout_source_episodes": sources,
        "branch_split": "train", "train_anchor_ids": [case["id"] for case in cases],
        "online_capacity": int(config.replay.max_size),
        "scope": "Original quota decays from 50% to zero; uniform windows within original and online pools.",
    }


class ScheduledReplay:
    """Fixed batch quotas taper original data; windows are uniform within each pool."""

    sources = ("expert", "branch", "online")

    def __init__(self, expert, bank, batch_size, sequence_length, seed, updates, start_fraction=.5):
        if updates < 2 or not 0 < start_fraction < 1 or batch_size < 2:
            raise ValueError("Require at least two updates/rows and an initial fraction between zero and one.")
        self.expert = expert
        self.batch_size, self.length = int(batch_size), int(sequence_length)
        self.updates, self.start_fraction = int(updates), float(start_fraction)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.expert_episodes = np.asarray(expert.episodes, dtype=np.int64)
        # HDF5 stores N transitions and N+1 observations; online buffers store N
        # observed current frames, so their valid-start counts differ by one.
        self.expert_ends = np.cumsum(expert.lengths[self.expert_episodes] - self.length + 2)
        self.branches = [(case, branch) for case in bank["splits"]["train"]
                         for branch in range(len(case["action"]))]
        if any(case["split"] != "train" for case, _ in self.branches):
            raise ValueError("Validation/test branches cannot enter replay.")
        self.branch_ends = np.cumsum([len(case["prefix"]) + case["image"].shape[1] - self.length + 1
                                     for case, _ in self.branches])
        if (not len(self.expert_ends) or not len(self.branch_ends)
                or np.any(np.diff(np.r_[0, self.expert_ends]) <= 0)
                or np.any(np.diff(np.r_[0, self.branch_ends]) <= 0)):
            raise ValueError("Original pools must contain usable training windows.")
        self.total_samples = dict.fromkeys(self.sources, 0)
        self.last_draws = []

    @staticmethod
    def locate(ends, index):
        episode = int(np.searchsorted(ends, index, side="right"))
        return episode, int(index - (ends[episode - 1] if episode else 0))

    def inventory(self, online):
        return {"expert": int(self.expert_ends[-1]), "branch": int(self.branch_ends[-1]),
                "online": sum(len(episode) - self.length + 1 for episode in online.episodes(self.length))}

    def quota(self, update_index):
        if update_index < 0:
            raise ValueError("Update index must be nonnegative.")
        progress = min(update_index / (self.updates - 1), 1.)
        fraction = self.start_fraction * (1. - progress)
        return fraction, math.floor(self.batch_size * fraction + .5)

    def sample(self, online, update_index):
        episodes = online.episodes(self.length)
        online_ends = np.cumsum([len(episode) - self.length + 1 for episode in episodes])
        counts = {"expert": int(self.expert_ends[-1]), "branch": int(self.branch_ends[-1]),
                  "online": int(online_ends[-1]) if len(online_ends) else 0}
        fraction, original_rows = self.quota(update_index)
        online_rows = self.batch_size - original_rows
        if not counts["online"]:
            raise RuntimeError("Collect usable online sequences before starting the handover; no offline-only fallback.")
        original_windows = counts["expert"] + counts["branch"]
        draws = torch.cat((
            torch.randint(original_windows, (original_rows,), generator=self.generator),
            original_windows + torch.randint(counts["online"], (online_rows,), generator=self.generator),
        ))
        # Shuffling keeps the minibatch layout independent of source membership.
        draws = draws[torch.randperm(self.batch_size, generator=self.generator)].tolist()
        probabilities = {"expert": original_rows / self.batch_size * counts["expert"] / original_windows,
                         "branch": original_rows / self.batch_size * counts["branch"] / original_windows,
                         "online": online_rows / self.batch_size}
        rows, actions, sampled, self.last_draws = [], [], dict.fromkeys(self.sources, 0), []
        for index in draws:
            if index < counts["expert"]:
                ordinal, start = self.locate(self.expert_ends, index)
                episode = int(self.expert_episodes[ordinal])
                obs = {key: torch.as_tensor(value) for key, value in
                       self.expert._read_observations(episode, start, self.length).items()}
                action = torch.as_tensor(self.expert.actions[episode, start:start + self.length - 1]).float()
                source, identity = "expert", episode
            elif index < counts["expert"] + counts["branch"]:
                ordinal, start = self.locate(self.branch_ends, index - counts["expert"])
                case, branch = self.branches[ordinal]
                obs = {"image": torch.cat((case["prefix"], case["image"][branch]))[start:start + self.length],
                       STATE_KEY: torch.cat((case["prefix_state"], case["states"][branch]))[start:start + self.length]}
                action = torch.cat((case["past_action"], case["action"][branch]))[start:start + self.length - 1].float()
                source, identity = "branch", (case["id"], branch)
            else:
                ordinal, start = self.locate(online_ends, index - counts["expert"] - counts["branch"])
                window = episodes[ordinal][start:start + self.length]
                obs = {key: window[key] for key in ("image", STATE_KEY)}
                action = window["action"][:-1].float()
                source, identity = "online", ordinal
            rows.append(obs)
            actions.append(action)
            sampled[source] += 1
            self.last_draws.append((source, identity, start))
        batch = ({key: torch.stack([row[key] for row in rows]).to(online.device) for key in rows[0]},
                 torch.stack(actions).to(online.device))
        metrics = {"replay/offline_target_fraction": fraction,
                   "replay/offline_fraction": original_rows / self.batch_size,
                   "replay/decay_progress": min(update_index / (self.updates - 1), 1.),
                   "replay/sampling_update": update_index + 1}
        for source in self.sources:
            self.total_samples[source] += sampled[source]
            metrics.update({f"native/{source}_sequences": sampled[source],
                            f"native/{source}_sequences_total": self.total_samples[source],
                            f"replay/{source}_windows": counts[source],
                            f"replay/{source}_fraction": probabilities[source],
                            f"replay/{source}_pool_fraction": counts[source] / sum(counts.values())})
        metrics["native/offline_sequences"] = sampled["expert"] + sampled["branch"]
        return batch, metrics


@contextmanager
def original_replay(config, family, bank, specification, expected_dataset):
    settings = copy.deepcopy(config)
    # Draw individual valid windows, not a fixed number of episodes.
    settings.replay.episodes_per_batch = 1
    with family.build_replay(settings) as expert:
        if dataset_identity(expert.metadata) != expected_dataset:
            raise ValueError("Original training dataset differs from the source checkpoint.")
        yield ScheduledReplay(expert, bank, specification["batch_sequences"],
                              specification["sequence_length"], int(config.replay.seed) + 3_000_017,
                              specification["decay_updates"], specification["offline_start_fraction"])


class OfflineOnlineSession(OnlineSession):
    """Unchanged real collection/native losses, with an original-to-online handover."""

    def __init__(self, config, model, envs, original):
        super().__init__(config, model, envs)
        self.original = original
        self.updates = 0

    def update(self, update_count):
        metrics = {}
        for _ in range(update_count):
            batch, sampling = self.original.sample(self.replay, self.updates)
            # Keep the full readout label budget throughout the native handover.
            obs, action, reward, terminal = self.replay.sample(sequence_length=self.model.sequence_length)
            readout = (obs, action[:, :-1], reward[:, :-1], terminal[:, :-1])
            metrics = self.model.update(batch, readout_batch=readout)
            self.updates += 1
            metrics.update(sampling)
        return metrics
