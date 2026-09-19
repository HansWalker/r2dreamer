"""Independent training-split samplers for native and detached-readout retention."""

import copy
import math
from contextlib import contextmanager

from dmc_expert.storage import dataset_identity


def native_mixture(config):
    fraction = float(config.training.online.get("expert_fraction", 0.0))
    if not 0 <= fraction < 1:
        raise ValueError("Native expert fraction must be in [0, 1).")
    if not fraction:
        return 0, 0
    if str(config.model_family) not in {"leworldmodel", "temporal_straightening"}:
        raise ValueError("Native expert mixing is currently supported only for the latent planners.")
    rows, sources = int(config.replay.batch_size), int(config.replay.episodes_per_batch)
    expert_rows, expert_sources = round(rows * fraction), round(sources * fraction)
    if (not math.isclose(expert_rows, rows * fraction) or not math.isclose(expert_sources, sources * fraction)
            or not 0 < expert_sources < sources or rows % sources):
        raise ValueError("Native expert mixing must split whole source episodes and preserve windows per episode.")
    history = int(config.jepa_model.history_size)
    available = (rows - expert_rows) * (int(config.replay.sequence_length) - history + 1)
    if available < int(config.state_head.samples_per_update):
        raise ValueError("Native mixing leaves too few online labels for the unchanged readout budget.")
    return expert_rows, expert_sources


@contextmanager
def online_native_replay(config, family, model, checkpoint=None, expected_dataset=None):
    rows, sources = native_mixture(config)
    replay = None
    try:
        if rows:
            settings = copy.deepcopy(config)
            settings.training.expert.batch_size = rows
            settings.replay.episodes_per_batch = sources
            settings.replay.seed = int(config.replay.seed) + 3_000_017
            replay = family.build_replay(settings)
            if expected_dataset is not None and dataset_identity(replay.metadata) != expected_dataset:
                raise ValueError("Native retention dataset differs from the checkpoint's expert dataset.")
            if (checkpoint or {}).get("phase") == "online":
                if "native_expert_replay_state" not in checkpoint:
                    raise ValueError("Online checkpoint is missing the native expert sampler state.")
                replay.load_state_dict(checkpoint["native_expert_replay_state"])
        model._online_expert_replay = replay
        yield replay
    finally:
        model._online_expert_replay = None
        if replay is not None:
            replay.close()


@contextmanager
def online_readout(config, family, model, checkpoint=None, expected_dataset=None):
    head = model.state_head
    resumed = (checkpoint or {}).get("phase") == "online"
    replay = None
    try:
        if head.expert_fraction:
            settings = copy.deepcopy(config)
            # Keep source episode diversity and native history, but decode only enough labels.
            length = int(config.replay.sequence_length) - int(str(config.model_family) == "storm")
            targets = length - head.history + 1
            if targets <= 0:
                raise ValueError("Expert readout sequences must contain the complete readout history.")
            sources = int(config.replay.episodes_per_batch)
            labels = int(head.samples_per_update * head.expert_fraction)
            settings.training.expert.batch_size = max(sources, math.ceil(labels / targets / sources) * sources)
            settings.replay.seed = int(config.replay.seed) + 2_000_003
            replay = family.build_replay(settings)
            if expected_dataset is not None and dataset_identity(replay.metadata) != expected_dataset:
                raise ValueError("Readout retention dataset differs from the checkpoint's expert dataset.")
            if resumed:
                state = checkpoint.get("readout_replay_state")
                if state is None:
                    raise ValueError("Online checkpoint is missing the readout expert sampler state.")
                replay.load_state_dict(state)
        source = (lambda: model.readout_features(replay.sample_episode_batch())) if replay is not None else None
        head.configure_online(source, resumed=resumed)
        with online_native_replay(config, family, model, checkpoint, expected_dataset):
            yield replay
    finally:
        head._expert_source = None
        head._online = False
        if replay is not None:
            replay.close()
