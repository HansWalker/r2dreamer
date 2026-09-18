"""Training-only expert retention for the detached physical readout, not the native model."""

import copy
import math
from contextlib import contextmanager

from dmc_expert.storage import dataset_identity


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
        yield replay
    finally:
        head._expert_source = None
        head._online = False
        if replay is not None:
            replay.close()
