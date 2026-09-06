"""Model-family training implementations."""

import importlib

MODEL_FAMILIES = {
    "dreamer": "training.dreamer",
    "storm": "training.storm",
    "tdmpc2": "training.tdmpc2",
    "leworldmodel": "training.leworldmodel",
    "temporal_straightening": "training.temporal_straightening",
}


def load_model_family(name):
    return importlib.import_module(MODEL_FAMILIES[str(name)])
