"""Model-family training implementations."""

import importlib

MODEL_FAMILIES = {
    "dreamer": "training.dreamer",
    "storm": "training.storm",
    "tdmpc2": "training.planning",
    "leworldmodel": "training.planning",
    "temporal_straightening": "training.planning",
}


def load_model_family(name):
    return importlib.import_module(MODEL_FAMILIES[str(name)])
