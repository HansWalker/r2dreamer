import functools
import json
import math
import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def format_scalar(value, digits=1):
    if value is None:
        return "-"
    value = value.detach().item() if isinstance(value, torch.Tensor) else float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else "-"


def format_eta(seconds):
    if seconds < 0:
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


class Logger:
    def __init__(self, logdir):
        self._metrics = (logdir / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)
        self._writer = SummaryWriter(str(logdir))

    def write(self, step, scalars, console_message=None):
        scalars = {
            name: value.detach().item() if isinstance(value, torch.Tensor) else float(value)
            for name, value in scalars.items()
        }
        if console_message:
            print(console_message)
        self._metrics.write(json.dumps({"step": step, **scalars}) + "\n")
        for name, value in scalars.items():
            self._writer.add_scalar(name, value, step)
        self._writer.flush()

    def close(self):
        self._metrics.close()
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def configure_randomness(seed, deterministic=False):
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def get_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def preserve_rng_state(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        state = get_rng_state()
        try:
            return function(*args, **kwargs)
        finally:
            set_rng_state(state)

    return wrapped
