"""TD-MPC2 hooks for the shared DMC trainer."""

from models.tdmpc2 import TDMPC2

from .planning import (
    ExpertReplay,
    OnlineSession,
    checkpoint,
    evaluate,
    expert_update,
    load_checkpoint,
)

__all__ = [
    "EXPERT_METRICS",
    "ONLINE_METRICS",
    "ExpertReplay",
    "OnlineSession",
    "build_model",
    "checkpoint",
    "evaluate",
    "expert_update",
    "load_checkpoint",
]

EXPERT_METRICS = {
    "loss": "loss",
    "consistency": "consistency_loss",
    "value": "value_loss",
    "policy": "policy_loss",
}
ONLINE_METRICS = EXPERT_METRICS


def build_model(config):
    model = TDMPC2(config, config.model_io).to(config.device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model
