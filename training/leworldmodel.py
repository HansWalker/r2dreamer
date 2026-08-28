"""LeWorldModel hooks for the shared DMC trainer."""

from models.leworldmodel import LeWorldModel

from .planning import ExpertReplay, checkpoint, evaluate, expert_update, load_checkpoint

__all__ = [
    "EXPERT_METRICS",
    "ExpertReplay",
    "build_model",
    "checkpoint",
    "evaluate",
    "expert_update",
    "load_checkpoint",
]

EXPERT_METRICS = {
    "loss": "loss",
    "prediction": "prediction_loss",
    "sigreg": "sigreg_loss",
    "goal": "goal_relation_loss",
}


def build_model(config):
    model = LeWorldModel(config, config.model_io).to(config.device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model
