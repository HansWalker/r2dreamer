"""Temporal Straightening hooks for the shared DMC trainer."""

from models.temporal_straightening import TemporalStraightening

from .planning import ExpertReplay, checkpoint, evaluate, expert_update, load_checkpoint

__all__ = [
    "EXPERT_METRICS",
    "ExpertReplay",
    "build_model",
    "checkpoint",
    "configure_expert_replay",
    "evaluate",
    "expert_update",
    "load_checkpoint",
]

EXPERT_METRICS = {
    "loss": "loss",
    "prediction": "prediction_loss",
    "proprio": "proprio_prediction_loss",
    "curvature": "curvature_loss",
    "decoder": "decoder_loss",
    "goal": "goal_relation_loss",
}


def build_model(config):
    return TemporalStraightening(config, config.model_io).to(config.device)


def configure_expert_replay(model, replay):
    model.set_proprio_stats(replay.proprio_mean, replay.proprio_std)
