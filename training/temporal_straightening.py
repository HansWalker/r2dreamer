"""Temporal Straightening hooks for the shared DMC trainer."""

from models.temporal_straightening import TemporalStraightening

from . import planning

ExpertReplay = planning.ExpertReplay
OnlineSession = planning.OnlineSession
checkpoint = planning.checkpoint
evaluate = planning.evaluate
expert_update = planning.expert_update
load_checkpoint = planning.load_checkpoint

EXPERT_METRICS = {
    "loss": "loss",
    "prediction": "prediction_loss",
    "proprio": "proprio_prediction_loss",
    "curvature": "curvature_loss",
    "decoder": "decoder_loss",
    "goal": "goal_relation_loss",
}
ONLINE_METRICS = EXPERT_METRICS


def build_model(config):
    return TemporalStraightening(config, config.model_io).to(config.device)


def configure_expert_replay(model, replay):
    model.set_proprio_stats(replay.proprio_mean, replay.proprio_std)
