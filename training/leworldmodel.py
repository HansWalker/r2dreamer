"""LeWorldModel hooks for the shared DMC trainer."""

from models.leworldmodel import LeWorldModel

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
    "sigreg": "sigreg_loss",
    "goal": "goal_relation_loss",
}
ONLINE_METRICS = EXPERT_METRICS


def build_model(config):
    return LeWorldModel(config, config.model_io).to(config.device)
