"""TD-MPC2 hooks for the shared DMC trainer."""

from dmc_expert.replay import DMCExpertFrameStackReplay
from models.tdmpc2 import TDMPC2

from . import planning

ExpertReplay = DMCExpertFrameStackReplay
OnlineSession = planning.OnlineSession
checkpoint = planning.checkpoint
evaluate = planning.evaluate
expert_update = planning.expert_update
load_checkpoint = planning.load_checkpoint

EXPERT_METRICS = {
    "loss": "loss",
    "consistency": "consistency_loss",
    "value": "value_loss",
    "policy": "policy_loss",
}
ONLINE_METRICS = EXPERT_METRICS


def build_model(config):
    return TDMPC2(config, config.model_io).to(config.device)
