"""Complete STORM model assembled from independent components."""

from __future__ import annotations

import math

from torch import nn

from models.shared.utils import parse_model_io

from .actor_critic import ActorCriticAgent
from .world_model import WorldModel


class StormModel(nn.Module):
    """Assemble the configured STORM sequence core, world model, and policy."""

    def __init__(self, config, model_io):
        super().__init__()
        obs_shapes, action_shape, _ = parse_model_io(model_io)
        action_dim = math.prod(action_shape)
        self.world_model = WorldModel(obs_shapes, action_dim, config.storm_model)
        self.actor_critic = ActorCriticAgent(
            self.world_model.feat_size,
            action_dim,
            config.actor_critic,
        )
