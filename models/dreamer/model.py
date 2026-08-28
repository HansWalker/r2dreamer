"""Complete Dreamer model assembled from independent components."""

from __future__ import annotations

import math

import torch
from torch import nn

from models.shared.utils import parse_model_io

from .actor_critic import DreamerActor, DreamerValue
from .heads import DreamerContinuationHead, DreamerRewardHead
from .networks import ImageDecoder, ImageEncoder
from .rssm import RSSM


class DreamerModel(nn.Module):
    """Assemble the configured Dreamer recurrent core and prediction modules."""

    def __init__(self, config, model_io):
        super().__init__()
        self.device = torch.device(config.device)
        obs_shapes, action_shape, action_kind = parse_model_io(model_io)
        self.act_dim = math.prod(action_shape)

        self.encoder = ImageEncoder(config.encoder, obs_shapes)
        self.rssm = RSSM(config.rssm, self.encoder.out_dim, self.act_dim)
        heads = config.heads
        self.reward = DreamerRewardHead(self.rssm.feat_size, heads)
        self.cont = DreamerContinuationHead(self.rssm.feat_size, heads)
        self.actor = DreamerActor(config.actor, self.rssm.feat_size, action_shape, action_kind)
        self.value = DreamerValue(config.critic, self.rssm.feat_size)
        self.decoder = ImageDecoder(
            config.decoder,
            self.rssm._deter,
            self.rssm.flat_stoch,
            obs_shapes,
        )
