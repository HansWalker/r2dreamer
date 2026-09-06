"""STORM's original convolutional observation edge."""

import math

import torch
from torch import nn

from models.shared.vision import channel_first, image_spec


class StormImageEncoder(nn.Module):
    def __init__(self, obs_shapes, config):
        super().__init__()
        self.key, (height, width, channels) = image_spec(obs_shapes)
        stem = int(config.vision.stem_channels)
        stages = int(math.log2(min(height, width))) - 2
        layers = []
        in_channels = channels
        out_channels = stem
        for _ in range(stages):
            layers.extend((
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))
            in_channels = out_channels
            out_channels *= 2
        self.layers = nn.Sequential(*layers)
        self.out_dim = in_channels * 4 * 4

    def forward(self, obs):
        value, prefix = channel_first(obs[self.key])
        if obs[self.key].dtype == torch.uint8:
            value = value / 255.0
        value = self.layers(value).flatten(1)
        return value.reshape(*prefix, self.out_dim)


class StormImageDecoder(nn.Module):
    def __init__(self, obs_shapes, stoch_dim, config):
        super().__init__()
        self.key, (height, width, channels) = image_spec(obs_shapes)
        stem = int(config.vision.stem_channels)
        stages = int(math.log2(min(height, width))) - 2
        last_channels = stem * 2 ** (stages - 1)
        layers = [
            nn.Linear(stoch_dim, last_channels * 4 * 4, bias=False),
            nn.Unflatten(-1, (last_channels, 4, 4)),
            nn.BatchNorm2d(last_channels),
            nn.ReLU(inplace=True),
        ]
        current = last_channels
        for _ in range(stages - 1):
            layers.extend((
                nn.ConvTranspose2d(current, current // 2, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(current // 2),
                nn.ReLU(inplace=True),
            ))
            current //= 2
        layers.append(nn.ConvTranspose2d(current, channels, 4, stride=2, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, stoch):
        prefix = stoch.shape[:-1]
        value = self.layers(stoch.reshape(-1, stoch.shape[-1])).permute(0, 2, 3, 1)
        return {self.key: value.reshape(*prefix, *value.shape[-3:])}

    def reconstruction_loss(self, prediction, observation):
        target = observation[self.key].float()
        if observation[self.key].dtype == torch.uint8:
            target = target / 255.0
        return (prediction[self.key] - target).square().mean()
