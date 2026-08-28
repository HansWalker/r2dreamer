"""Dreamer actor and value networks."""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from models.shared import distributions as dists
from models.shared.utils import weight_init_


class _DreamerMLP(nn.Module):
    def __init__(self, config, input_dim: int):
        super().__init__()
        activation = getattr(nn, config.act)
        self.layers = nn.Sequential()
        for index in range(int(config.layers)):
            self.layers.add_module(
                f"{config.name}_linear{index}",
                nn.Linear(int(input_dim), int(config.units), bias=True),
            )
            self.layers.add_module(
                f"{config.name}_norm{index}",
                nn.RMSNorm(int(config.units), eps=1e-4, dtype=torch.float32),
            )
            self.layers.add_module(f"{config.name}_act{index}", activation())
            input_dim = int(config.units)
        self.out_dim = int(input_dim)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.layers(feature)


class _DreamerDistributionHead(nn.Module):
    def __init__(self, config, input_dim: int, shape: tuple[int, ...], dist_config):
        super().__init__()
        self.mlp = _DreamerMLP(config, input_dim)
        dist_name = str(dist_config.name)
        if dist_name == "bounded_normal":
            output_dim = int(shape[0]) * 2
            kwargs = {
                "min_std": float(dist_config.min_std),
                "max_std": float(dist_config.max_std),
            }
        elif dist_name == "symexp_twohot":
            output_dim = int(shape[0])
            kwargs = {"bin_num": int(dist_config.bin_num)}
        else:
            raise ValueError(f"Unsupported Dreamer distribution: {dist_name}")

        self.output = nn.Linear(self.mlp.out_dim, output_dim, bias=True)
        self._distribution = partial(getattr(dists, dist_name), **kwargs)
        self.apply(weight_init_)
        with torch.no_grad():
            self.output.weight.mul_(float(config.outscale))

    def forward(self, feature: torch.Tensor):
        return self._distribution(self.output(self.mlp(feature)))


class DreamerActor(_DreamerDistributionHead):
    def __init__(self, config, input_dim: int, action_shape, action_kind: str):
        if action_kind != "continuous":
            raise ValueError("DMC Dreamer requires continuous actions.")
        super().__init__(config, input_dim, tuple(action_shape), config.dist.cont)


class DreamerValue(_DreamerDistributionHead):
    def __init__(self, config, input_dim: int):
        super().__init__(
            config,
            input_dim,
            tuple(map(int, config.shape)),
            config.dist,
        )
