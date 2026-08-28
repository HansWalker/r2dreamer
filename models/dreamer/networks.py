import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from models.shared import distributions as dists
from models.shared.utils import weight_init_
from models.shared.vision import image_spec


class BlockLinear(nn.Module):
    """Block-wise linear layer.

    Weight layout is chosen to cooperate with PyTorch's fan-in/fan-out
    calculation used by initializers.
    """

    def __init__(self, in_ch: int, out_ch: int, blocks: int):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.blocks = int(blocks)

        # Store weight in a layout that works with torch's fan calculation.
        # (O/G, I/G, G)
        self.weight = nn.Parameter(torch.empty(self.out_ch // self.blocks, self.in_ch // self.blocks, self.blocks))
        self.bias = nn.Parameter(torch.empty(self.out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., I)
        batch_shape = x.shape[:-1]
        # Reshape to expose block dimension.
        # (..., I) -> (..., G, I/G)
        x = x.view(*batch_shape, self.blocks, self.in_ch // self.blocks)

        # Block-wise multiplication
        # (..., G, I/G), (O/G, I/G, G) -> (..., G, O/G)
        x = torch.einsum("...gi,oig->...go", x, self.weight)
        # Merge block dimension back.
        # (..., G, O/G) -> (..., O)
        x = x.reshape(*batch_shape, self.out_ch)
        return x + self.bias


class Conv2dSamePad(nn.Conv2d):
    """A Conv2d layer that emulates TensorFlow's 'SAME' padding."""

    def _calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        i_div_s_ceil = (i + s - 1) // s
        return max((i_div_s_ceil - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        pad_h = self._calc_same_pad(ih, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = self._calc_same_pad(iw, self.kernel_size[1], self.stride[1], self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            )

        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class RMSNorm2D(nn.RMSNorm):
    """RMSNorm over channel-last format applied to 4D tensors."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply RMSNorm over the channel dimension.
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ImageEncoder(nn.Module):
    def __init__(self, config, shapes):
        super().__init__()
        self.key, shape = image_spec(shapes)
        self.encoder = ConvEncoder(config.cnn, shape)
        self.out_dim = self.encoder.out_dim
        self.apply(weight_init_)

    def forward(self, obs):
        return self.encoder(obs[self.key])


class ImageDecoder(nn.Module):
    def __init__(self, config, deter, flat_stoch, shapes):
        super().__init__()
        self.key, (height, width, channels) = image_spec(shapes)
        self.all_keys = [self.key]
        self.decoder = ConvDecoder(config.cnn, deter, flat_stoch, (channels, height, width))
        self.distribution = partial(getattr(dists, str(config.cnn_dist.name)), **config.cnn_dist)

    def forward(self, stoch, deter):
        return {self.key: self.distribution(self.decoder(stoch, deter))}


class ConvEncoder(nn.Module):
    def __init__(self, config, input_shape):
        super().__init__()
        act = getattr(torch.nn, config.act)
        height, width, input_ch = input_shape
        self.depths = tuple(int(config.depth) * int(mult) for mult in list(config.mults))
        final_height = height // 2 ** len(self.depths)
        final_width = width // 2 ** len(self.depths)
        self.kernel_size = int(config.kernel_size)
        in_dim = input_ch
        layers = []
        for depth in self.depths:
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=depth,
                    kernel_size=self.kernel_size,
                    stride=1,
                    bias=True,
                )
            )
            layers.append(nn.MaxPool2d(2, 2))
            if config.norm:
                layers.append(RMSNorm2D(depth, eps=1e-04, dtype=torch.float32))
            layers.append(act())
            in_dim = depth

        self.out_dim = self.depths[-1] * final_height * final_width
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        """Encode image-like observations with a CNN."""
        # (B, T, H, W, C)
        obs = obs - 0.5
        # (B*T, H, W, C)
        x = obs.reshape(-1, *obs.shape[-3:])
        # (B*T, C, H, W)
        x = x.permute(0, 3, 1, 2)
        # (B*T, C_feat, H_feat, W_feat)
        x = self.layers(x)
        # (B*T, C_feat*H_feat*W_feat)
        x = x.reshape(x.shape[0], -1)
        # (B, T, C_feat*H_feat*W_feat)
        return x.reshape(*obs.shape[:-3], x.shape[-1])


class ConvDecoder(nn.Module):
    def __init__(self, config, deter, flat_stoch, shape=(3, 64, 64)):
        super().__init__()
        act = getattr(torch.nn, config.act)
        self._shape = shape
        self.depths = tuple(int(config.depth) * int(mult) for mult in list(config.mults))
        self.min_shape = (
            shape[1] // 2 ** len(self.depths),
            shape[2] // 2 ** len(self.depths),
            self.depths[-1],
        )
        self.bspace = int(config.bspace)
        self.kernel_size = int(config.kernel_size)
        self.units = int(config.units)
        u, g = math.prod(self.min_shape), self.bspace
        self.sp0 = BlockLinear(deter, u, g)
        self.sp1 = nn.Sequential(
            nn.Linear(flat_stoch, 2 * self.units), nn.RMSNorm(2 * self.units, eps=1e-04, dtype=torch.float32), act()
        )
        self.sp2 = nn.Linear(2 * self.units, math.prod(self.min_shape))
        self.sp_norm = nn.Sequential(nn.RMSNorm(self.depths[-1], eps=1e-04, dtype=torch.float32), act())
        layers = []
        in_dim = self.depths[-1]
        for depth in reversed(self.depths[:-1]):
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            layers.append(Conv2dSamePad(in_dim, depth, self.kernel_size, stride=1, bias=True))
            layers.append(RMSNorm2D(depth, eps=1e-04, dtype=torch.float32))
            layers.append(act())
            in_dim = depth
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        layers.append(Conv2dSamePad(in_dim, self._shape[0], self.kernel_size, stride=1, bias=True))
        self.layers = nn.Sequential(*layers)
        self.apply(weight_init_)

    def forward(self, stoch, deter):
        """Decode latent states into images.

        Notes
        -----
        The decoder first constructs a low-resolution spatial feature map from
        the deterministic state (block-linear projection) and from the stochastic
        state (MLP projection), concats them, then upsamples back to the target
        resolution.
        """
        # (B, T, S, K), (B, T, D)
        B_T = deter.shape[:-1]
        # (B*T, D), (B*T, S*K)
        x0, x1 = deter.reshape(B_T.numel(), deter.shape[-1]), stoch.reshape(B_T.numel(), -1)

        # Spatial features from deterministic state
        # (H_feat, W_feat, C_feat)
        H_feat, W_feat, C_feat = self.min_shape
        # (B*T, H_feat*W_feat*C_feat)
        x0 = self.sp0(x0)
        # (B*T, G, H_feat, W_feat, C_feat/G)
        x0 = x0.reshape(-1, self.bspace, H_feat, W_feat, C_feat // self.bspace)
        # (B*T, H_feat, W_feat, C_feat)
        x0 = x0.permute(0, 2, 3, 1, 4).reshape(-1, H_feat, W_feat, C_feat)

        # Spatial features from stochastic state
        # (B*T, 2*U)
        x1 = self.sp1(x1)
        # (B*T, H_feat, W_feat, C_feat)
        x1 = self.sp2(x1).reshape(-1, H_feat, W_feat, C_feat)

        # Combine and upsample
        # (B*T, H_feat, W_feat, C_feat)
        x = self.sp_norm(x0 + x1)
        # (B*T, C_feat, H_feat, W_feat)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (B*T, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = torch.sigmoid(x)
        # (B, T, H, W, C)
        return x.reshape(*B_T, *x.shape[1:])


class ReturnEMA(nn.Module):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        super().__init__()
        self.alpha = alpha
        self.register_buffer(
            "range",
            torch.tensor([0.05, 0.95], dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer("ema_vals", torch.zeros(2, dtype=torch.float32, device=device))

    def forward(self, x):
        x_quantile = torch.quantile(torch.flatten(x.detach()), self.range)
        # Using out-of-place update for torch.compile compatibility
        self.ema_vals.copy_(self.alpha * x_quantile.detach() + (1 - self.alpha) * self.ema_vals)
        scale = torch.clip(self.ema_vals[1] - self.ema_vals[0], min=1.0)
        offset = self.ema_vals[0]
        return offset.detach(), scale.detach()
