"""Temporal Straightening architecture and image-specific components."""

import math
from itertools import chain

import torch
import torch.nn.functional as F
from torch import nn, optim

from models.planning import LatentPlanner
from models.shared.transformer import Attention, FeedForward
from models.shared.utils import parse_model_io
from models.shared.vision import channel_first, image_spec


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(action_dim, embedding_dim), nn.LayerNorm(embedding_dim))

    def forward(self, action):
        return self.net(action.float())


class ProprioEncoder(nn.Module):
    """Reference-style per-frame proprioceptive embedding."""

    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("std", torch.ones(input_dim))

    def set_stats(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, device=self.mean.device))
        self.std.copy_(torch.as_tensor(std, device=self.std.device))

    def forward(self, value):
        return self.norm(self.linear((value.float() - self.mean) / self.std))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.attention = Attention(dim, heads, dim_head, dropout)
        self.feed_forward = FeedForward(dim, mlp_dim, dropout)

    def forward(self, value, mask):
        value = value + self.attention(value, mask)
        return value + self.feed_forward(value)


class Predictor(nn.Module):
    def __init__(self, state_dim, action_dim, history, config, tokens_per_frame=1):
        super().__init__()
        self.state_dim = int(state_dim)
        dim = self.state_dim + int(action_dim)
        self.position = nn.Parameter(torch.randn(1, int(tokens_per_frame) * history, dim))
        self.dropout = nn.Dropout(float(config.dropout))
        self.blocks = nn.ModuleList(
            TransformerBlock(
                dim,
                int(config.heads),
                int(config.dim_head),
                int(config.mlp_dim),
                float(config.dropout),
            )
            for _ in range(int(config.layers))
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, state, action):
        single_token = state.ndim == 3
        if single_token:
            state = state.unsqueeze(-2)
        batch, frames, patches, _ = state.shape
        action = action.unsqueeze(-2).expand(-1, -1, patches, -1)
        tokens = torch.cat((state, action), dim=-1).reshape(batch, frames * patches, -1)
        if tokens.shape[1] > self.position.shape[1]:
            position = F.interpolate(
                self.position.transpose(1, 2),
                size=tokens.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            position = self.position[:, : tokens.shape[1]]
        tokens = self.dropout(tokens + position)
        frame = torch.arange(frames, device=state.device).repeat_interleave(patches)
        mask = (frame[:, None] >= frame[None, :])[None, None]
        for block in self.blocks:
            tokens = block(tokens, mask)
        output = self.norm(tokens).reshape(batch, frames, patches, -1)[..., : self.state_dim]
        return output.squeeze(-2) if single_token else output


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        if down:
            self.skip = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(in_channels, out_channels, 3, padding=1))
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
            )

    def forward(self, value):
        return self.skip(value) + self.main(value)


class SensoryEncoder(nn.Module):
    def __init__(self, model_io, config):
        super().__init__()
        observations, _, _ = parse_model_io(model_io)
        self.key, (height, width, channels) = image_spec(model_io)
        if "proprio" not in observations:
            raise ValueError("Temporal Straightening requires a proprioceptive observation.")
        self.keys = (self.key, "proprio")
        self.shapes = {key: observations[key] for key in self.keys}
        self.visual_dim = int(config.embedding_dim)
        proprio_dim = math.prod(observations["proprio"])
        proprio_embedding_dim = int(config.proprio_embedding_dim)
        self.out_dim = self.visual_dim + proprio_embedding_dim
        base = int(config.vision.base_channels)
        blocks = []
        in_channels = channels
        for out_channels in (base, 2 * base, 4 * base, 8 * base):
            blocks.extend(
                (ResidualBlock(in_channels, out_channels, down=True), ResidualBlock(out_channels, out_channels))
            )
            in_channels = out_channels
        blocks.append(ResidualBlock(in_channels, self.visual_dim))
        self.backbone = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(self.visual_dim)
        self.proprio = ProprioEncoder(proprio_dim, proprio_embedding_dim)
        self.grid = (height // 16, width // 16)
        self.num_tokens = math.prod(self.grid)

    def forward(self, obs):
        pixels, prefix = channel_first(obs[self.key])
        if obs[self.key].dtype == torch.uint8:
            pixels = pixels / 255.0
        features = self.backbone(2 * pixels - 1)
        tokens = self.norm(features.flatten(2).transpose(1, 2))
        visual = tokens.reshape(*prefix, tokens.shape[-2], self.visual_dim)
        proprio = self.proprio(obs["proprio"]).unsqueeze(-2).expand(*prefix, self.num_tokens, -1)
        return torch.cat((visual, proprio), dim=-1)

    @staticmethod
    def pool(latent):
        return latent.mean(dim=-2)

    def target(self, observation):
        value = observation[self.key].float()
        if observation[self.key].dtype == torch.uint8:
            value = value / 255.0
        return 2 * value - 1

    def set_proprio_stats(self, mean, std):
        self.proprio.set_stats(mean, std)


class DecoderResidualBlock(nn.Module):
    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1),
        )

    def forward(self, value):
        return value + self.layers(value)


class VisionDecoder(nn.Module):
    def __init__(self, latent_dim, image_shape, base_channels):
        super().__init__()
        height, width, channels = image_shape
        self.grid = (height // 16, width // 16)
        if math.prod(self.grid) < 1:
            raise ValueError(f"Image shape {image_shape} is too small for a four-stage decoder.")
        stage_channels = 4 * int(base_channels)
        residual_channels = max(1, stage_channels // 3)

        def stage(output_channels):
            return (
                nn.Conv2d(latent_dim, stage_channels, 3, padding=1),
                *(DecoderResidualBlock(stage_channels, residual_channels) for _ in range(4)),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(stage_channels, stage_channels // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(stage_channels // 2, output_channels, 4, stride=2, padding=1),
            )

        self.layers = nn.Sequential(*stage(latent_dim), *stage(channels))

    def forward(self, latent):
        prefix = latent.shape[:-2]
        value = latent.reshape(-1, *latent.shape[-2:]).transpose(1, 2)
        value = value.reshape(-1, latent.shape[-1], *self.grid)
        value = self.layers(value).permute(0, 2, 3, 1)
        return value.reshape(*prefix, *value.shape[-3:])


class TemporalStraightening(LatentPlanner):
    def __init__(self, config, model_io):
        settings = config.jepa_model
        observations, action_shape, _ = parse_model_io(model_io)
        encoder = SensoryEncoder(model_io, settings.encoder)
        latent_dim = encoder.out_dim
        action_embedding_dim = int(settings.predictor.action_embedding_dim)
        decoder = VisionDecoder(
            encoder.visual_dim,
            observations[encoder.key],
            int(settings.decoder.vision.base_channels),
        )
        super().__init__(
            config,
            model_io,
            Predictor(
                latent_dim,
                action_embedding_dim,
                int(settings.history_size),
                settings.predictor,
                tokens_per_frame=encoder.num_tokens,
            ),
            encoder,
            ActionEncoder(math.prod(action_shape), action_embedding_dim),
            goal_readout=nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(encoder.num_tokens * latent_dim, 2),
            ),
            decoder=decoder,
        )
        weight_decay = float(settings.optim.weight_decay)
        self.optimizers = {
            "encoder": optim.Adam(
                chain(self.encoder.backbone.parameters(), self.encoder.norm.parameters()),
                lr=float(settings.optim.encoder_lr),
            ),
            "predictor": optim.AdamW(
                self.predictor.parameters(),
                lr=float(settings.optim.predictor_lr),
                weight_decay=weight_decay,
            ),
            "action_encoder": optim.AdamW(
                chain(self.action_encoder.parameters(), self.encoder.proprio.parameters()),
                lr=float(settings.optim.action_encoder_lr),
                weight_decay=weight_decay,
            ),
            "decoder": optim.Adam(self.decoder.parameters(), lr=float(settings.optim.decoder_lr)),
            "goal": optim.AdamW(
                self.goal_readout.parameters(),
                lr=float(settings.goal.lr),
                weight_decay=float(settings.goal.weight_decay),
            ),
        }
        self.curvature_weight = float(settings.curvature_weight)
        self.decoder_weight = float(settings.decoder.weight)

    def set_proprio_stats(self, mean, std):
        self.encoder.set_proprio_stats(mean, std)

    def representation_loss(self, obs, latent, action):
        prediction = self.predict(latent[:, :-1], action)
        target = latent[:, 1:].detach()
        prediction_loss = F.mse_loss(prediction, target)
        visual_prediction_loss = F.mse_loss(
            prediction[..., : self.encoder.visual_dim],
            target[..., : self.encoder.visual_dim],
        )
        proprio_prediction_loss = F.mse_loss(
            prediction[..., self.encoder.visual_dim :],
            target[..., self.encoder.visual_dim :],
        )
        visual = latent[..., : self.encoder.visual_dim]
        trajectory = visual.flatten(-2)
        velocity = trajectory[:, 1:] - trajectory[:, :-1]
        previous, current = velocity[:, :-1], velocity[:, 1:]
        curvature = 1 - F.cosine_similarity(previous, current, dim=-1, eps=1e-6)
        moving = (previous.norm(dim=-1) > 1e-6) & (current.norm(dim=-1) > 1e-6)
        curvature_loss = curvature[moving].mean() if moving.any() else curvature.new_zeros(())
        image = self.encoder.target(obs)
        reconstruction_loss = F.mse_loss(self.decoder(visual.detach()), image)
        predicted_reconstruction_loss = F.mse_loss(
            self.decoder(prediction[..., : self.encoder.visual_dim].detach()),
            image[:, 1:],
        )
        decoder_loss = reconstruction_loss + predicted_reconstruction_loss
        return prediction_loss + self.curvature_weight * curvature_loss + self.decoder_weight * decoder_loss, {
            "prediction_loss": prediction_loss,
            "visual_prediction_loss": visual_prediction_loss,
            "proprio_prediction_loss": proprio_prediction_loss,
            "curvature_loss": curvature_loss,
            "decoder_reconstruction_loss": reconstruction_loss,
            "decoder_prediction_loss": predicted_reconstruction_loss,
            "decoder_loss": decoder_loss,
        }
