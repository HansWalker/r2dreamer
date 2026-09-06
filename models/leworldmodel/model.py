"""LeWorldModel architecture and image-specific components."""

import math

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
        self.net = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.Linear(action_dim, 4 * embedding_dim),
            nn.SiLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, action):
        return self.net(action.float())


class Projector(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, value):
        shape = value.shape
        value = self.net(value.reshape(-1, shape[-1]))
        return value.reshape(*shape[:-1], shape[-1])


class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.attention = Attention(dim, heads, dim_head, dropout)
        self.feed_forward = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(value, shift, scale):
        return value * (1 + scale) + shift

    def forward(self, value, condition):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).chunk(6, -1)
        value = value + gate1 * self.attention(self._modulate(self.norm1(value), shift1, scale1))
        return value + gate2 * self.feed_forward(self._modulate(self.norm2(value), shift2, scale2))


class Predictor(nn.Module):
    def __init__(self, dim, history, config):
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, history, dim))
        self.dropout = nn.Dropout(float(config.dropout))
        self.blocks = nn.ModuleList(
            ConditionalBlock(
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
        value = self.dropout(state + self.position[:, : state.shape[1]])
        for block in self.blocks:
            value = block(value, action)
        return self.norm(value)


class SIGReg(nn.Module):
    def __init__(self, knots=17, projections=1024):
        super().__init__()
        self.projections = int(projections)
        t = torch.linspace(0, 3, int(knots))
        dt = 3 / (int(knots) - 1)
        weights = torch.full((int(knots),), 2 * dt)
        weights[[0, -1]] = dt
        gaussian = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("gaussian", gaussian)
        self.register_buffer("weights", weights * gaussian)

    def forward(self, value):
        projection = torch.randn(value.shape[-1], self.projections, device=value.device)
        projection = projection / projection.norm(dim=0, keepdim=True)
        sample = (value @ projection).unsqueeze(-1) * self.t
        error = (sample.cos().mean(-3) - self.gaussian).square() + sample.sin().mean(-3).square()
        return ((error @ self.weights) * value.shape[-2]).mean()


class VisionBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, value):
        normalized = self.norm1(value)
        value = value + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return value + self.mlp(self.norm2(value))


class VisionEncoder(nn.Module):
    def __init__(self, model_io, config):
        super().__init__()
        self.key, (height, width, channels) = image_spec(model_io)
        self.keys = (self.key,)
        self.shapes = {self.key: (height, width, channels)}
        self.out_dim = int(config.embedding_dim)
        vision = config.vision
        patch = int(vision.patch_size)
        if height % patch or width % patch:
            raise ValueError(f"Image size {(height, width)} must be divisible by patch_size={patch}.")
        patches = height // patch * (width // patch)
        self.patch = nn.Conv2d(channels, self.out_dim, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, self.out_dim))
        self.position = nn.Parameter(torch.randn(1, patches + 1, self.out_dim) * 0.02)
        self.blocks = nn.Sequential(*(
            VisionBlock(
                self.out_dim,
                int(vision.heads),
                int(vision.mlp_dim),
                float(vision.dropout),
            )
            for _ in range(int(vision.layers))
        ))
        self.norm = nn.LayerNorm(self.out_dim)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1))

    def forward(self, obs):
        pixels, prefix = channel_first(obs[self.key])
        if obs[self.key].dtype == torch.uint8:
            pixels = pixels / 255.0
        pixels = (pixels - self.mean) / self.std
        tokens = self.patch(pixels).flatten(2).transpose(1, 2)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        output = self.norm(self.blocks(torch.cat((cls, tokens), dim=1) + self.position))[:, 0]
        return output.reshape(*prefix, self.out_dim)

    @staticmethod
    def pool(latent):
        return latent


class LeWorldModel(LatentPlanner):
    def __init__(self, config, model_io):
        settings = config.jepa_model
        encoder = VisionEncoder(model_io, settings.encoder)
        latent_dim = encoder.out_dim
        projector_dim = int(settings.projector_dim)
        super().__init__(
            config,
            model_io,
            Predictor(latent_dim, int(settings.history_size), settings.predictor),
            encoder,
            ActionEncoder(math.prod(parse_model_io(model_io)[1]), latent_dim),
            goal_readout=nn.Linear(latent_dim, 2),
            projector=Projector(latent_dim, projector_dim),
            pred_projector=Projector(latent_dim, projector_dim),
        )
        self.sigreg = SIGReg(
            knots=int(settings.sigreg.knots),
            projections=int(settings.sigreg.projections),
        )
        self.sigreg_weight = float(settings.sigreg.weight)
        self.model_lr = float(settings.optim.lr)
        goal_parameters = list(self.goal_readout.parameters())
        goal_ids = {id(parameter) for parameter in goal_parameters}
        self.optimizers = {
            "model": optim.AdamW(
                [parameter for parameter in self.parameters() if id(parameter) not in goal_ids],
                lr=self.model_lr,
                weight_decay=float(settings.optim.weight_decay),
            ),
            "goal": optim.AdamW(
                goal_parameters,
                lr=float(settings.goal.lr),
                weight_decay=float(settings.goal.weight_decay),
            ),
        }
        self.scheduler = None
        self._scheduler_state = None

    def optimizer_state_dict(self):
        state = super().optimizer_state_dict()
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        return state

    def load_optimizer_state_dict(self, state):
        super().load_optimizer_state_dict(state)
        self._scheduler_state = state.get("scheduler")

    def _configure_schedule(self, total_updates, resume):
        total_updates = int(total_updates)
        warmup = max(1, int(0.01 * total_updates))

        def scale(step):
            if step < warmup:
                return step / warmup
            progress = min(1.0, (step - warmup) / max(1, total_updates - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        optimizer = self.optimizers["model"]
        loaded_lrs = [group["lr"] for group in optimizer.param_groups]
        for group in optimizer.param_groups:
            group["lr"] = self.model_lr
            group["initial_lr"] = self.model_lr
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, scale)
        if resume and self._scheduler_state is not None:
            self.scheduler.load_state_dict(self._scheduler_state)
            for group, lr in zip(optimizer.param_groups, loaded_lrs, strict=True):
                group["lr"] = lr
        self._scheduler_state = None

    def configure_pretraining(self, total_updates):
        self._configure_schedule(total_updates, resume=self._scheduler_state is not None)

    def configure_online(self, total_updates, resumed=False):
        self._configure_schedule(total_updates, resume=resumed)

    def update(self, batch):
        metrics = super().update(batch)
        if self.scheduler is not None:
            self.scheduler.step()
            metrics["lr"] = self.optimizers["model"].param_groups[0]["lr"]
        return metrics

    def representation_loss(self, obs, latent, action):
        prediction = self.predict(latent[:, :-1], action)
        prediction_loss = F.mse_loss(prediction, latent[:, 1:])
        sigreg_loss = self.sigreg(latent.transpose(0, 1))
        return prediction_loss + self.sigreg_weight * sigreg_loss, {
            "prediction_loss": prediction_loss,
            "sigreg_loss": sigreg_loss,
        }
