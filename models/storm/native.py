"""Native STORM recipe components for continuous DMC control.

This file keeps the STORM-native experiment self contained inside this repo.
The intentional changes from upstream STORM are the DMC MLP observation edge,
continuous tanh-Gaussian actions, and optional Mamba sequence core.
"""

from __future__ import annotations

import copy
import math
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, OneHotCategorical

from models.dreamer.distributions import symlog, symexp

from .sequence_cores import MambaSequenceCore, TransformerSequenceCore
from .world_model import DistHead, MLPObservationEncoder, get_subsequent_mask_with_batch_length


def _activation(name: str):
    if name.lower() == "relu":
        return nn.ReLU
    return getattr(nn, name)


def mlp(in_dim: int, out_dim: int, hidden_dim: int, layers: int, *, act: str = "ReLU") -> nn.Sequential:
    act_cls = _activation(act)
    modules: list[nn.Module] = []
    dim = int(in_dim)
    for _ in range(int(layers)):
        modules.append(nn.Linear(dim, int(hidden_dim), bias=False))
        modules.append(nn.LayerNorm(int(hidden_dim)))
        modules.append(act_cls(inplace=True) if act_cls is nn.ReLU else act_cls())
        dim = int(hidden_dim)
    modules.append(nn.Linear(dim, int(out_dim)))
    return nn.Sequential(*modules)


class SymLogTwoHotLoss(nn.Module):
    def __init__(self, num_classes: int = 255, lower_bound: float = -20.0, upper_bound: float = 20.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.register_buffer(
            "bins",
            torch.linspace(self.lower_bound, self.upper_bound, self.num_classes),
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = symlog(target.to(dtype=torch.float32)).squeeze(-1)
        target = target.clamp(self.lower_bound, self.upper_bound)
        below = torch.sum(self.bins <= target.unsqueeze(-1), dim=-1) - 1
        above = torch.sum(self.bins < target.unsqueeze(-1), dim=-1)
        below = below.clamp(0, self.num_classes - 1)
        above = above.clamp(0, self.num_classes - 1)
        equal = below == above
        dist_below = torch.where(equal, torch.ones_like(target), (self.bins[below] - target).abs())
        dist_above = torch.where(equal, torch.ones_like(target), (self.bins[above] - target).abs())
        total = dist_below + dist_above
        weight_below = dist_above / total
        weight_above = dist_below / total
        target_prob = F.one_hot(below, self.num_classes).to(logits.dtype) * weight_below.unsqueeze(-1)
        target_prob = target_prob + F.one_hot(above, self.num_classes).to(logits.dtype) * weight_above.unsqueeze(-1)
        return -(target_prob * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        return symexp(F.softmax(logits.to(dtype=torch.float32), dim=-1) @ self.bins).unsqueeze(-1)


class CategoricalKLLossWithFreeBits(nn.Module):
    def __init__(self, free_bits: float = 1.0):
        super().__init__()
        self.free_bits = float(free_bits)

    def forward(self, p_logits: torch.Tensor, q_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).sum(dim=-1).mean()
        return torch.maximum(kl, torch.ones_like(kl) * self.free_bits), kl


class MLPObservationDecoder(nn.Module):
    def __init__(
        self,
        obs_shapes: Mapping[str, tuple[int, ...]],
        in_dim: int,
        hidden_dim: int,
        layers: int,
        *,
        act: str = "ReLU",
    ):
        super().__init__()
        excluded = {"is_first", "is_last", "is_terminal", "reward"}
        self.keys = tuple(k for k in obs_shapes if k not in excluded and not k.startswith("log_"))
        self.shapes = {key: tuple(obs_shapes[key]) for key in self.keys}
        self.heads = nn.ModuleDict(
            {
                key: mlp(in_dim, int(math.prod(self.shapes[key])), hidden_dim, layers, act=act)
                for key in self.keys
            }
        )

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        prefix = feat.shape[:-1]
        flat = feat.reshape(-1, feat.shape[-1])
        out = {}
        for key, head in self.heads.items():
            out[key] = head(flat).reshape(*prefix, *self.shapes[key])
        return out

    def reconstruction_loss(self, pred: Mapping[str, torch.Tensor], obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        losses = []
        for key in self.keys:
            target = obs[key].to(dtype=pred[key].dtype)
            loss = (pred[key] - target).pow(2)
            if len(loss.shape) > 2:
                loss = loss.reshape(*loss.shape[:2], -1).sum(dim=-1)
            losses.append(loss)
        return torch.stack(losses, dim=0).sum(dim=0).mean()


class WorldModel(nn.Module):
    def __init__(self, obs_shapes, action_dim: int, config):
        super().__init__()
        self.action_dim = int(action_dim)
        self.stoch_dim = int(config.stoch_dim)
        self.stoch_flattened_dim = self.stoch_dim * self.stoch_dim
        self.transformer_hidden_dim = int(config.hidden_dim)
        self.feat_size = self.stoch_flattened_dim + self.transformer_hidden_dim
        self.use_amp = bool(getattr(config, "use_amp", True))
        self.amp_dtype = torch.bfloat16 if str(getattr(config, "amp_dtype", "bfloat16")) == "bfloat16" else torch.float16
        self.grad_clip = float(getattr(config, "grad_clip", 1000.0))

        self.encoder = MLPObservationEncoder(
            obs_shapes,
            embedding_dim=self.transformer_hidden_dim,
            hidden_dim=int(config.encoder_hidden_dim),
            layers=int(config.encoder_layers),
            act=str(getattr(config, "act", "ReLU")),
        )
        core_name = str(getattr(config, "sequence_core", "transformer"))
        core_cls = TransformerSequenceCore if core_name == "transformer" else MambaSequenceCore
        kwargs = {}
        if core_name == "mamba":
            mcfg = getattr(config, "mamba", {})
            kwargs = {
                "d_state": int(getattr(mcfg, "d_state", 32)),
                "expand": int(getattr(mcfg, "expand", 2)),
                "headdim": int(getattr(mcfg, "headdim", 64)),
                "chunk_size": int(getattr(mcfg, "chunk_size", 16)),
                "is_mimo": bool(getattr(mcfg, "is_mimo", False)),
                "mimo_rank": int(getattr(mcfg, "mimo_rank", 1)),
                "is_outproj_norm": bool(getattr(mcfg, "is_outproj_norm", False)),
            }
        self.sequence_core = core_cls(
            stoch_dim=self.stoch_flattened_dim,
            action_dim=self.action_dim,
            feat_dim=self.transformer_hidden_dim,
            num_layers=int(config.num_layers),
            num_heads=int(config.num_heads),
            max_length=int(config.max_length),
            dropout=float(config.dropout),
            **kwargs,
        )
        self.dist_head = DistHead(
            encoder_feat_dim=self.encoder.out_dim,
            transformer_hidden_dim=self.transformer_hidden_dim,
            stoch_dim=self.stoch_dim,
        )
        self.decoder = MLPObservationDecoder(
            obs_shapes,
            in_dim=self.stoch_flattened_dim,
            hidden_dim=int(config.decoder_hidden_dim),
            layers=int(config.decoder_layers),
            act=str(getattr(config, "act", "ReLU")),
        )
        self.reward = mlp(self.transformer_hidden_dim, 255, int(config.head_hidden_dim), int(config.head_layers))
        self.termination = mlp(self.transformer_hidden_dim, 1, int(config.head_hidden_dim), int(config.head_layers))
        self.twohot = SymLogTwoHotLoss(255, -20, 20)
        self.kl = CategoricalKLLossWithFreeBits(float(config.kl_free))
        self.dyn_scale = float(config.dyn_scale)
        self.rep_scale = float(config.rep_scale)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(getattr(config, "lr", 1e-4)),
            eps=float(getattr(config, "eps", 1e-5)),
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and torch.cuda.is_available())

    @property
    def device(self):
        return next(self.parameters()).device

    def straight_through_gradient(self, logits: torch.Tensor, sample_mode: str = "random_sample") -> torch.Tensor:
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            return dist.sample() + dist.probs - dist.probs.detach()
        if sample_mode == "mode":
            return F.one_hot(torch.argmax(logits, dim=-1), logits.shape[-1]).to(logits.dtype)
        if sample_mode == "probs":
            return dist.probs
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    def flatten_sample(self, sample: torch.Tensor) -> torch.Tensor:
        return sample.reshape(*sample.shape[:-2], self.stoch_flattened_dim)

    def _encode_obs_with_logits(self, obs, sample_mode: str = "random_sample") -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.encoder(obs)
        post_logits = self.dist_head.forward_post(embed)
        sample = self.straight_through_gradient(post_logits, sample_mode=sample_mode)
        return self.flatten_sample(sample), post_logits

    def encode_obs(self, obs, sample_mode: str = "random_sample") -> torch.Tensor:
        sample, _ = self._encode_obs_with_logits(obs, sample_mode=sample_mode)
        return sample

    def observe(self, obs, action) -> dict[str, torch.Tensor]:
        stoch, post_logits = self._encode_obs_with_logits(obs)
        mask = get_subsequent_mask_with_batch_length(stoch.shape[1], stoch.device)
        deter = self.sequence_core(stoch, action, mask)
        prior_logits = self.dist_head.forward_prior(deter)
        feat = torch.cat([stoch, deter], dim=-1)
        return {"stoch": stoch, "deter": deter, "feat": feat, "post_logits": post_logits, "prior_logits": prior_logits}

    def loss(self, obs, action, reward, terminal) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp and self.device.type == "cuda"):
            state = self.observe(obs, action)
            recon = self.decoder(state["stoch"])
            recon_loss = self.decoder.reconstruction_loss(recon, obs)
            reward_logits = self.reward(state["deter"])
            terminal_logits = self.termination(state["deter"]).squeeze(-1)
            reward_loss = self.twohot(reward_logits, reward)
            terminal_loss = F.binary_cross_entropy_with_logits(terminal_logits, terminal.to(terminal_logits.dtype).squeeze(-1))
            dyn_loss, dyn_real = self.kl(state["post_logits"][:, 1:].detach(), state["prior_logits"][:, :-1])
            rep_loss, rep_real = self.kl(state["post_logits"][:, 1:], state["prior_logits"][:, :-1].detach())
            total = recon_loss + reward_loss + terminal_loss + self.dyn_scale * dyn_loss + self.rep_scale * rep_loss
        metrics = {
            "wm/loss": total.detach(),
            "wm/recon": recon_loss.detach(),
            "wm/reward": reward_loss.detach(),
            "wm/terminal": terminal_loss.detach(),
            "wm/dyn": dyn_loss.detach(),
            "wm/dyn_real": dyn_real.detach(),
            "wm/rep": rep_loss.detach(),
            "wm/rep_real": rep_real.detach(),
        }
        return total, metrics, state

    def update(self, obs, action, reward, termination):
        self.train()
        obs = {key: value.to(self.device, non_blocking=True) for key, value in obs.items()}
        action = action.to(self.device, non_blocking=True)
        reward = reward.to(self.device, non_blocking=True)
        termination = termination.to(self.device, non_blocking=True)
        loss, metrics, state = self.loss(obs, action, reward, termination)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return metrics, state, (obs, action, reward, termination)

    def calc_last_dist_feat(self, latent: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = get_subsequent_mask_with_batch_length(latent.shape[1], latent.device)
        deter = self.sequence_core(latent, action, mask)
        last_deter = deter[:, -1]
        prior_logits = self.dist_head.forward_prior(last_deter)
        prior = self.flatten_sample(self.straight_through_gradient(prior_logits))
        return prior, last_deter

    @torch.no_grad()
    def context_feature(self, obs, action) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode_obs(obs)
        prior, deter = self.calc_last_dist_feat(latent, action)
        return torch.cat([prior, deter], dim=-1), prior, deter

    @torch.no_grad()
    def imagine(self, actor_critic, context_obs, context_action, horizon: int) -> dict[str, torch.Tensor]:
        self.eval()
        actor_critic.eval()
        stoch = self.encode_obs(context_obs)
        cache = self.sequence_core.initial_cache(stoch.shape[0], dtype=stoch.dtype, device=stoch.device)
        deter = None
        for idx in range(stoch.shape[1]):
            deter, cache = self.sequence_core.forward_step_with_cache(
                stoch[:, idx : idx + 1],
                context_action[:, idx : idx + 1],
                cache,
            )
        assert deter is not None
        prior_logits = self.dist_head.forward_prior(deter[:, 0])
        stoch = self.flatten_sample(self.straight_through_gradient(prior_logits))
        deter = deter[:, 0]
        feats = [torch.cat([stoch, deter], dim=-1)]
        actions, rewards, terminals = [], [], []
        for _ in range(int(horizon)):
            action, _ = actor_critic.sample(feats[-1], deterministic=False)
            next_deter, cache = self.sequence_core.forward_step_with_cache(stoch.unsqueeze(1), action.unsqueeze(1), cache)
            next_deter = next_deter[:, 0]
            reward = self.twohot.decode(self.reward(next_deter))
            termination_logit = self.termination(next_deter)
            terminal = (torch.sigmoid(termination_logit) > 0.5).to(reward.dtype)
            prior_logits = self.dist_head.forward_prior(next_deter)
            stoch = self.flatten_sample(self.straight_through_gradient(prior_logits))
            deter = next_deter
            actions.append(action)
            rewards.append(reward)
            terminals.append(terminal)
            feats.append(torch.cat([stoch, deter], dim=-1))
        return {
            "feat": torch.stack(feats, dim=1),
            "action": torch.stack(actions, dim=1),
            "reward": torch.stack(rewards, dim=1),
            "terminal": torch.stack(terminals, dim=1),
        }

    @torch.no_grad()
    def imagine_data(
        self,
        agent,
        sample_obs,
        sample_action,
        imagine_batch_length: int,
    ):
        imagined = self.imagine(agent, sample_obs, sample_action, int(imagine_batch_length))
        return imagined["feat"], imagined["action"], imagined["reward"], imagined["terminal"]


class TanhNormal:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std
        self.normal = Normal(mean, std)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.normal.rsample()
        action = torch.tanh(raw)
        return action, self.log_prob(action, raw)

    def mode(self) -> torch.Tensor:
        return torch.tanh(self.mean)

    def log_prob(self, action: torch.Tensor, raw: torch.Tensor | None = None) -> torch.Tensor:
        eps = torch.finfo(action.dtype).eps
        action = action.clamp(-1 + 1e-6, 1 - 1e-6)
        if raw is None:
            raw = 0.5 * (torch.log1p(action) - torch.log1p(-action))
        log_prob = self.normal.log_prob(raw) - torch.log(1 - action.pow(2) + eps)
        return log_prob.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self.normal.entropy().sum(dim=-1)


def percentile(x: torch.Tensor, percentage: float) -> torch.Tensor:
    flat = x.reshape(-1)
    kth = max(1, int(float(percentage) * flat.numel()))
    return torch.kthvalue(flat, kth).values


def lambda_return(reward, value, termination, gamma: float, lambd: float) -> torch.Tensor:
    termination = termination.squeeze(-1) if termination.shape[-1] == 1 else termination
    reward = reward.squeeze(-1) if reward.shape[-1] == 1 else reward
    value = value.squeeze(-1) if value.shape[-1] == 1 else value
    cont = 1.0 - termination.to(value.dtype)
    returns = torch.zeros_like(value)
    returns[:, -1] = value[:, -1]
    for idx in reversed(range(reward.shape[1])):
        returns[:, idx] = reward[:, idx] + gamma * cont[:, idx] * (
            (1 - lambd) * value[:, idx] + lambd * returns[:, idx + 1]
        )
    return returns[:, :-1].unsqueeze(-1)


class EMAScalar:
    def __init__(self, decay: float = 0.99):
        self.decay = float(decay)
        self.value = None

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        value = value.detach()
        self.value = value if self.value is None else self.decay * self.value + (1 - self.decay) * value
        return self.value


class ActorCriticAgent(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, config) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.gamma = float(config.gamma)
        self.lambd = float(config.lambd)
        self.entropy_coef = float(config.entropy_coef)
        self.slow_critic_decay = float(getattr(config, "slow_critic_decay", 0.98))
        self.use_amp = bool(getattr(config, "use_amp", True))
        self.amp_dtype = torch.bfloat16 if str(getattr(config, "amp_dtype", "bfloat16")) == "bfloat16" else torch.float16
        self.grad_clip = float(getattr(config, "grad_clip", 100.0))

        hidden = int(config.hidden_dim)
        layers = int(config.layers)
        act = str(getattr(config, "act", "ReLU"))
        self.actor = mlp(feat_dim, 2 * self.action_dim, hidden, layers, act=act)
        self.critic = mlp(feat_dim, 255, hidden, layers, act=act)
        self.slow_critic = copy.deepcopy(self.critic)
        self.twohot = SymLogTwoHotLoss(255, -20, 20)
        self.lowerbound_ema = EMAScalar(0.99)
        self.upperbound_ema = EMAScalar(0.99)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(getattr(config, "lr", 3e-5)),
            eps=float(getattr(config, "eps", 1e-5)),
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and torch.cuda.is_available())

    @property
    def device(self):
        return next(self.parameters()).device

    def dist(self, feat: torch.Tensor) -> TanhNormal:
        raw = self.actor(feat)
        mean, log_std = torch.chunk(raw, 2, dim=-1)
        log_std = log_std.clamp(-5.0, 2.0)
        return TanhNormal(mean.to(torch.float32), torch.exp(log_std).to(torch.float32))

    @torch.no_grad()
    def slow_value(self, feat: torch.Tensor) -> torch.Tensor:
        return self.twohot.decode(self.slow_critic(feat))

    @torch.no_grad()
    def sample(self, feat: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.eval()
        dist = self.dist(feat)
        if deterministic:
            return dist.mode(), None
        return dist.sample()

    @torch.no_grad()
    def update_slow_critic(self, decay: float | None = None):
        decay = float(decay if decay is not None else self.slow_critic_decay)
        for slow, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow.data.copy_(slow.data * decay + param.data * (1 - decay))

    def update(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor | None = None,
        termination: torch.Tensor | None = None,
    ) -> Mapping[str, torch.Tensor]:
        self.train()
        assert reward is not None and termination is not None
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp and self.device.type == "cuda"):
            dist = self.dist(latent[:, :-1])
            raw_value = self.critic(latent)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            value = self.twohot.decode(raw_value)
            slow_value = self.slow_value(latent)
            returns = lambda_return(reward, value, termination, self.gamma, self.lambd)
            slow_returns = lambda_return(reward, slow_value, termination, self.gamma, self.lambd)

            value_loss = self.twohot(raw_value[:, :-1], returns.detach())
            slow_value_loss = self.twohot(raw_value[:, :-1], slow_returns.detach())
            low = self.lowerbound_ema(percentile(returns, 0.05))
            high = self.upperbound_ema(percentile(returns, 0.95))
            scale = torch.maximum(torch.ones_like(high), high - low)
            advantage = (returns - value[:, :-1]) / scale
            policy_loss = -(log_prob * advantage.detach().squeeze(-1)).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + value_loss + slow_value_loss - self.entropy_coef * entropy_loss

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.update_slow_critic()

        metrics = {
            "ac/loss": loss.detach(),
            "ac/policy": policy_loss.detach(),
            "ac/value": value_loss.detach(),
            "ac/slow_value": slow_value_loss.detach(),
            "ac/entropy": entropy_loss.detach(),
            "ac/return": returns.mean().detach(),
        }
        return metrics

    def update_expert(self, feat: torch.Tensor, action: torch.Tensor, returns: torch.Tensor) -> Mapping[str, torch.Tensor]:
        self.train()
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp and self.device.type == "cuda"):
            feat = feat.detach()
            dist = self.dist(feat)
            bc_loss = -dist.log_prob(action.to(torch.float32)).mean()
            value_logits = self.critic(feat)
            value_loss = self.twohot(value_logits, returns)
            loss = bc_loss + value_loss

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.update_slow_critic()

        metrics = {
            "expert/bc": bc_loss.detach(),
            "expert/value": value_loss.detach(),
            "expert/ac_loss": loss.detach(),
        }
        return metrics


StormNativeWorldModel = WorldModel
ContinuousActorCritic = ActorCriticAgent
