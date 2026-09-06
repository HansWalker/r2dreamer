"""STORM world model for continuous DMC control."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import OneHotCategorical

from .cores import (
    HyenaSequenceCore,
    MambaSequenceCore,
    S5SequenceCore,
    SlidingWindowSequenceCore,
    TransformerSequenceCore,
)
from .heads import StormRewardHead, StormTerminationHead
from .objectives import CategoricalKLLossWithFreeBits, optimize
from .posterior import StormPosterior
from .prior import StormPrior
from .vision import StormImageDecoder, StormImageEncoder

CORE_TYPES = {
    "transformer": TransformerSequenceCore,
    "mamba": MambaSequenceCore,
    "sliding_window": SlidingWindowSequenceCore,
    "s5": S5SequenceCore,
    "hyena": HyenaSequenceCore,
}
CORE_INIT_OFFSET = 1_000_003


def categorical_sample(logits: torch.Tensor) -> torch.Tensor:
    dist = OneHotCategorical(logits=logits)
    return dist.sample() + dist.probs - dist.probs.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_shapes, action_dim: int, config):
        super().__init__()
        action_dim = int(action_dim)
        posterior = config.posterior
        stoch_dim = int(posterior.stoch_dim)
        hidden_dim = int(config.hidden_dim)
        self.stoch_flattened_dim = stoch_dim * stoch_dim
        self.feat_size = self.stoch_flattened_dim + hidden_dim
        self.use_amp = bool(config.use_amp)
        self.amp_dtype = torch.bfloat16 if str(config.amp_dtype) == "bfloat16" else torch.float16
        self.grad_clip = float(config.grad_clip)

        self.encoder = StormImageEncoder(obs_shapes, config.encoder)
        sequence_stem = nn.Sequential(
            nn.Linear(self.stoch_flattened_dim + action_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
        )
        core_name = str(config.sequence_core)
        recurrent = config.recurrent
        if core_name not in CORE_TYPES:
            raise ValueError(f"Unsupported STORM sequence core: {core_name}")
        # Keep all modules outside the tested sequence core identically
        # initialized across STORM variants with the same experiment seed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((torch.initial_seed() + CORE_INIT_OFFSET) % (2**63 - 1))
            self.sequence_core = CORE_TYPES[core_name](
                stem=sequence_stem,
                feat_dim=hidden_dim,
                config=recurrent,
            )

        self.posterior = StormPosterior(
            encoder_feat_dim=self.encoder.out_dim,
            stoch_dim=stoch_dim,
            unimix_ratio=float(posterior.unimix_ratio),
        )
        prior = config.prior
        self.prior = StormPrior(
            deter_dim=hidden_dim,
            stoch_dim=stoch_dim,
            unimix_ratio=float(prior.unimix_ratio),
        )
        self.decoder = StormImageDecoder(obs_shapes, self.stoch_flattened_dim, config.decoder)
        self.reward = StormRewardHead(hidden_dim, config)
        self.termination = StormTerminationHead(hidden_dim, config)
        self.kl = CategoricalKLLossWithFreeBits(float(config.kl_free))
        self.dyn_scale = float(config.dyn_scale)
        self.rep_scale = float(config.rep_scale)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(config.lr),
            eps=float(config.eps),
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and torch.cuda.is_available(),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def _amp(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp and self.device.type == "cuda",
        )

    def _encode_obs_with_logits(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.encoder(obs)
        post_logits = self.posterior(embed)
        return categorical_sample(post_logits).flatten(-2), post_logits

    def encode_obs(self, obs) -> torch.Tensor:
        sample, _ = self._encode_obs_with_logits(obs)
        return sample

    @property
    def streaming(self):
        return bool(getattr(self.sequence_core, "streaming", False))

    def observe(self, obs, action, cache=None) -> dict[str, torch.Tensor]:
        stoch, post_logits = self._encode_obs_with_logits(obs)
        if cache is None:
            deter = self.sequence_core(stoch, action)
        else:
            outputs = []
            for position in range(stoch.shape[1]):
                output, cache = self.sequence_core.step(
                    stoch[:, position : position + 1],
                    action[:, position : position + 1],
                    cache,
                )
                outputs.append(output)
            deter = torch.cat(outputs, dim=1)
        prior_logits = self.prior(deter)
        feat = torch.cat([stoch, deter], dim=-1)
        return {"stoch": stoch, "deter": deter, "feat": feat, "post_logits": post_logits, "prior_logits": prior_logits}

    @torch.no_grad()
    def replay_cache(self, contexts):
        """Rebuild streaming state at each sampled window boundary."""
        if not self.streaming or not contexts:
            return None

        cache_rows = None
        cache_dtype = self.amp_dtype if self.use_amp and self.device.type == "cuda" else next(self.parameters()).dtype
        # Reconstruct prefixes with the same fixed normalization used for online inference.
        # Batch statistics would make an early cache anchor depend on later sampled frames.
        batch_norms = [module for module in self.encoder.modules() if isinstance(module, nn.BatchNorm2d)]
        batch_norm_training = [norm.training for norm in batch_norms]
        for norm in batch_norms:
            norm.eval()
        try:
            for context, starts in contexts:
                wanted = set(starts)
                cache = self.sequence_core.initial_cache(1, dtype=cache_dtype, device=self.device)
                anchors = {0: tuple(value.detach().clone() for value in cache)} if 0 in wanted else {}
                if max(starts, default=0):
                    context = context.to(self.device, non_blocking=True)
                    with self._amp():
                        stoch = self.encode_obs({self.encoder.key: context[self.encoder.key]})
                        action = context["action"]
                        for position in range(max(starts)):
                            _, cache = self.sequence_core.step(
                                stoch[:, position : position + 1],
                                action[:, position : position + 1],
                                cache,
                            )
                            if position + 1 in wanted:
                                anchors[position + 1] = tuple(value.detach().clone() for value in cache)

                for start in starts:
                    anchor = anchors[start]
                    if cache_rows is None:
                        cache_rows = [[] for _ in anchor]
                    for rows, value in zip(cache_rows, anchor, strict=True):
                        rows.append(value)
        finally:
            for norm, training in zip(batch_norms, batch_norm_training, strict=True):
                norm.train(training)
        return tuple(torch.cat(rows, dim=0) for rows in cache_rows)

    def loss(
        self, obs, action, reward, terminal, cache=None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        with self._amp():
            state = self.observe(obs, action, cache)
            recon = self.decoder(state["stoch"])
            recon_loss = self.decoder.reconstruction_loss(recon, obs)
            reward_logits = self.reward(state["deter"])
            terminal_logits = self.termination(state["deter"])
            reward_loss = self.reward.loss(reward_logits, reward)
            terminal_loss = self.termination.loss(terminal_logits, terminal)
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

    def update(self, obs, action, reward, termination, contexts=None):
        self.train()
        obs = {key: value.to(self.device, non_blocking=True) for key, value in obs.items()}
        action = action.to(self.device, non_blocking=True)
        reward = reward.to(self.device, non_blocking=True)
        termination = termination.to(self.device, non_blocking=True)
        cache = self.replay_cache(contexts)
        loss, metrics, state = self.loss(obs, action, reward, termination, cache)
        optimize(self, self.optimizer, self.scaler, loss, self.grad_clip)
        return metrics, state, (obs, action, reward, termination)

    @torch.no_grad()
    def next_policy_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        prior = categorical_sample(state["prior_logits"][:, :-1]).flatten(-2)
        return torch.cat([prior, state["deter"][:, :-1]], dim=-1)

    @torch.no_grad()
    def context_feature(self, obs, action) -> torch.Tensor:
        with self._amp():
            stoch = self.encode_obs(obs)
            deter = self.sequence_core(stoch, action)[:, -1]
            prior = categorical_sample(self.prior(deter)).flatten(-2)
            return torch.cat([prior, deter], dim=-1)

    @torch.no_grad()
    def context_step(self, obs, action, cache=None):
        with self._amp():
            obs = {key: value.unsqueeze(1) for key, value in obs.items()}
            stoch = self.encode_obs(obs)
            deter, cache = self.sequence_core.step(stoch, action.unsqueeze(1), cache)
            deter = deter[:, 0]
            prior = categorical_sample(self.prior(deter)).flatten(-2)
            return torch.cat([prior, deter], dim=-1), cache

    @torch.no_grad()
    def imagine(self, actor_critic, context_obs, context_action, horizon: int, cache=None) -> dict[str, torch.Tensor]:
        self.eval()
        actor_critic.eval()
        with self._amp():
            stoch = self.encode_obs(context_obs)
            if cache is None:
                cache = self.sequence_core.initial_cache(stoch.shape[0], dtype=stoch.dtype, device=stoch.device)
            deter, cache = self.sequence_core.step(stoch[:, :1], context_action[:, :1], cache)
            for idx in range(1, stoch.shape[1]):
                deter, cache = self.sequence_core.step(
                    stoch[:, idx : idx + 1],
                    context_action[:, idx : idx + 1],
                    cache,
                )
            prior_logits = self.prior(deter[:, 0])
            stoch = categorical_sample(prior_logits).flatten(-2)
            deter = deter[:, 0]
            feats = [torch.cat([stoch, deter], dim=-1)]
            actions, rewards, terminals = [], [], []
            for _ in range(int(horizon)):
                action, _ = actor_critic.sample(feats[-1], deterministic=False)
                next_deter, cache = self.sequence_core.step(stoch.unsqueeze(1), action.unsqueeze(1), cache)
                next_deter = next_deter[:, 0]
                reward = self.reward.decode(self.reward(next_deter))
                termination_logit = self.termination(next_deter)
                terminal = (self.termination.probability(termination_logit) > 0.5).to(reward.dtype)
                prior_logits = self.prior(next_deter)
                stoch = categorical_sample(prior_logits).flatten(-2)
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
