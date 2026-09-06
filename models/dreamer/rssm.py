from contextlib import contextmanager

import torch
from torch import distributions as torchd
from torch import nn

from models.shared import distributions as dists
from models.shared.utils import rpad, weight_init_

from .cores import (
    DreamerGRUCore,
    DreamerHyenaCore,
    DreamerMambaCore,
    DreamerS5Core,
    DreamerSlidingWindowCore,
)
from .posterior import DreamerPosterior
from .prior import DreamerPrior

CORE_TYPES = {
    "block_gru": (DreamerGRUCore, "gru"),
    "mamba3": (DreamerMambaCore, "mamba3"),
    "s5": (DreamerS5Core, "s5"),
    "hyena": (DreamerHyenaCore, "hyena"),
    "sliding_window": (DreamerSlidingWindowCore, "sliding_window"),
}
CORE_INIT_OFFSET = 1_000_003


class RSSM(nn.Module):
    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        posterior = config.posterior
        self._stoch = int(posterior.stoch_dim)
        self._deter = int(config.deter)
        self._discrete = int(posterior.discrete_dim)
        self._unimix_ratio = float(posterior.unimix_ratio)
        self._device = torch.device(config.device)
        self._core = str(config.core)
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter
        if self._core not in CORE_TYPES:
            raise ValueError(f"Unsupported RSSM core: {self._core}")
        core_type, settings = CORE_TYPES[self._core]
        # Core variants must not perturb the initialization of the shared
        # posterior, prior, heads, or policy modules built after this RSSM.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((torch.initial_seed() + CORE_INIT_OFFSET) % (2**63 - 1))
            self._deter_net = core_type(
                self._deter,
                self.flat_stoch,
                act_dim,
                getattr(config, settings),
            )
            if self._core == "block_gru":
                self._deter_net.apply(weight_init_)

        self._obs_net = DreamerPosterior(
            config,
            deter_dim=self._deter,
            embed_dim=embed_size,
        )
        self._prior_net = DreamerPrior(
            config,
            deter_dim=self._deter,
        )
        self._obs_net.apply(weight_init_)
        self._prior_net.apply(weight_init_)

    @property
    def uses_context(self):
        return bool(self.cache_keys)

    @property
    def cache_keys(self):
        return tuple(getattr(self._deter_net, "cache_keys", ()))

    @contextmanager
    def sequence_context(self, reference):
        prepare = getattr(self._deter_net, "prepare_sequence", None)
        if prepare is not None:
            prepare(reference)
        try:
            yield
        finally:
            clear = getattr(self._deter_net, "clear_sequence", None)
            if clear is not None:
                clear()

    def initial_context(self, batch_size, dtype=None):
        if not self.uses_context:
            return None
        return self._deter_net.initial_context(batch_size, device=self._device, dtype=dtype)

    def _unpack_state(self, state):
        if len(state) == 2:
            stoch, deter = state
            return stoch, deter, None
        if len(state) == 2 + len(self.cache_keys):
            return state[0], state[1], tuple(state[2:])
        raise ValueError(f"Expected RSSM state with 2 or {2 + len(self.cache_keys)} tensors, got {len(state)}.")

    def _ensure_cache(self, cache, batch_size, device):
        if not self.uses_context:
            return None
        if not cache or any(tensor is None for tensor in cache):
            return self._deter_net.initial_context(batch_size, device=device)
        return tuple(tensor.to(device=device) for tensor in cache)

    def _reset_cache(self, cache, reset):
        if cache is None:
            return None
        reset = reset.reshape(reset.shape[0])
        return tuple(
            torch.where(
                rpad(reset, tensor.dim() - int(reset.dim())),
                torch.zeros_like(tensor),
                tensor,
            )
            for tensor in cache
        )

    def _stack_cache(self, caches):
        return tuple(torch.stack(items, dim=1) for items in zip(*caches, strict=True))

    def initial(self, batch_size):
        """Return an initial latent state."""
        # (B, D), (B, S, K)
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        return stoch, deter

    def initial_actor_state(self, batch_size, act_dim):
        stoch, deter = self.initial(batch_size)
        state = {
            "stoch": stoch,
            "deter": deter,
            "prev_action": torch.zeros(batch_size, act_dim, dtype=torch.float32, device=self._device),
        }
        cache = self.initial_context(batch_size)
        if cache is not None:
            state.update({key: value for key, value in zip(self.cache_keys, cache, strict=True)})
        return state

    def actor_step(self, embed, state, reset):
        cache = tuple(state[key] for key in self.cache_keys) if self.cache_keys else ()
        stoch, deter, _, *cache = self.obs_step(
            state["stoch"],
            state["deter"],
            state["prev_action"],
            embed,
            reset,
            *cache,
        )
        next_state = {key: value for key, value in zip(self.cache_keys, cache, strict=True)}
        next_state.update({"stoch": stoch, "deter": deter})
        return self.get_feat(stoch, deter), next_state

    def actor_state_after_action(self, state_update, action):
        state_update["prev_action"] = action
        return state_update

    def observe(self, embed, action, initial, reset, return_cache=True, cache_start=0):
        """Posterior rollout using observations."""
        # (B, T, E), (B, T, A), ((B, S, K), (B, D), optional cache tensors), (B, T)
        L = action.shape[1]
        stoch, deter, cache = self._unpack_state(initial)
        cache = self._ensure_cache(cache, stoch.shape[0], deter.device) if self.uses_context else ()
        stochs, deters, logits = [], [], []
        caches = [] if self.uses_context and return_cache else None
        with self.sequence_context(embed):
            for i in range(L):
                # (B, S, K), (B, D), (B, S, K)
                stoch, deter, logit, *cache = self.obs_step(
                    stoch,
                    deter,
                    action[:, i],
                    embed[:, i],
                    reset[:, i],
                    *cache,
                )
                cache = tuple(cache)
                stochs.append(stoch)
                deters.append(deter)
                logits.append(logit)
                if caches is not None and i >= cache_start:
                    caches.append(tuple(tensor.detach().clone() for tensor in cache))
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        logits = torch.stack(logits, dim=1)
        if caches is not None:
            return stochs, deters, logits, *self._stack_cache(caches)
        return stochs, deters, logits

    def _deter_step(self, stoch, deter, action, cache):
        if self.uses_context:
            deter, *cache = self._deter_net(stoch, action, *cache)
            return deter, tuple(cache)
        return self._deter_net(stoch, deter, action), ()

    def obs_step(
        self,
        stoch,
        deter,
        prev_action,
        embed,
        reset,
        *cache,
    ):
        """Single posterior step."""
        # (B, S, K), (B, D), (B, A), (B, E), (B,), optional recurrent cache tensors
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )
        if self.uses_context:
            cache = self._ensure_cache(cache, stoch.shape[0], deter.device)
            cache = self._reset_cache(cache, reset)

        # Deterministic transition then posterior logits conditioned on embed.
        deter, cache = self._deter_step(stoch, deter, prev_action, cache if self.uses_context else ())
        # (B, S, K)
        logit = self._obs_net(deter, embed)

        # Sample discrete stochastic state via straight-through Gumbel-Softmax.
        # (B, S, K)
        stoch = self.get_dist(logit).rsample()
        return stoch, deter, logit, *cache

    def img_step(
        self,
        stoch,
        deter,
        prev_action,
        *cache,
    ):
        """Single prior step (no observation)."""

        if self.uses_context:
            cache = self._ensure_cache(cache, stoch.shape[0], deter.device)
        else:
            cache = ()
        deter, cache = self._deter_step(stoch, deter, prev_action, cache)
        # (B, S, K)
        stoch, _ = self.prior(deter)
        return stoch, deter, *cache

    def prior(self, deter):
        """Compute prior distribution parameters and sample stoch."""

        # (B, S, K)
        logit = self._prior_net(deter)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def imagination_start(self, post_stoch, post_deter, post_cache=None, sample_size=None):
        total = post_stoch.shape[0] * post_stoch.shape[1]
        indices = None
        if sample_size is not None and int(sample_size) < total:
            indices = torch.randperm(total, device=post_stoch.device)[: int(sample_size)]

        def flatten(value):
            value = value.reshape(total, *value.shape[2:]).detach()
            return value if indices is None else value[indices]

        start = (flatten(post_stoch), flatten(post_deter))
        if post_cache:
            return start + tuple(flatten(value) for value in post_cache)
        return start

    def imagine_step(self, stoch, deter, action, cache):
        stoch, deter, *cache = self.img_step(stoch, deter, action, *cache)
        return stoch, deter, tuple(cache)

    def get_feat(self, stoch, deter):
        """Flatten stoch and concatenate with deter."""
        # (B, S, K), (B, D)
        # (B, S*K)
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        # (B, S*K + D)
        return torch.cat([stoch, deter], -1)

    def get_dist(self, logit):
        return torchd.independent.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        kld = dists.kl
        rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
        # Clipped gradients are not backpropagated using torch.clip.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)

        return dyn_loss, rep_loss
