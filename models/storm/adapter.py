"""Dreamer-facing adapter for the STORM world model."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Independent, OneHotCategorical

from constants import storm_cache_keys
from .world_model import StormWorldModel, get_subsequent_mask_with_batch_length


class StormDreamerAdapter(nn.Module):
    """Expose STORM dynamics through a Dreamer RSSM-like interface."""

    uses_context = True

    def __init__(self, world_model: StormWorldModel):
        super().__init__()
        self.world_model = world_model
        self.flat_stoch = int(world_model.stoch_flattened_dim)
        self._deter = int(world_model.transformer_hidden_dim)
        self.feat_size = self.flat_stoch + self._deter
        self.cache_keys = storm_cache_keys(len(world_model.storm_transformer.layer_stack))
        self.store_cache_in_replay = False
        self.returns_sequence_cache = False

    @property
    def device(self):
        return next(self.parameters()).device

    def initial(self, batch_size: int):
        device = self.device
        stoch = torch.zeros(batch_size, self.flat_stoch, dtype=torch.float32, device=device)
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=device)
        return stoch, deter

    def initial_context(self, batch_size: int, dtype=None):
        dtype = dtype or torch.float32
        return self.world_model.storm_transformer.initial_cache(batch_size, dtype=dtype, device=self.device)

    def _unpack_state(self, state):
        if len(state) < 2:
            raise ValueError(f"Expected STORM state with at least 2 tensors, got {len(state)}.")
        stoch, deter = state[:2]
        cache = tuple(state[2:]) if len(state) > 2 else None
        return stoch, deter, cache

    def _reset_state(self, stoch, deter, cache, reset):
        reset = reset.reshape(reset.shape[0], -1).any(dim=-1)
        state_reset = reset.reshape(reset.shape[0], *([1] * (stoch.dim() - 1)))
        stoch = torch.where(state_reset, torch.zeros_like(stoch), stoch)
        deter_reset = reset.reshape(reset.shape[0], *([1] * (deter.dim() - 1)))
        deter = torch.where(deter_reset, torch.zeros_like(deter), deter)
        if cache is not None:
            cache = self.world_model.storm_transformer.reset_cache(cache, reset)
        return stoch, deter, cache

    def _post_from_embed(self, embed, sample_mode: str = "random_sample"):
        post_logit = self.world_model.dist_head.forward_post(embed)
        post_sample = self.world_model.straight_through_gradient(post_logit, sample_mode=sample_mode)
        return self.world_model.flatten_sample(post_sample), post_logit

    def _can_vector_observe(self, cache, reset, return_cache: bool) -> bool:
        if return_cache:
            return False
        position, *layer_cache = cache
        if bool(torch.any(position != 0)):
            return False
        if any(tensor.shape[1] != 0 for tensor in layer_cache):
            return False
        if reset.shape[1] <= 1:
            return True
        return not bool(reset[:, 1:].reshape(reset.shape[0], -1).any())

    def _observe_vectorized(self, embed, action, initial_stoch, reset):
        stoch, logit = self._post_from_embed(embed)
        initial_stoch = initial_stoch.to(device=stoch.device, dtype=stoch.dtype)
        prev_stoch = torch.cat([initial_stoch.unsqueeze(1), stoch[:, :-1]], dim=1)
        reset_mask = reset.reshape(reset.shape[0], reset.shape[1], -1).any(dim=-1)
        prev_stoch = torch.where(reset_mask.unsqueeze(-1), torch.zeros_like(prev_stoch), prev_stoch)
        action = torch.where(reset_mask.unsqueeze(-1), torch.zeros_like(action), action)
        mask = get_subsequent_mask_with_batch_length(stoch.shape[1], stoch.device)
        deter = self.world_model.storm_transformer(prev_stoch, action, mask)
        return stoch, deter, logit

    def observe(self, embed, action, initial, reset, *, return_cache: bool = False):
        """Teacher-force STORM over encoder embeddings and previous actions."""
        stoch, deter, cache = self._unpack_state(tuple(initial))
        if cache is None:
            cache = self.initial_context(stoch.shape[0], dtype=stoch.dtype)
        if self._can_vector_observe(cache, reset, return_cache):
            return self._observe_vectorized(embed, action, stoch, reset)
        stochs, deters, logits = [], [], []
        for i in range(action.shape[1]):
            stoch, deter, logit, *cache = self.obs_step(
                stoch,
                deter,
                action[:, i],
                embed[:, i],
                reset[:, i],
                *cache,
            )
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
            cache = tuple(cache)
        out = (torch.stack(stochs, dim=1), torch.stack(deters, dim=1), torch.stack(logits, dim=1))
        if return_cache:
            # STORM caches grow with time, so they cannot be returned as a simple
            # B x T tensor stack like Mamba's fixed-size recurrent state.
            return out + tuple(cache)
        return out

    def obs_step(self, prev_stoch, prev_deter, prev_action, embed, reset, *cache):
        cache = tuple(cache) if cache else None
        prev_stoch, prev_deter, cache = self._reset_state(prev_stoch, prev_deter, cache, reset)
        reset_mask = reset.reshape(reset.shape[0], *([1] * (prev_action.dim() - 1))).to(dtype=torch.bool)
        prev_action = torch.where(reset_mask, torch.zeros_like(prev_action), prev_action)
        if embed.dim() == 3:
            if embed.shape[1] != 1:
                raise ValueError(f"STORM obs_step expects a single embedding step, got shape {tuple(embed.shape)}.")
            embed = embed[:, 0]
        if prev_action.dim() == 2:
            prev_action = prev_action.unsqueeze(1)

        # Match the Dreamer replay convention:
        # previous latent + previous action -> current deterministic feature,
        # then current observation embedding -> current posterior stochastic state.
        prev_stoch_seq = prev_stoch.unsqueeze(1) if prev_stoch.dim() == 2 else prev_stoch
        deter, cache = self.world_model.storm_transformer.forward_step_with_cache(
            prev_stoch_seq,
            prev_action,
            cache,
        )
        stoch, logit = self._post_from_embed(embed)
        return stoch, deter[:, 0], logit, *cache

    def prior(self, deter):
        logit = self.world_model.dist_head.forward_prior(deter)
        sample = self.world_model.straight_through_gradient(logit, sample_mode="random_sample")
        return self.world_model.flatten_sample(sample), logit

    def img_step(self, stoch, deter, prev_action, *cache):
        cache = tuple(cache) if cache else None
        if stoch.dim() == 2:
            stoch = stoch.unsqueeze(1)
        if prev_action.dim() == 2:
            prev_action = prev_action.unsqueeze(1)
        deter, cache = self.world_model.storm_transformer.forward_step_with_cache(stoch, prev_action, cache)
        stoch, _ = self.prior(deter)
        return stoch[:, 0], deter[:, 0], *cache

    def imagine_with_action(self, stoch, deter, actions, *cache):
        cache = tuple(cache) if cache else None
        stochs, deters = [], []
        for i in range(actions.shape[1]):
            stoch, deter, *cache = self.img_step(stoch, deter, actions[:, i], *cache)
            cache = tuple(cache)
            stochs.append(stoch)
            deters.append(deter)
        return torch.stack(stochs, dim=1), torch.stack(deters, dim=1), *cache

    def get_feat(self, stoch, deter):
        return torch.cat([stoch.reshape(*stoch.shape[:-1], self.flat_stoch), deter], dim=-1)

    def get_dist(self, logit):
        return Independent(OneHotCategorical(logits=logit), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        post = OneHotCategorical(logits=post_logit)
        prior = OneHotCategorical(logits=prior_logit)
        rep = torch.distributions.kl.kl_divergence(post, OneHotCategorical(logits=prior_logit.detach())).sum(-1)
        dyn = torch.distributions.kl.kl_divergence(OneHotCategorical(logits=post_logit.detach()), prior).sum(-1)
        return torch.clip(dyn, min=free), torch.clip(rep, min=free)


__all__ = ["StormDreamerAdapter"]
