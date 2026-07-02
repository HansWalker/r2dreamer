import copy
import math
from collections import OrderedDict
from contextlib import contextmanager

import torch
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

import tools
from models.dreamer import networks, rssm
from models.storm import StormDreamerAdapter, StormWorldModel
from optim import LaProp, clip_grad_agc_
from tools import to_f32


def _cfg_get(config, name, default=None):
    try:
        return config[name]
    except Exception:
        return default


@contextmanager
def _freeze_module_params(*modules):
    params = [(param, param.requires_grad) for module in modules for param in module.parameters()]
    for param, _ in params:
        param.requires_grad_(False)
    try:
        yield
    finally:
        for param, requires_grad in params:
            param.requires_grad_(requires_grad)


class Dreamer(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else math.prod(act_space.shape)

        # World model components
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.world_model = str(_cfg_get(config, "world_model", "dreamer"))
        if self.world_model == "storm":
            storm_cfg = _cfg_get(config, "storm")
            storm_model = StormWorldModel(
                shapes,
                self.act_dim,
                transformer_max_length=_cfg_get(storm_cfg, "transformer_max_length"),
                batch_length=_cfg_get(storm_cfg, "batch_length"),
                transformer_hidden_dim=int(_cfg_get(storm_cfg, "transformer_hidden_dim", 512)),
                transformer_num_layers=int(_cfg_get(storm_cfg, "transformer_num_layers", 2)),
                transformer_num_heads=int(_cfg_get(storm_cfg, "transformer_num_heads", 8)),
                stoch_dim=int(_cfg_get(storm_cfg, "stoch_dim", 32)),
                encoder_hidden_dim=int(_cfg_get(storm_cfg, "encoder_hidden_dim", 512)),
                encoder_layers=int(_cfg_get(storm_cfg, "encoder_layers", 3)),
                dropout=float(_cfg_get(storm_cfg, "dropout", 0.1)),
            )
            self.encoder = storm_model.encoder
            self.embed_size = self.encoder.out_dim
            self.rssm = StormDreamerAdapter(
                storm_model,
                context_length=int(_cfg_get(storm_cfg, "context_length", 16)),
            )
        else:
            self.encoder = networks.MultiEncoder(config.encoder, shapes)
            self.embed_size = self.encoder.out_dim
            self.rssm = rssm.RSSM(
                config.rssm,
                self.embed_size,
                self.act_dim,
            )
        self.cache_keys = tuple(getattr(self.rssm, "cache_keys", ()))
        self.returns_sequence_cache = bool(getattr(self.rssm, "returns_sequence_cache", True))
        self.reward = networks.MLPHead(config.reward, self.rssm.feat_size)
        self.cont = networks.MLPHead(config.cont, self.rssm.feat_size)

        config.actor.shape = (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
        elif hasattr(act_space, "discrete"):
            config.actor.dist = config.actor.dist.disc
        else:
            config.actor.dist = config.actor.dist.cont

        # Actor-critic components
        self.actor = networks.MLPHead(config.actor, self.rssm.feat_size)
        self.value = networks.MLPHead(config.critic, self.rssm.feat_size)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0

        self._loss_scales = dict(config.loss_scales)
        self._log_grads = bool(config.log_grads)

        modules = {
            "rssm": self.rssm,
            "actor": self.actor,
            "value": self.value,
            "reward": self.reward,
            "cont": self.cont,
        }
        if self.world_model != "storm":
            modules["encoder"] = self.encoder

        self.decoder = networks.MultiDecoder(
            config.decoder,
            self.rssm._deter,
            self.rssm.flat_stoch,
            shapes,
        )
        recon = self._loss_scales.pop("recon")
        self._loss_scales.update({k: recon for k in self.decoder.all_keys})
        modules.update({"decoder": self.decoder})
        # count number of parameters in each module
        for key, module in modules.items():
            if isinstance(module, nn.Parameter):
                count = module.numel() if module.requires_grad else 0
            else:
                count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{count:>14,}: {key}")
        self._named_params = OrderedDict()
        for name, module in modules.items():
            if isinstance(module, nn.Parameter):
                if module.requires_grad:
                    self._named_params[name] = module
            else:
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        self._named_params[f"{name}.{param_name}"] = param
        print(f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters.")

        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

        self._agc = _agc
        self._optimizer = LaProp(
            self._named_params.values(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        self._scaler = GradScaler()

        def lr_lambda(step):
            if config.warmup:
                return min(1.0, (step + 1) / config.warmup)
            return 1.0

        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        self.train()
        if config.compile:
            print("Compiling update function with torch.compile...")
            self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")

    def _update_slow_target(self):
        """Update slow-moving value target network."""
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def _optimizer_step(self, metrics):
        """Apply one optimizer step and append optimizer metrics."""
        self._scaler.unscale_(self._optimizer)
        if self._log_grads:
            old_params = [p.data.clone().detach() for p in self._named_params.values()]
            grads = [p.grad for p in self._named_params.values() if p.grad is not None]
            metrics["opt/grad_norm"] = tools.compute_global_norm(grads)
            metrics["opt/grad_rms"] = tools.compute_rms(grads)
        self._agc(self._named_params.values())
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)
        metrics["opt/lr"] = self._scheduler.get_lr()[0]
        metrics["opt/grad_scale"] = self._scaler.get_scale()
        if self._log_grads:
            updates = [(new - old) for (new, old) in zip(self._named_params.values(), old_params)]
            metrics["opt/param_rms"] = tools.compute_rms(self._named_params.values())
            metrics["opt/update_rms"] = tools.compute_rms(updates)
        return metrics

    def train(self, mode=True):
        super().train(mode)
        # slow_value should be always eval mode
        self._slow_value.train(False)
        return self

    def _state_tuple(self, state):
        if isinstance(state, (tuple, list)):
            return tuple(state)
        out = [state["stoch"], state["deter"]]
        if all(key in state.keys() for key in self.cache_keys):
            out.extend(state[key] for key in self.cache_keys)
        return tuple(out)

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        """Policy inference step."""
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        embed = self.encoder(p_obs)
        feat, state_update, random_mask = self.rssm.actor_step(embed, state, obs["is_first"])
        action_dist = self.actor(feat)
        action = action_dist.mode if eval else action_dist.rsample()
        if random_mask is not None:
            random_action = torch.empty_like(action).uniform_(-1.0, 1.0)
            action = torch.where(random_mask.reshape(-1, 1), random_action, action)
        next_state = self.rssm.actor_state_after_action(state_update, action)
        return action, TensorDict(next_state, batch_size=state.batch_size)

    @torch.no_grad()
    def get_initial_state(self, B):
        return TensorDict(self.rssm.initial_actor_state(B, self.act_dim), batch_size=(B,))

    def update(self, replay_buffer):
        """Sample a batch from replay and perform one optimization step."""
        warmup_data, data, index, initial = replay_buffer.sample()
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        initial = self._replay_initial(initial, warmup_data)
        self._update_slow_target()
        metrics = {}
        with autocast(device_type=self.device.type, dtype=torch.float16):
            post_state, mets = self._cal_grad(p_data, initial)
        metrics.update(self._optimizer_step(mets))
        # update latent vectors in replay buffer
        replay_buffer.update(index, *(value.detach() for value in post_state), cache_keys=self.cache_keys)
        return metrics

    def update_expert_pretrain(self, data):
        """Perform one supervised expert pretraining update from full episodes."""
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        initial = self._initial_tuple(data.shape[0])
        self._update_slow_target()
        metrics = {}
        with autocast(device_type=self.device.type, dtype=torch.float16):
            mets = self._cal_expert_pretrain_grad(p_data, initial)
        metrics.update(self._optimizer_step(mets))
        return metrics

    @torch.no_grad()
    def _replay_initial(self, initial, warmup_data=None):
        initial = self._state_tuple(initial)
        if warmup_data is None or warmup_data.shape[1] == 0:
            return initial
        p_warmup = self.preprocess(warmup_data)
        embed = self.encoder(p_warmup)
        post = self.rssm.observe(embed, p_warmup["action"], initial, p_warmup["is_first"], return_cache=True)
        return self._final_state_from_post(post)

    def _final_state_from_post(self, post):
        stoch, deter = post[:2]
        out = [stoch[:, -1].detach(), deter[:, -1].detach()]
        if len(post) > 3:
            if self.returns_sequence_cache:
                out.extend(value[:, -1].detach().clone() for value in post[3:])
            else:
                out.extend(value.detach().clone() for value in post[3:])
        return tuple(out)

    def _initial_tuple(self, batch_size):
        stoch, deter = self.rssm.initial(batch_size)
        initial = [stoch, deter]
        cache = self.rssm.initial_context(batch_size)
        if cache is not None:
            initial.extend(cache)
        return tuple(initial)

    def _cal_grad(self, data, initial):
        """Compute gradients for one batch.

        Notes
        -----
        This function computes:
        1) World model loss (dynamics + representation)
        2) Observation reconstruction
        3) Imagination rollouts for actor-critic updates
        """
        # data: dict of (B, T, *), initial: (stoch, deter, optional Mamba3 cache tensors)
        losses = {}
        metrics = {}

        # === World model: posterior rollout and KL losses ===
        # (B, T, E)
        embed = self.encoder(data)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        post = self.rssm.observe(embed, data["action"], initial, data["is_first"])
        post_stoch, post_deter, post_logit = post[:3]
        post_cache = post[3:] if len(post) > 3 else None
        # (B, T, S, K)
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)
        # (B, T, F)
        feat = self.rssm.get_feat(post_stoch, post_deter)
        recon_losses = {
            key: torch.mean(-dist.log_prob(data[key])) for key, dist in self.decoder(post_stoch, post_deter).items()
        }
        losses.update(recon_losses)

        # reward and continue
        losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["reward"])))
        cont = 1.0 - to_f32(data["is_terminal"])
        losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))
        # log
        metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())

        # === Imagination rollout for actor-critic ===
        # (B*T, S, K), (B*T, D)
        start = self.rssm.imagination_start(post_stoch, post_deter, data, post_cache=post_cache)
        with _freeze_module_params(self.rssm, self.reward, self.cont, self.value, self._slow_value):
            imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1)

            # (B*T, T_imag, 1)
            imag_reward = self.reward(imag_feat).mode()
            # (B*T, T_imag, 1)  probability of continuation
            imag_cont = self.cont(imag_feat).mean
            # (B*T, T_imag, 1)
            imag_value = self.value(imag_feat).mode()
            imag_slow_value = self._slow_value(imag_feat).mode()
            disc = 1 - 1 / self.horizon
            # (B*T, T_imag, 1)
            weight = torch.cumprod(imag_cont * disc, dim=1)
            last = torch.zeros_like(imag_cont)
            term = 1 - imag_cont
            ret = self._lambda_return(
                last, term, imag_reward, imag_value, imag_value, disc, self.lamb
            )  # (B*T, T_imag-1, 1)
            ret_offset, ret_scale = self.return_ema(ret)
            # (B*T, T_imag-1, 1)
            adv = (ret - imag_value[:, :-1]) / ret_scale

            policy = self.actor(imag_feat.detach())
            entropy = policy.entropy()[:, :-1].unsqueeze(-1)
            actor_target = (ret - ret_offset) / ret_scale
            losses["policy"] = torch.mean(weight[:, :-1].detach() * -(actor_target + self.act_entropy * entropy))

        imag_value_dist = self.value(imag_feat.detach())
        # (B*T, T_imag, 1)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        losses["value"] = torch.mean(
            weight[:, :-1].detach()
            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[
                :, :-1
            ].unsqueeze(-1)
        )
        # log
        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = torch.mean(ret_normed)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_action, "action"))

        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        if post_cache:
            return (post_stoch, post_deter, *post_cache), metrics
        return (post_stoch, post_deter), metrics

    def _cal_expert_pretrain_grad(self, data, initial):
        """Compute expert pretraining gradients without imagined policy rollouts."""
        losses = {}
        metrics = {}

        embed = self.encoder(data)
        post = self.rssm.observe(embed, data["action"], initial, data["is_first"], return_cache=False)
        post_stoch, post_deter, post_logit = post[:3]
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)

        feat = self.rssm.get_feat(post_stoch, post_deter)
        recon_losses = {
            key: torch.mean(-dist.log_prob(data[key])) for key, dist in self.decoder(post_stoch, post_deter).items()
        }
        losses.update(recon_losses)

        losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["reward"])))
        cont = 1.0 - to_f32(data["is_terminal"])
        losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))
        bc_dist = self.actor(feat.detach())
        losses["bc"] = torch.mean(-bc_dist.log_prob(to_f32(data["action"])))

        last, term, reward = (
            to_f32(data["is_last"]),
            to_f32(data["is_terminal"]),
            to_f32(data["reward"]),
        )
        with torch.no_grad():
            value = self.value(feat).mode()
            slow_value = self._slow_value(feat).mode()
        disc = 1 - 1 / self.horizon
        ret = self._lambda_return(last, term, reward, value, value, disc, self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        weight = 1.0 - last
        value_dist = self.value(feat)
        losses["repval"] = torch.mean(
            weight[:, :-1]
            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(
                -1
            )
        )

        metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())
        metrics["bc_logprob"] = -losses["bc"]
        metrics.update(tools.tensorstats(ret, "ret_replay"))
        metrics.update(tools.tensorstats(value, "value_replay"))
        metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))

        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()
        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return metrics

    def _imagine(self, start, imag_horizon):
        """Roll out the policy in latent space."""
        # (B, S, K), (B, D), optional recurrent/context tensors
        feats = []
        actions = []
        if len(start) >= 2:
            stoch, deter, *cache = start
            cache = tuple(value.clone() for value in cache)
        else:
            raise ValueError(f"Expected imagination start with at least 2 tensors, got {len(start)}.")
        for _ in range(imag_horizon):
            # (B, F)
            feat = self.rssm.get_feat(stoch, deter)
            # (B, A)
            action = self.actor(feat).rsample()
            # Append feat and its corresponding sampled action at the same time step.
            feats.append(feat)
            actions.append(action)
            stoch, deter, cache = self.rssm.imagine_step(stoch, deter, action, cache)

        # Stack along sequence dim T_imag.
        # (B, T_imag, F), (B, T_imag, A)
        return torch.stack(feats, dim=1), torch.stack(actions, dim=1)

    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        """
        lamb=1 means discounted Monte Carlo return.
        lamb=0 means fixed 1-step return.
        """
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    @torch.no_grad()
    def preprocess(self, data):
        if "image" in data:
            data["image"] = to_f32(data["image"]) / 255.0
        return data
