import copy
from collections import OrderedDict
from functools import partial

import torch
from tensordict import TensorDict
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

from models.shared.utils import to_f32
from optim import LaProp, clip_grad_agc_

from .model import DreamerModel
from .networks import ReturnEMA


class Dreamer(DreamerModel):
    def __init__(self, config, model_io):
        super().__init__(config, model_io)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.imag_batch_size = int(config.imag_batch_size)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = ReturnEMA(device=self.device)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0

        self._loss_scales = dict(config.loss_scales)
        recon = self._loss_scales.pop("recon")
        self._loss_scales.update({k: recon for k in self.decoder.all_keys})
        module_names = ("rssm", "encoder", "actor", "value", "reward", "cont", "decoder")
        self._named_params = OrderedDict(
            (f"{name}.{param_name}", param)
            for name in module_names
            for param_name, param in getattr(self, name).named_parameters()
            if param.requires_grad
        )
        self._agc = partial(
            clip_grad_agc_,
            clip=float(config.agc),
            pmin=float(config.pmin),
            foreach=True,
        )
        self._optimizer = LaProp(
            self._named_params.values(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        self._scaler = GradScaler("cuda", enabled=self.device.type == "cuda")

        warmup = int(config.warmup)
        self._scheduler = LambdaLR(
            self._optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup) if warmup else 1.0,
        )

        self.train()
        if config.compile:
            print("Model | compiling=torch.compile")
            self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")

    def training_state_dict(self):
        return {
            "scheduler": self._scheduler.state_dict(),
            "scaler": self._scaler.state_dict(),
            "slow_value_updates": self._slow_value_updates,
        }

    def load_training_state_dict(self, state):
        if not state:
            return
        self._scheduler.load_state_dict(state["scheduler"])
        self._scaler.load_state_dict(state["scaler"])
        self._slow_value_updates = int(state["slow_value_updates"])

    def _update_slow_target(self):
        """Update slow-moving value target network."""
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters(), strict=True):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def _optimizer_step(self, metrics):
        """Apply one optimizer step and append optimizer metrics."""
        self._scaler.unscale_(self._optimizer)
        self._agc(self._named_params.values())
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)
        metrics["opt/lr"] = self._scheduler.get_last_lr()[0]
        metrics["opt/grad_scale"] = self._scaler.get_scale()
        return metrics

    def train(self, mode=True):
        super().train(mode)
        # slow_value should be always eval mode
        self._slow_value.train(False)
        return self

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        """Policy inference step."""
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        embed = self.encoder(p_obs)
        feat, state_update = self.rssm.actor_step(embed, state, obs["is_first"])
        action_dist = self.actor(feat)
        action = (action_dist.mode if eval else action_dist.rsample()).clamp(-1.0, 1.0)
        next_state = self.rssm.actor_state_after_action(state_update, action)
        return action, TensorDict(next_state, batch_size=state.batch_size)

    @torch.no_grad()
    def get_initial_state(self, B):
        return TensorDict(self.rssm.initial_actor_state(B, self.act_dim), batch_size=(B,))

    def update(self, replay_buffer):
        """Sample a batch from replay and perform one optimization step."""
        contexts, data = replay_buffer.sample()
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        initial = self._replay_initial(contexts)
        self._update_slow_target()
        with autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            metrics = self._cal_grad(p_data, initial)
        self._optimizer_step(metrics)
        return metrics

    def update_expert_pretrain(self, data, contexts=None):
        """Perform one supervised expert update from a reconstructed replay state."""
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        initial = self._replay_initial(contexts) if contexts is not None else self._initial_tuple(data.shape[0])
        self._update_slow_target()
        with autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            metrics = self._cal_expert_pretrain_grad(p_data, initial)
        return self._optimizer_step(metrics)

    @torch.no_grad()
    def _replay_initial(self, contexts):
        max_start = max((max(starts, default=0) for _, starts in contexts), default=0)
        state = self._initial_tuple(len(contexts))
        anchors = [[] for _ in contexts]
        capture_at = {}
        for group, (_, starts) in enumerate(contexts):
            for start in starts:
                capture_at.setdefault(start, []).append(group)

        def capture(position):
            for group in capture_at.get(position, ()):
                anchors[group].append(tuple(value[group : group + 1].detach().clone() for value in state))

        capture(0)
        if max_start:
            context_batch = TensorDict(
                {
                    key: torch.nn.utils.rnn.pad_sequence(
                        [context[key][0] for context, _ in contexts],
                        batch_first=True,
                    )
                    for key in contexts[0][0].keys()
                },
                batch_size=(len(contexts), max_start),
            )
            embed = self.encoder(self.preprocess(context_batch))
            action = context_batch["action"]
            is_first = context_batch["is_first"]

            with self.rssm.sequence_context(embed):
                for position in range(max_start):
                    stoch, deter, *cache = state
                    stoch, deter, _, *cache = self.rssm.obs_step(
                        stoch,
                        deter,
                        action[:, position],
                        embed[:, position],
                        is_first[:, position],
                        *cache,
                    )
                    state = (stoch, deter, *cache)
                    capture(position + 1)

        ordered = [anchor for group in anchors for anchor in group]
        return tuple(torch.cat(values, dim=0) for values in zip(*ordered, strict=True))

    def _initial_tuple(self, batch_size):
        stoch, deter = self.rssm.initial(batch_size)
        cache = self.rssm.initial_context(batch_size)
        return (stoch, deter, *(cache or ()))

    def _world_model_loss(self, data, initial, *, return_cache, cache_start=0):
        embed = self.encoder(data)
        post = self.rssm.observe(
            embed,
            data["action"],
            initial,
            data["is_first"],
            return_cache=return_cache,
            cache_start=cache_start,
        )
        post_stoch, post_deter, post_logit = post[:3]
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses = {
            "dyn": torch.mean(dyn_loss),
            "rep": torch.mean(rep_loss),
        }
        losses.update(
            {key: torch.mean(-dist.log_prob(data[key])) for key, dist in self.decoder(post_stoch, post_deter).items()}
        )

        feat = self.rssm.get_feat(post_stoch, post_deter)
        losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["reward"])))
        losses["con"] = torch.mean(-self.cont(feat).log_prob(1.0 - to_f32(data["is_terminal"])))
        metrics = {
            "dyn_entropy": torch.mean(self.rssm.get_dist(prior_logit).entropy()),
            "rep_entropy": torch.mean(self.rssm.get_dist(post_logit).entropy()),
        }
        return losses, metrics, feat, post

    def _backward_losses(self, losses, metrics):
        total_loss = sum(loss * self._loss_scales[name] for name, loss in losses.items())
        self._scaler.scale(total_loss).backward()
        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics["opt/loss"] = total_loss
        return metrics

    def _value_loss(self, feature, target, slow_target, weight):
        value = self.value(feature)
        loss = -value.log_prob(target.detach()) - value.log_prob(slow_target.detach())
        return torch.mean(weight * loss.unsqueeze(-1))

    def _cal_grad(self, data, initial):
        """Compute one world-model and actor-critic update."""
        batch_size, sequence_length = data["reward"].shape[:2]
        start_length = min(sequence_length, max(1, self.imag_batch_size // batch_size))
        losses, metrics, _, post = self._world_model_loss(
            data,
            initial,
            return_cache=True,
            cache_start=sequence_length - start_length,
        )
        post_stoch, post_deter = post[:2]
        post_cache = post[3:]

        # Use the most recent K states from every replay sequence. This keeps
        # Dreamer's replay-value targets contiguous while matching the shared
        # imagination-start budget.
        start = self.rssm.imagination_start(
            post_stoch[:, -start_length:],
            post_deter[:, -start_length:],
            post_cache=post_cache,
        )
        with torch.no_grad():
            imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1)

            # (B*K, T_imag, 1)
            imag_reward = self.reward(imag_feat).mode()
            # Probability of continuation.
            imag_cont = self.cont(imag_feat).mean
            imag_value = self.value(imag_feat).mode()
            imag_slow_value = self._slow_value(imag_feat).mode()
            disc = 1 - 1 / self.horizon
            weight = torch.cumprod(imag_cont * disc, dim=1) / disc
            last = torch.zeros_like(imag_cont)
            term = 1 - imag_cont
            ret = self._lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)
            ret_offset, ret_scale = self.return_ema(ret)
            adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_feat.detach())
        logpi = policy.log_prob(imag_action.detach())[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1] * -(logpi * adv.detach() + self.act_entropy * entropy))
        losses["value"] = self._value_loss(
            imag_feat[:, :-1].detach(),
            ret,
            imag_slow_value[:, :-1],
            weight[:, :-1],
        )

        # Dreamer also learns values directly on replay states. The first
        # imagined return provides a bootstrap for each of the K replay states.
        replay_feat = self.rssm.get_feat(
            post_stoch[:, -start_length:],
            post_deter[:, -start_length:],
        )
        replay_last = to_f32(data["is_last"][:, -start_length:])
        replay_term = to_f32(data["is_terminal"][:, -start_length:])
        replay_reward = to_f32(data["reward"][:, -start_length:])
        replay_boot = ret[:, 0].reshape(batch_size, start_length, 1)
        with torch.no_grad():
            replay_value = self.value(replay_feat).mode()
            replay_slow_value = self._slow_value(replay_feat).mode()
            replay_return = self._lambda_return(
                replay_last,
                replay_term,
                replay_reward,
                replay_value,
                replay_boot,
                disc,
                self.lamb,
            )
        replay_weight = 1.0 - replay_last
        losses["repval"] = self._value_loss(
            replay_feat[:, :-1],
            replay_return,
            replay_slow_value[:, :-1],
            replay_weight[:, :-1],
        )

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
        metrics["imag_starts"] = start[0].shape[0]

        return self._backward_losses(losses, metrics)

    def _cal_expert_pretrain_grad(self, data, initial):
        """Compute expert pretraining gradients without imagined policy rollouts."""
        losses, metrics, feat, _ = self._world_model_loss(data, initial, return_cache=False)
        # Feature t is the decision state for action t + 1 in the padded expert sequence.
        policy_feat = feat[:, :-1].detach()
        expert_action = to_f32(data["action"][:, 1:])
        bc_dist = self.actor(policy_feat)
        losses["bc"] = torch.mean(-bc_dist.log_prob(expert_action))

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
        weight = (1.0 - last)[:, :-1]
        losses["repval"] = self._value_loss(policy_feat, ret, slow_value[:, :-1], weight)

        metrics["bc_logprob"] = -losses["bc"]
        return self._backward_losses(losses, metrics)

    def _imagine(self, start, imag_horizon):
        """Roll out the policy in latent space."""
        # (B, S, K), (B, D), optional recurrent/context tensors
        feats = []
        actions = []
        stoch, deter, *cache = start
        cache = tuple(value.clone() for value in cache)
        with self.rssm.sequence_context(stoch):
            for _ in range(imag_horizon):
                # (B, F)
                feat = self.rssm.get_feat(stoch, deter)
                # (B, A)
                action = self.actor(feat).rsample().clamp(-1.0, 1.0)
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
