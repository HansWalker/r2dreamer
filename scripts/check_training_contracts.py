"""Update-level regression checks for Dreamer, STORM, and TD-MPC2.

Run with: python -m scripts.check_training_contracts
Uses small models and synthetic replay; Mamba3 checks require CUDA and its kernels.
Nothing in this module is called by the training launcher.
"""

import copy
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from models.shared.physical_state import STATE_KEY
from scripts.smoke_models import synthetic_batch
from training import load_model_family


VARIANTS = {
    "dreamer": ("gru", "sliding_window", "s5", "hyena", "mamba3"),
    "storm": ("transformer", "sliding_window", "s5", "hyena", "mamba3"),
    "tdmpc2": ("default",),
}


def config_for(family, variant):
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        smoke = compose(config_name="dmc_smoke")
        OmegaConf.resolve(smoke)
        entry = smoke.models[family][variant]
        return compose(config_name=entry.config, overrides=[
            *smoke.training.overrides, *entry.overrides,
            f"device={'cuda:0' if variant == 'mamba3' else 'cpu'}",
            "scenario=ball_in_cup", "state_head.samples_per_update=4",
        ])


def online_update(config, model, batch):
    family = str(config.model_family)
    if family == "tdmpc2":
        return model.update(batch)
    if family == "dreamer":
        contexts = [(batch[index, :0], [0]) for index in range(batch.shape[0])]
        def sample(**kwargs):
            return contexts, batch
    else:
        def sample(batch_size=None, sequence_length=None, with_context=False):
            obs, action, reward, terminal = batch
            selection = (slice(None, batch_size), slice(None, sequence_length))
            result = ({key: value[selection] for key, value in obs.items()},
                      action[selection], reward[selection], terminal[selection])
            return (None, result) if with_context else result
    session = load_model_family(family).OnlineSession(config, model, None)
    session.replay = SimpleNamespace(sample=sample)
    return session.update(1)


def native_optimizers(model, family):
    if family == "dreamer":
        return (model._optimizer,)
    if family == "storm":
        return model.world_model.optimizer, model.actor_critic.optimizer
    return model.model_optimizer, model.policy_optimizer


def target_pair(model, family):
    if family == "dreamer":
        mix = model.slow_target_fraction if model._slow_value_updates % model.slow_target_update == 0 else 0
        return model._slow_value, model.value, mix
    if family == "storm":
        agent = model.actor_critic
        return agent.slow_critic, agent.critic, 1 - agent.slow_critic_decay
    return model.target_qs, model.qs, model.tau


class TrainingContractsTest(unittest.TestCase):
    def assert_tree_equal(self, left, right):
        if isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        elif isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=False)
        else:
            self.assertEqual(left, right)

    def assert_optimizer_coverage(self, model, family):
        native = [p for name, p in model.named_parameters() if p.requires_grad and "state_head." not in name]
        actual = [p for opt in native_optimizers(model, family) for group in opt.param_groups for p in group["params"]]
        self.assertEqual(Counter(map(id, actual)), Counter(map(id, native)))
        head = [p for group in model.state_head.optimizer.param_groups for p in group["params"]]
        self.assertFalse(set(map(id, head)) & set(map(id, actual)))

    def check_updates(self, family_name, variant):
        torch.manual_seed(19)
        config = config_for(family_name, variant)
        if family_name == "storm":
            # Tiny random posteriors can legitimately stay below the free-bits floor.
            # Disable it only here to test that the prior receives learning gradients.
            config.storm_model.kl_free = 0
        adapter = load_model_family(family_name)
        model = adapter.build_model(config)
        self.assert_optimizer_coverage(model, family_name)
        batch, _, _ = synthetic_batch(config, model)
        if family_name == "dreamer":
            batch = batch.to(config.device)
        initial = {name: p.detach().clone() for name, p in model.named_parameters()}

        for step in range(4):
            target, _, _ = target_pair(model, family_name)
            old_target = [p.detach().clone() for p in target.parameters()]
            metrics = adapter.expert_update(model, batch) if step < 2 else online_update(config, model, batch)
            self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values()))
            self.assertEqual(int(model.state_head.updates), step + 1)
            target, source, mix = target_pair(model, family_name)
            self.assertFalse(target.training)
            for before, after, live in zip(old_target, target.parameters(), source.parameters()):
                self.assertFalse(after.requires_grad)
                self.assertIsNone(after.grad)
                torch.testing.assert_close(after, before * (1 - mix) + live.detach() * mix)

        components = {
            "dreamer": ("encoder", "rssm", "decoder", "actor", "value", "reward", "cont"),
            "storm": ("world_model.encoder", "world_model.sequence_core", "world_model.posterior",
                      "world_model.prior", "world_model.decoder", "world_model.reward",
                      "world_model.termination", "actor_critic.actor", "actor_critic.critic"),
            "tdmpc2": ("encoder", "dynamics", "reward", "policy", "qs"),
        }[family_name]
        for prefix in components:
            parameters = [(name, p) for name, p in model.named_parameters() if name.startswith(prefix + ".")]
            self.assertTrue(parameters, prefix)
            self.assertTrue(any(not torch.equal(initial[name], p) for name, p in parameters), prefix)
        self.assert_optimizer_coverage(model, family_name)

        # Restoring optimizer moments, schedules, and EMAs must reproduce the next update.
        restored = adapter.build_model(config)
        adapter.load_checkpoint(restored, copy.deepcopy(adapter.checkpoint(model)), training=True)
        torch.manual_seed(97)
        online_update(config, model, batch)
        torch.manual_seed(97)
        online_update(config, restored, batch)
        self.assert_tree_equal(adapter.checkpoint(model), adapter.checkpoint(restored))

    def check_label_isolation(self, family_name, variant):
        torch.manual_seed(23)
        config = config_for(family_name, variant)
        adapter = load_model_family(family_name)
        models = [adapter.build_model(config), adapter.build_model(config)]
        adapter.load_checkpoint(models[1], copy.deepcopy(adapter.checkpoint(models[0])), training=True)
        batch, _, _ = synthetic_batch(config, models[0])
        if family_name == "dreamer":
            batch = batch.to(config.device)
        poisoned = copy.deepcopy(batch)
        labels = poisoned[STATE_KEY] if family_name == "dreamer" else poisoned[0][STATE_KEY]
        labels.fill_(10_000)
        for model, data in zip(models, (batch, poisoned)):
            torch.manual_seed(53)
            adapter.expert_update(model, data)
            owner = model.world_model if family_name == "storm" else model
            model.state_head.configure_online(lambda owner=owner, data=data: owner.readout_features(data))
            online_update(config, model, data)

        for (name, left), (other_name, right) in zip(models[0].state_dict().items(), models[1].state_dict().items()):
            self.assertEqual(name, other_name)
            if "state_head." not in name:
                torch.testing.assert_close(left, right, rtol=0, atol=0, msg=name)
        for left, right in zip(native_optimizers(models[0], family_name), native_optimizers(models[1], family_name)):
            self.assert_tree_equal(left.state_dict(), right.state_dict())
        self.assertTrue(any(not torch.equal(left, right) for left, right in
                            zip(models[0].state_head.parameters(), models[1].state_head.parameters())))

    def test_td_targets_bootstrap_timeouts_but_not_true_terminals(self):
        config = config_for("tdmpc2", "default")
        model = load_model_family("tdmpc2").build_model(config)
        latent = torch.randn(2, 3, model.latent_dim, requires_grad=True)
        reward = torch.ones(2, 3, 1, requires_grad=True)
        terminal = torch.zeros_like(reward)
        terminal[1] = 1
        with patch.object(model, "_q_value", return_value=torch.full_like(reward, 7)):
            target = model._td_target(latent, reward, terminal)
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(target[0], torch.full_like(target[0], 1 + model.gamma * 7))
        torch.testing.assert_close(target[1], torch.ones_like(target[1]))

    def test_storm_sequence_and_recurrent_gradients_match(self):
        for variant in ("transformer", "sliding_window", "s5", "hyena"):
            with self.subTest(variant=variant):
                torch.manual_seed(61)
                config = config_for("storm", variant)
                config.storm_model.recurrent.transformer.max_length = 16
                config.storm_model.recurrent.sliding_window.window_size = 4
                if variant in {"transformer", "sliding_window"}:
                    config.storm_model.recurrent.layers = 2
                model = load_model_family("storm").build_model(config).world_model.eval()
                reference = copy.deepcopy(model)
                stoch = torch.randn(2, 11, model.stoch_flattened_dim)
                action = torch.randn(2, 11, 2)

                # Repeat after a parameter update to catch retained graphs/stale constants.
                for _ in range(2):
                    outputs = []
                    for instance, optimized in ((model, True), (reference, False)):
                        instance.zero_grad(set_to_none=True)
                        core = instance.sequence_core
                        cache = core.initial_cache(2, torch.float32)
                        with torch.no_grad():
                            for index in range(3):
                                _, cache = core.step(stoch[:, index:index + 1], action[:, index:index + 1], cache)
                        if optimized:
                            with instance.sequence_context(stoch):
                                output, _ = instance._scan(stoch[:, 3:], action[:, 3:], cache)
                        else:
                            values = []
                            for index in range(3, 11):
                                value, cache = core.step(stoch[:, index:index + 1], action[:, index:index + 1], cache)
                                values.append(value)
                            output = torch.cat(values, dim=1)
                        output.sin().sum().backward()
                        outputs.append(output)
                    torch.testing.assert_close(*outputs, atol=2e-5, rtol=2e-5)
                    with torch.no_grad():
                        for (name, left), (_, right) in zip(model.sequence_core.named_parameters(),
                                                          reference.sequence_core.named_parameters()):
                            self.assertIsNotNone(left.grad, name)
                            torch.testing.assert_close(left.grad, right.grad, atol=2e-4, rtol=2e-4, msg=name)
                            # Use the same increment so the next comparison isolates cache reuse.
                            increment = -1e-4 * left.grad
                            left.add_(increment)
                            right.add_(increment)

    def test_storm_transformer_teacher_forcing_is_causal_and_matches_acting(self):
        config = config_for("storm", "transformer")
        config.storm_model.recurrent.transformer.max_length = 16
        config.storm_model.recurrent.layers = 2
        model = load_model_family("storm").build_model(config).world_model.eval()
        core = model.sequence_core
        stoch = torch.randn(2, 11, model.stoch_flattened_dim)
        action = torch.randn(2, 11, 2)
        with torch.no_grad():
            expected = core(stoch, action)
            cache, outputs = None, []
            for index in range(11):
                value, cache = core.step(stoch[:, index:index + 1], action[:, index:index + 1], cache)
                outputs.append(value)
            torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=2e-5, rtol=2e-5)
            changed_stoch, changed_action = stoch.clone(), action.clone()
            changed_stoch[:, 5:] = torch.randn_like(changed_stoch[:, 5:]) * 10
            changed_action[:, 5:] *= -10
            changed = core(changed_stoch, changed_action)
            torch.testing.assert_close(changed[:, :5], expected[:, :5], rtol=0, atol=0)


def variant_test(check, family, variant):
    def test(self):
        if variant == "mamba3" and not torch.cuda.is_available():
            self.skipTest("Mamba3 training requires CUDA and installed Mamba3 kernels")
        check(self, family, variant)
    return test


for _family, _variants in VARIANTS.items():
    for _variant in _variants:
        for _name, _check in (("updates", TrainingContractsTest.check_updates),
                             ("label_isolation", TrainingContractsTest.check_label_isolation)):
            setattr(TrainingContractsTest, f"test_{_family}_{_variant}_{_name}", variant_test(_check, _family, _variant))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
