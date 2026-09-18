"""CPU checks for physical-head loss conditioning and prediction-preserving migration.

Run with: python -m scripts.check_state_normalization
"""

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from models.shared.physical_state import TARGET_VERSION, PhysicalStateHead, readout_mode
from scripts.smoke_models import synthetic_batch
from training import load_model_family


def settings(scenario="cartpole_balance_sparse"):
    task, fields = {
        "cartpole_balance_sparse": ("cartpole_balance_sparse", {"position": [0, 1, 2], "velocity": [0, 1]}),
        "reacher": ("reacher_easy", {"position": [0, 1], "to_target": [0, 1], "velocity": [0, 1]}),
        "ball_in_cup": ("ball_in_cup_catch", {"position": [0, 1, 2, 3], "velocity": [0, 1, 2, 3]}),
        "point_mass": ("point_mass_easy", {"position": [0, 1], "velocity": [0, 1]}),
    }[scenario]
    return SimpleNamespace(
        target_version=TARGET_VERSION, task=f"dmc_{task}", fields=fields,
        samples_per_update=4, grad_clip=10, projection_dim=4, hidden_dim=8, lr=3e-4,
    )


def tiny_config(family, scenario):
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        smoke = compose(config_name="dmc_smoke")
        OmegaConf.resolve(smoke)
        entry = next(iter(smoke.models[family].values()))
        return compose(config_name=entry.config, overrides=[
            *smoke.training.overrides, *entry.overrides, "device=cpu", f"scenario={scenario}",
        ])


def legacy_weights(weights, prefix="", version=1):
    weights = copy.deepcopy(weights)
    scale = weights.pop(prefix + "output_scale")
    if version == 2:
        weights[prefix + "training_scale"] = scale
    del weights[prefix + "loss_scale"], weights[prefix + "online_updates"]
    weights._metadata[prefix.rstrip(".")]["version"] = version
    return weights


class StateNormalizationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_output_loss_and_evaluation_scales_have_distinct_roles(self):
        for scenario, angular in (
            ("cartpole_balance_sparse", [1, 2]), ("reacher", [0, 1, 2, 3]),
            ("ball_in_cup", []), ("point_mass", []),
        ):
            with self.subTest(scenario=scenario):
                config = settings(scenario)
                head = PhysicalStateHead(8, config)
                count = len(head.coordinates)
                mean, std = torch.arange(count).float(), torch.arange(count).float() / 10
                head.set_stats(mean, std)
                self.assertEqual(head.targets.trigonometric, angular)
                torch.testing.assert_close(head.mean, mean, rtol=0, atol=0)
                torch.testing.assert_close(head.std, std.clamp_min(1e-3), rtol=0, atol=0)
                torch.testing.assert_close(head.output_scale, std.clamp_min(1e-3), rtol=0, atol=0)
                torch.testing.assert_close(head.loss_scale, torch.ones(count), rtol=0, atol=0)
                # Field order must not change which coordinates receive unit scale.
                config.fields = dict(reversed(list(config.fields.items())))
                reordered = PhysicalStateHead(8, config)
                self.assertEqual(
                    {head.coordinates[index] for index in angular},
                    {reordered.coordinates[index] for index in reordered.targets.trigonometric},
                )

    def test_online_only_defaults_remain_zero_mean_unit_scale(self):
        head = PhysicalStateHead(8, settings())
        head.prepare_training()
        torch.testing.assert_close(head.mean, torch.zeros(5), rtol=0, atol=0)
        torch.testing.assert_close(head.std, torch.ones(5), rtol=0, atol=0)
        torch.testing.assert_close(head.output_scale, torch.ones(5), rtol=0, atol=0)
        torch.testing.assert_close(head.loss_scale, torch.ones(5), rtol=0, atol=0)

    def test_horizontal_pole_does_not_overwhelm_loss_and_features_stay_detached(self):
        head = PhysicalStateHead(8, settings())
        head.set_stats([0, 1, 0, 0, 0], [.036, .001, .01, .15, .21])
        with torch.no_grad():
            head.readout[-1].weight.zero_()
            head.readout[-1].bias.zero_()
        features = torch.randn(2, 2, 8, requires_grad=True)
        labels = torch.tensor([.036, 0, 1, 0, 0]).expand(2, 2, -1)
        prediction = head(features)
        expected = torch.nn.functional.smooth_l1_loss(prediction, labels)
        old_loss = ((prediction - labels) / head.std).square().mean()
        self.assertGreater(old_loss.item(), 200_000)
        self.assertLess(expected.item(), 1)
        bias_gradients = []
        hook = head.readout[-1].bias.register_hook(lambda gradient: bias_gradients.append(gradient.clone()))
        metrics = head.fit(features, labels)
        hook.remove()
        torch.testing.assert_close(metrics["state/loss"], expected)
        self.assertLess(bias_gradients[0].norm().item(), 1)
        self.assertIsNone(features.grad)
        self.assertEqual(head.updates.item(), 1)
        self.assertEqual(head.examples.item(), 4)
        torch.testing.assert_close(head.std, torch.tensor([.036, .001, .01, .15, .21]), rtol=0, atol=0)

    def test_legacy_load_preserves_predictions_then_migrates_only_head_conditioning(self):
        for history, tokens in ((1, 1), (3, 1), (3, 16)):
            with self.subTest(history=history, tokens=tokens):
                old = PhysicalStateHead(8, settings(), history=history, tokens=tokens)
                old.set_stats([.06, 1, 0, .008, 0], [.036, .001, .01, .15, .21])
                features = torch.randn(2, 5, tokens, 8, requires_grad=True)
                labels = torch.randn(2, 5, 5)
                old.fit(features, labels)
                weights = legacy_weights(old.state_dict())
                original = copy.deepcopy(weights)
                optimizer = copy.deepcopy(old.optimizer.state_dict())
                head = PhysicalStateHead(8, settings(), history=history, tokens=tokens)
                head.load_state_dict(weights)
                torch.testing.assert_close(head(features), old(features), rtol=0, atol=0)
                self.assertEqual(weights.keys(), original.keys())
                head.load_optimizer_state_dict(optimizer)
                self.assertFalse(head.optimizer.state)
                torch.testing.assert_close(head(features), old(features), rtol=2e-6, atol=1e-7)
                # Planners differentiate these predictions with respect to latent states/actions.
                before = torch.autograd.grad(old(features).sum(), features)[0]
                after = torch.autograd.grad(head(features).sum(), features)[0]
                torch.testing.assert_close(after, before, rtol=2e-5, atol=1e-7)
                for name, value in original.items():
                    torch.testing.assert_close(weights[name], value, rtol=0, atol=0)
                    torch.testing.assert_close(head.state_dict()[name], value, rtol=0, atol=0)
                migrated = copy.deepcopy(head.state_dict())
                head.set_stats(old.mean, old.std)
                head.prepare_training()
                for name, value in migrated.items():
                    torch.testing.assert_close(head.state_dict()[name], value, rtol=0, atol=0)
                head.fit(features, labels)
                self.assertTrue(head.optimizer.state)
                self.assertEqual(head.updates.item(), 2)

    def test_modern_resume_retains_optimizer_and_next_update_exactly(self):
        head = PhysicalStateHead(8, settings(), history=3, tokens=2)
        head.set_stats([0, 1, 0, 0, 0], [.03, .001, .01, .15, .21])
        features, labels = torch.randn(2, 5, 2, 8), torch.randn(2, 5, 5)
        head.configure_online(lambda: (features, labels))
        head.fit(features, labels)
        restored = PhysicalStateHead(8, settings(), history=3, tokens=2)
        restored.load_state_dict(copy.deepcopy(head.state_dict()))
        restored.load_optimizer_state_dict(copy.deepcopy(head.optimizer.state_dict()))
        restored.configure_online(lambda: (features, labels), resumed=True)
        self.assertTrue(restored.optimizer.state)
        restored.set_stats(head.mean, head.std)
        for instance in (head, restored):
            instance.fit(features, labels)
        for name, value in head.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)

    def test_v2_affine_is_preserved_but_old_loss_moments_are_reset(self):
        head = PhysicalStateHead(8, settings("ball_in_cup"))
        head.set_stats(torch.zeros(8), torch.linspace(.01, .2, 8))
        head.fit(torch.randn(2, 3, 8), torch.randn(2, 3, 8))
        head.output_scale.fill_(1)
        weights = legacy_weights(head.state_dict(), version=2)
        restored = PhysicalStateHead(8, settings("ball_in_cup"))
        restored.load_state_dict(weights)
        restored.load_optimizer_state_dict(copy.deepcopy(head.optimizer.state_dict()))
        self.assertFalse(restored.optimizer.state)
        for name, value in head.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)

    def test_missing_unrelated_weights_are_still_rejected(self):
        head = PhysicalStateHead(8, settings())
        weights = legacy_weights(head.state_dict())
        del weights["project.0.weight"]
        with self.assertRaisesRegex(RuntimeError, "project.0.weight"):
            head.load_state_dict(weights)

    def test_modern_checkpoint_requires_its_scales(self):
        head = PhysicalStateHead(8, settings())
        weights = copy.deepcopy(head.state_dict())
        del weights["output_scale"]
        with self.assertRaisesRegex(RuntimeError, "output_scale"):
            head.load_state_dict(weights)

    def test_online_budget_warmup_and_detachment(self):
        head = PhysicalStateHead(8, settings())
        x, y = torch.randn(2, 3, 8, requires_grad=True), torch.randn(2, 3, 5)
        expert = torch.randn(2, 3, 8, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "expert source"):
            head.configure_online(None)
        head.configure_online(lambda: (expert, y))
        metrics = head.fit(x, y)
        self.assertEqual(metrics["state/expert_examples"], 2)
        self.assertEqual(metrics["state/examples"], 4)
        self.assertAlmostEqual(metrics["state/lr"], 3e-7)
        self.assertEqual(head.online_updates.item(), 1)
        self.assertIsNone(x.grad)
        self.assertIsNone(expert.grad)

    def test_feature_extraction_preserves_rng_bn_and_individual_modes(self):
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.BatchNorm1d(8), torch.nn.Dropout(.5))
        model[2].eval()
        rng = torch.get_rng_state().clone()
        state = copy.deepcopy(model.state_dict())
        with readout_mode(model):
            self.assertFalse(model.training)
            output = model(torch.rand(4, 8))
            self.assertFalse(output.requires_grad)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertTrue(model[1].training)
        self.assertFalse(model[2].training)
        for key, value in state.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_readout_features_do_not_change_native_expert_updates(self):
        for name in ("dreamer", "storm", "tdmpc2", "leworldmodel", "temporal_straightening"):
            with self.subTest(family=name):
                config = tiny_config(name, "cartpole_balance_sparse")
                family = load_model_family(name)
                left, right = family.build_model(config), family.build_model(config)
                right.load_state_dict(left.state_dict())
                batch, _, _ = synthetic_batch(config, left)
                rng = torch.get_rng_state().clone()
                family.expert_update(left, batch)
                after_rng = torch.get_rng_state().clone()
                torch.set_rng_state(rng)
                feature_owner = right.world_model if name == "storm" else right
                with patch.object(feature_owner, "readout_features", return_value=(None, None)), patch.object(right.state_head, "fit", return_value={}):
                    family.expert_update(right, batch)
                torch.testing.assert_close(torch.get_rng_state(), after_rng, rtol=0, atol=0)
                for key, value in left.state_dict().items():
                    if "state_head." not in key:
                        torch.testing.assert_close(right.state_dict()[key], value, rtol=0, atol=0)

    def test_all_family_checkpoint_paths_migrate_and_preserve_native_state(self):
        for family in ("dreamer", "storm", "tdmpc2", "leworldmodel", "temporal_straightening"):
            for scenario in ("cartpole_balance_sparse", "reacher", "ball_in_cup"):
                with self.subTest(family=family, scenario=scenario):
                    config = tiny_config(family, scenario)
                    adapter = load_model_family(family)
                    model = adapter.build_model(config)
                    head = model.state_head
                    count = len(head.coordinates)
                    head.set_stats(torch.zeros(count), torch.linspace(.001, .2, count))
                    batch, _, _ = synthetic_batch(config, model)
                    adapter.expert_update(model, batch)
                    payload = copy.deepcopy(adapter.checkpoint(model))
                    state_key = {"dreamer": "agent_state_dict", "storm": "world_model"}.get(family, "model_state_dict")
                    payload[state_key] = legacy_weights(payload[state_key], "state_head.")
                    restored = adapter.build_model(config)
                    adapter.load_checkpoint(restored, copy.deepcopy(payload), training=False)
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
                    adapter.load_checkpoint(restored, copy.deepcopy(payload), training=True)
                    self.assertFalse(restored.state_head.optimizer.state)
                    for name, value in model.state_dict().items():
                        if "state_head." not in name:
                            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
                    after = adapter.checkpoint(restored)
                    for key, value in payload.items():
                        if "optim" in key and key != "state_optimizer":
                            self.assert_nested_equal_without_head(value, after[key])

    def assert_nested_equal_without_head(self, expected, actual):
        if isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key, value in expected.items():
                if key not in {"state_head", "state_optimizer"}:
                    self.assert_nested_equal_without_head(value, actual[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_nested_equal_without_head(left, right)
        elif isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
