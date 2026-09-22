"""CPU contracts for the opt-in TS spatial aggregation recipe."""

import copy
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from scripts.smoke_models import synthetic_batch
from scripts.smoke_tiny_planners import tiny_config
from training import load_model_family


def config_for(mode="patch", full=False):
    if full:
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
                                   version_base=None):
            config = compose(config_name="temporal_straightening_dmc_vision", overrides=["device=cpu"])
        OmegaConf.resolve(config)
    else:
        args = SimpleNamespace(scenario="cartpole_balance_sparse", device="cpu", seed=0,
                               episode_steps=64, offline_updates=128, online_updates=128)
        config = tiny_config("temporal_straightening", args)
    with open_dict(config.jepa_model):
        if mode is None:
            config.jepa_model.pop("curvature_mode", None)
            config.jepa_model.pop("aggregation", None)
        else:
            config.jepa_model.curvature_mode = mode
    return config


def build(config, seed=319):
    torch.manual_seed(seed)
    return load_model_family("temporal_straightening").build_model(config)


def head_parameters(model):
    return {name: value for name, value in model.named_parameters() if name.startswith("encoder.agg_")}


class AggregationTests(unittest.TestCase):
    def test_missing_mode_and_patch_keep_weights_rng_optimizer_order_and_loss(self):
        legacy = build(config_for(None))
        legacy_rng = torch.get_rng_state().clone()
        explicit = build(config_for("patch"))
        self.assertTrue(torch.equal(legacy_rng, torch.get_rng_state()))
        self.assertEqual(legacy.curvature_mode, "patch")
        self.assertFalse(head_parameters(explicit))
        self.assertEqual(list(legacy.state_dict()), list(explicit.state_dict()))
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[name]), name)
        expected = [*explicit.encoder.backbone.parameters(), *explicit.encoder.norm.parameters()]
        actual = explicit.optimizers["encoder"].param_groups[0]["params"]
        self.assertEqual([id(p) for p in expected], [id(p) for p in actual])

        latent = torch.randn(2, 4, 16, 16, requires_grad=True)
        predicted = torch.randn(2, 3, 16, 16, requires_grad=True)
        velocity = latent[:, 1:] - latent[:, :-1]
        previous, current = velocity[:, :-1], velocity[:, 1:]
        moving = (previous.norm(dim=-1) > 1e-6) & (current.norm(dim=-1) > 1e-6)
        original = (explicit.prediction_weight * F.mse_loss(predicted, latent[:, 1:].detach())
                    + explicit.curvature_weight * (1 - F.cosine_similarity(
                        previous, current, dim=-1, eps=1e-6))[moving].mean())
        with patch.object(explicit, "predict", return_value=predicted):
            actual_loss, _ = explicit.representation_loss({}, latent, None)
        self.assertTrue(torch.equal(original, actual_loss))
        for before, after in zip(torch.autograd.grad(original, (latent, predicted), retain_graph=True),
                                 torch.autograd.grad(actual_loss, (latent, predicted))):
            self.assertTrue(torch.equal(before, after))

    def test_aggregation_preserves_common_initialization_and_spatial_planning(self):
        baseline = build(config_for("patch")).eval()
        baseline_rng = torch.get_rng_state().clone()
        aggregate = build(config_for("agg")).eval()
        self.assertTrue(torch.equal(baseline_rng, torch.get_rng_state()))
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, aggregate.state_dict()[name]), name)
        observations = {"image": torch.randint(256, (2, 3, 64, 64, 3), dtype=torch.uint8)}
        past = torch.randn(2, 2, 1)
        candidates = torch.randn(2, 2, 5, 1, requires_grad=True)
        goal = baseline.encode({"image": observations["image"][:, -1]})
        with patch.object(aggregate.encoder, "agg", side_effect=AssertionError("Aggregation in planner")):
            encoded = [model.encode(observations) for model in (baseline, aggregate)]
            self.assertTrue(torch.equal(*encoded))
            self.assertEqual(encoded[0].shape, (2, 3, 16, 16))
            costs = [model._goal_cost(history, past, candidates, goal)
                     for model, history in zip((baseline, aggregate), encoded)]
            self.assertTrue(torch.equal(*costs))
            gradients = [torch.autograd.grad(cost.sum(), candidates, retain_graph=True)[0] for cost in costs]
            self.assertTrue(torch.equal(*gradients))

    def test_nonlinear_aggregation_precedes_differences_and_constant_motion_is_finite(self):
        model = build(config_for("agg"))
        latent = torch.randn(2, 4, 16, 16, requires_grad=True)
        pooled = model.encoder.agg(latent.reshape(8, 16, 16)).reshape(2, 4, -1)
        velocity = pooled[:, 1:] - pooled[:, :-1]
        expected = (1 - F.cosine_similarity(velocity[:, :-1], velocity[:, 1:], dim=-1, eps=1e-6)).mean()
        wrong = model.encoder.agg((latent[:, 1:] - latent[:, :-1]).reshape(6, 16, 16)).reshape(2, 3, -1)
        wrong_loss = (1 - F.cosine_similarity(wrong[:, :-1], wrong[:, 1:], dim=-1, eps=1e-6)).mean()
        with patch.object(model, "predict", return_value=latent[:, 1:]) as predict:
            loss, metrics = model.representation_loss({}, latent, None)
        self.assertTrue(torch.equal(predict.call_args.args[0], latent[:, :-1]))
        torch.testing.assert_close(metrics["curvature_loss"], expected)
        self.assertGreater(abs(float((expected - wrong_loss).detach())), .1)
        gradients = torch.autograd.grad(loss, (latent, *head_parameters(model).values()))
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        self.assertGreater(float(gradients[0].abs().sum()), 0.)
        self.assertGreater(sum(float(value.abs().sum()) for value in gradients[1:]), 0.)
        self.assertGreater(float(metrics["spatial_spread"]), 0.)
        self.assertGreater(float(metrics["aggregate_spread"]), 0.)
        self.assertEqual(float(metrics["curvature_moving_fraction"]), 1.)

        constant = torch.zeros_like(latent, requires_grad=True)
        with patch.object(model, "predict", return_value=constant[:, 1:]):
            loss, metrics = model.representation_loss({}, constant, None)
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(float(metrics["curvature_loss"]), 0.)
        self.assertEqual(float(metrics["aggregate_spread"]), 0.)
        self.assertEqual(float(metrics["curvature_moving_fraction"]), 0.)
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, constant)[0]).all())

    def test_update_clips_and_optimizes_head_and_checkpoint_resumes_it(self):
        config = config_for("agg")
        model = build(config)
        before = {name: value.detach().clone() for name, value in head_parameters(model).items()}
        head_ids = {id(value) for value in head_parameters(model).values()}
        optimizer_ids = {id(value) for group in model.optimizers["encoder"].param_groups for value in group["params"]}
        self.assertTrue(head_ids <= optimizer_ids)
        calls = []
        clip = torch.nn.utils.clip_grad_norm_

        def tracked_clip(parameters, *args, **kwargs):
            parameters = list(parameters)
            calls.append({id(value) for value in parameters})
            return clip(parameters, *args, **kwargs)

        batch, _, _ = synthetic_batch(config, model)
        with patch("torch.nn.utils.clip_grad_norm_", side_effect=tracked_clip):
            metrics = model.update(batch)
        self.assertTrue(head_ids <= calls[0])
        self.assertTrue(all(torch.isfinite(torch.as_tensor(value)) for value in metrics.values()))
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in head_parameters(model).items()))
        self.assertTrue(all(value in model.optimizers["encoder"].state for value in head_parameters(model).values()))
        family = load_model_family("temporal_straightening")
        stream = io.BytesIO()
        torch.save(family.checkpoint(model), stream)
        stream.seek(0)
        restored = build(config, seed=123)
        family.load_checkpoint(restored, torch.load(stream, weights_only=False), training=True)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)
        for parameter, restored_parameter in zip(model.encoder.parameters(), restored.encoder.parameters()):
            for key, value in model.optimizers["encoder"].state[parameter].items():
                self.assertTrue(torch.equal(value, restored.optimizers["encoder"].state[restored_parameter][key]))
        torch.manual_seed(71)
        model.update(batch)
        torch.manual_seed(71)
        restored.update(batch)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)

    def test_exact_full_parameter_delta_and_configurable_widths(self):
        baseline = build(config_for("patch", full=True))
        aggregate = build(config_for("agg", full=True))
        difference = sum(p.numel() for p in aggregate.parameters()) - sum(p.numel() for p in baseline.parameters())
        self.assertEqual(difference, 2_033_024)
        self.assertEqual(difference, sum(p.numel() for p in head_parameters(aggregate).values()))
        self.assertEqual([type(layer) for layer in aggregate.encoder.agg_mlp],
                         [torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear])
        config = config_for("agg")
        config.jepa_model.aggregation.hidden_dim = 19
        config.jepa_model.aggregation.output_dim = 7
        narrow = build(config)
        self.assertEqual(narrow.encoder.agg_mlp[0].out_features, 19)
        self.assertEqual(narrow.encoder.agg(torch.randn(2, 16, 16)).shape, (2, 7))
        for mode, width in (("unknown", 19), ("agg", 0)):
            invalid = copy.deepcopy(config)
            invalid.jepa_model.curvature_mode = mode
            invalid.jepa_model.aggregation.hidden_dim = width
            with self.assertRaises(ValueError):
                build(invalid)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
