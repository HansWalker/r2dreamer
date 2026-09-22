"""CPU contracts for the opt-in existing-aggregation-head planning objective."""

import copy
import unittest
from unittest.mock import patch

import torch
from omegaconf import open_dict

from models.shared.latent_goal import latent_goal_cost
from models.shared.physical_state import readout_mode
from scripts.check_paper_faithful_support import fake_case
from scripts.check_state_normalization import tiny_config
from scripts.check_ts_aggregation import build, config_for
from scripts.paper_faithful_support import make_split_manifest, score_branches
from training import load_model_family


def nonlinear(value):
    flat = value.flatten(1)
    return torch.stack((flat[:, 0].square() + .5 * flat[:, 1],
                        flat[:, 0] * flat[:, 1] - flat[:, 1].square()), -1)


def manual_cost(prediction, goal, history, mode, reduction, tail_steps=3):
    """Independent scalar/time reference; goals are already explicit [B,G,...]."""
    if mode == "last":
        path = prediction[:, :, -1:]
    elif mode == "tail":
        path = prediction[:, :, -tail_steps:]
    else:
        path = torch.cat((history.detach()[:, None].expand(-1, prediction.shape[1], -1, *history.shape[2:]),
                          prediction), 2)
    error = (path[..., None, :, :] - goal.detach()[:, None, None]).square().flatten(-2)
    error = error.mean(-1) if reduction == "mean" else error.sum(-1)
    if mode == "ts_mpc":
        weights = torch.exp2(torch.arange(1 - path.shape[2], 1, dtype=torch.float32))
        error = error * (weights / weights.sum())[None, None, :, None]
    return error.mean(2)


class PlanningCostTests(unittest.TestCase):
    def test_absent_and_zero_weight_preserve_exact_cost_and_action_gradients(self):
        for mode in ("patch", "agg"):
            model = build(config_for(mode)).eval()
            history = torch.randn(2, 3, 16, 16)
            goal = torch.randn(2, 2, 16, 16)
            past = torch.randn(2, 2, 1)
            action = torch.randn(2, 3, 4, 1, requires_grad=True)
            prediction = model.rollout(history, past, action)
            expected = latent_goal_cost(prediction, goal, reduction=model.goal_reduction,
                                        mode="ts_mpc", history=history)
            gradient = torch.autograd.grad(expected.sum(), action, retain_graph=True)[0]
            for weight in (None, 0.):
                with open_dict(model.planner):
                    if weight is None:
                        model.planner.pop("aggregate_goal_weight", None)
                    else:
                        model.planner.aggregate_goal_weight = weight
                with patch.object(model.encoder, "agg", side_effect=AssertionError("Zero-weight head invocation")):
                    actual = model.planning_cost(prediction, goal, history=history)
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(torch.autograd.grad(actual.sum(), action, retain_graph=True)[0], gradient))

    def test_nonlinear_states_are_pooled_before_scoring_with_correct_time_weights(self):
        model = build(config_for("agg")).eval()
        model.planner.aggregate_goal_weight = .1
        prediction = torch.randn(2, 3, 4, 16, 16, requires_grad=True)
        goal = torch.randn(2, 2, 16, 16, requires_grad=True)
        history = torch.randn(2, 3, 16, 16, requires_grad=True)

        def pooled(value):
            return nonlinear(value.reshape(-1, 16, 16)).reshape(*value.shape[:-2], 1, 2)

        for mode in ("last", "tail", "ts_mpc"):
            for reduction in ("sum", "mean"):
                model.planner.objective, model.goal_reduction = mode, reduction
                spatial = manual_cost(prediction, goal, history, mode, reduction)
                aggregate = manual_cost(pooled(prediction), pooled(goal), pooled(history), mode, reduction)
                expected = (spatial + .1 * aggregate).min(-1).values
                with patch.object(model.encoder, "agg", side_effect=nonlinear):
                    actual = model.planning_cost(prediction, goal, history=history)
                torch.testing.assert_close(actual, expected)
                reference_gradient = torch.autograd.grad(expected.sum(), prediction, retain_graph=True)[0]
                gradients = torch.autograd.grad(actual.sum(), (prediction, goal, history), allow_unused=True, retain_graph=True)
                torch.testing.assert_close(gradients[0], reference_gradient)
                self.assertIsNone(gradients[1])
                self.assertIsNone(gradients[2])
                self.assertGreater(float(gradients[0].abs().sum()), 0.)

    def test_goal_minimum_is_shared_between_spatial_and_aggregate_terms(self):
        model = build(config_for("agg")).eval()
        model.planner.aggregate_goal_weight, model.planner.objective = .1, "last"
        prediction = torch.ones(1, 1, 1, 16, 16)
        goals = torch.stack((-torch.ones(16, 16), 2 * torch.ones(16, 16)))[None]
        with patch.object(model.encoder, "agg", side_effect=lambda value: value.flatten(1).mean(1, keepdim=True).square()):
            actual = model.planning_cost(prediction, goals)
        # Goal -1: 4 + .1*0; goal +2: 1 + .1*9. Separate minima would give 1.
        torch.testing.assert_close(actual, torch.tensor([[1.9]]))

    def test_invalid_weights_and_missing_or_non_ts_heads_are_rejected(self):
        for value in (-.1, float("nan"), float("inf")):
            config = config_for("agg")
            config.jepa_model.planner.aggregate_goal_weight = value
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                build(config)
        config = config_for("patch")
        config.jepa_model.planner.aggregate_goal_weight = .1
        with self.assertRaisesRegex(ValueError, "existing TS aggregation head"):
            build(config)
        config = tiny_config("leworldmodel", "cartpole_balance_sparse")
        with open_dict(config.jepa_model.planner):
            config.jepa_model.planner.aggregate_goal_weight = .1
        with self.assertRaisesRegex(ValueError, "existing TS aggregation head"):
            load_model_family("leworldmodel").build_model(config)
        model = build(config_for("agg"))
        model.planner.aggregate_goal_weight = float("nan")
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            model.planning_cost(torch.zeros(1, 1, 1, 16, 16), torch.zeros(1, 16, 16))

    def test_real_head_preserves_action_gradients_and_production_diagnostic_parity(self):
        model = build(config_for("agg")).eval()
        model.planner.aggregate_goal_weight = .1
        case = fake_case(make_split_manifest(1, 1, 1, seed=81)["test"][0], images=True)
        before = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        metrics = score_branches(model, [case], horizons=(1, 3))
        self.assertEqual(metrics["aggregate_goal_weight"], .1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        with readout_mode(model):
            history = model.encode({"image": case["prefix"][None]})
            goal = model.encode({"image": case["goal_image"][None, None]})[:, 0]
            action = case["action"][None, :, :3].clone().requires_grad_()
            with torch.enable_grad():
                production = model._goal_cost(history, case["past_action"][None], action, goal)
                gradient = torch.autograd.grad(production.sum(), action, retain_graph=True)[0]
                prediction = model.rollout(history, case["past_action"][None], action)
                spatial = latent_goal_cost(prediction, goal, history=history,
                                           mode="ts_mpc", reduction=model.goal_reduction)
                aggregate_gradient = torch.autograd.grad((production - spatial).sum(), action)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.)
            self.assertTrue(torch.isfinite(aggregate_gradient).all())
            self.assertGreater(float(aggregate_gradient.abs().sum()), 0.)
            reported = metrics["cases"][0]["metrics"]["3"]["predicted_cost"]
            torch.testing.assert_close(production[0], torch.tensor(reported), rtol=1e-5, atol=1e-5)
            frames = case["image"][:, :3]
            actual = model.encode({"image": frames}).unsqueeze(0)
            cost = model.planning_cost(actual, goal, history=history)
            reported = metrics["cases"][0]["metrics"]["3"]["actual_cost"]
            torch.testing.assert_close(cost[0], torch.tensor(reported), rtol=1e-5, atol=1e-5)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
