"""CPU/simulator contracts for interruption-free native snapshot evaluation."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

import tools
from envs.dmc import make_env
from scripts.check_paper_faithful_support import fake_case
from scripts.check_state_normalization import tiny_config
from scripts.paper_faithful_followup_eval import (
    _equal, evaluate_snapshot, select_policy_cases, summarize_evaluation,
)
from scripts.paper_faithful_support import BranchReplay, collect_anchor, make_split_manifest
from training import load_model_family


def small_config(family):
    config = tiny_config(family, "cartpole_balance_sparse")
    config.env.time_limit = 1000
    config.jepa_model.planner.horizon = 5
    config.jepa_model.planner.samples = 4
    config.jepa_model.planner.elites = 2
    config.jepa_model.planner.iterations = 2
    return config


def assert_rng(test, before):
    after = tools.get_rng_state()
    test.assertEqual(before["python"], after["python"])
    test.assertEqual(before["numpy"][0], after["numpy"][0])
    np.testing.assert_array_equal(before["numpy"][1], after["numpy"][1])
    test.assertEqual(before["numpy"][2:], after["numpy"][2:])
    test.assertTrue(torch.equal(before["torch"], after["torch"]))
    test.assertEqual(before["cuda"], after["cuda"])


class FollowupEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = small_config("temporal_straightening")
        cls.specs = make_split_manifest(3, 3, 3, seed=381)
        cls.cases = []
        for spec in cls.specs["validation"]:
            env = make_env(config.env, spec["seed"], include_physical_state=False)
            try:
                cls.cases.append(collect_anchor(env, spec, candidates=6, horizon=25))
            finally:
                env.close()

    def test_selection_covers_cohorts_and_is_independent_of_input_order(self):
        chosen = select_policy_cases(self.cases, 3)
        self.assertEqual(len({c["cohort"] for c in chosen}), 3)
        self.assertEqual([c["id"] for c in chosen], [c["id"] for c in select_policy_cases(list(reversed(self.cases)), 3)])
        with self.assertRaises(ValueError):
            select_policy_cases(self.cases, 4)

    def test_real_native_policy_and_forecasts_preserve_state_modes_caches_and_gradients(self):
        for family in ("temporal_straightening", "leworldmodel"):
            with self.subTest(family=family):
                config = small_config(family)
                model = load_model_family(family).build_model(config).train()
                model.encoder.eval()
                first = next(model.parameters())
                first.requires_grad_(False)
                for parameter in model.parameters():
                    parameter.grad = torch.ones_like(parameter) * .01
                model._cem_mean = torch.randn(1, 5, 1)
                model._gradient_actions = torch.randn(1, 4, 5, 1)
                caches = (model._cem_mean, model._gradient_actions)
                cache_values = tuple(c.clone() for c in caches)
                before = copy.deepcopy(model.state_dict())
                modes = [module.training for module in model.modules()]
                gradients = [(p.grad, p.grad.clone(), p.requires_grad) for p in model.parameters()]
                rng = tools.get_rng_state()
                result = evaluate_snapshot(config, model, self.cases[:1], horizons=(1, 5, 15, 25), policy_cases=1, policy_steps=2)
                self.assertTrue(_equal(before, model.state_dict()))
                self.assertEqual(modes, [m.training for m in model.modules()])
                for parameter, (original, value, flag) in zip(model.parameters(), gradients):
                    self.assertIs(parameter.grad, original)
                    self.assertTrue(torch.equal(parameter.grad, value))
                    self.assertEqual(parameter.requires_grad, flag)
                self.assertIs(model._cem_mean, caches[0])
                self.assertIs(model._gradient_actions, caches[1])
                self.assertTrue(torch.equal(model._cem_mean, cache_values[0]))
                self.assertTrue(torch.equal(model._gradient_actions, cache_values[1]))
                assert_rng(self, rng)
                self.assertEqual(result["branches"]["horizons"], [1, 5, 15, 25])
                self.assertEqual(result["policy_planning_horizon"], 5)
                self.assertEqual(len(result["policy"]["traces"]), 2)
                self.assertEqual(config.jepa_model.planner.horizon, 5)
                summary = summarize_evaluation(result)
                self.assertEqual(summary["policy"]["cases"], 1)
                self.assertEqual(summary["horizons"]["1"]["informative"]["anchors"], 0)
                self.assertIsNone(summary["horizons"]["1"]["informative"]["matched_mse"])

    def test_next_native_training_update_is_bit_identical_after_evaluation(self):
        config = small_config("leworldmodel")
        model = load_model_family("leworldmodel").build_model(config)
        model.configure_pretraining(20)
        replay = BranchReplay([fake_case(spec, images=True) for spec in self.specs["train"]],
                              batch_size=4, episodes_per_batch=2, sequence_length=4, seed=19)
        batch = replay.sample_training_batch()
        model.train()
        model.update(batch)  # Populate Adam moments, BN statistics, gradients and scheduler.
        baseline = copy.deepcopy(model)
        rng = tools.get_rng_state()
        expected_metrics = baseline.update(batch)
        expected_weights = copy.deepcopy(baseline.state_dict())
        expected_optimizer = copy.deepcopy(baseline.optimizer_state_dict())
        tools.set_rng_state(rng)
        evaluate_snapshot(config, model, self.cases[:1], horizons=(1, 5), policy_cases=1, policy_steps=2)
        assert_rng(self, rng)
        actual_metrics = model.update(batch)
        self.assertEqual(expected_metrics, actual_metrics)
        self.assertTrue(_equal(expected_weights, model.state_dict()))
        self.assertTrue(_equal(expected_optimizer, model.optimizer_state_dict()))

    def test_forecasts_only_and_failed_evaluation_restore_training_state(self):
        config = small_config("leworldmodel")
        model = load_model_family("leworldmodel").build_model(config).train()
        with patch("scripts.paper_faithful_followup_eval.policy_trial", side_effect=AssertionError("Must not act")):
            result = evaluate_snapshot(config, model, self.cases[:1], horizons=(1,), policy_steps=0)
        self.assertIsNone(result["policy"])
        self.assertIsNone(summarize_evaluation(result)["policy"]["return_mean"])
        before, rng = copy.deepcopy(model.state_dict()), tools.get_rng_state()
        optimizer = copy.deepcopy(model.optimizer_state_dict())
        def failure(*args, **kwargs):
            with torch.no_grad():
                next(model.parameters()).add_(1)
            model.optimizers["model"].param_groups[0]["lr"] = 17
            model.eval()
            torch.rand(4)
            raise ValueError("Injected evaluation failure")
        with patch("scripts.paper_faithful_followup_eval.score_branches", side_effect=failure):
            with self.assertRaisesRegex(ValueError, "Injected"):
                evaluate_snapshot(config, model, self.cases, horizons=(1,), policy_cases=0)
        self.assertTrue(model.training)
        self.assertTrue(_equal(before, model.state_dict()))
        self.assertTrue(_equal(optimizer, model.optimizer_state_dict()))
        assert_rng(self, rng)


if __name__ == "__main__":
    unittest.main()
