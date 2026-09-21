"""CPU and real-simulator checks for the paired horizon diagnostic."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from envs.dmc import make_env
from models.shared.physical_state import readout_mode
from scripts.check_goal_objective import fake_case
from scripts.check_online_checkpoint_smoke import fixture
from scripts.check_state_normalization import tiny_config
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.diagnose_goal_objective import (
    cart_state,
    collect_objective_cases,
    observe_prefix,
    set_cart_state,
)
from scripts.diagnose_planner_oracle import encode_images, simulator_branch
from scripts.diagnose_planning_horizons import (
    arguments,
    evaluate_horizon,
    main,
    score_rankings,
    summarize_rankings,
)
from scripts.smoke_tiny_planners import FAMILIES
from training import load_model_family


class PlanningHorizonTest(unittest.TestCase):
    def test_cli_defaults_and_invalid_budgets(self):
        args = arguments(["--dataset-root", "/tmp/data"])
        self.assertEqual(args.horizons, [5, 15, 25])
        self.assertEqual((args.expert_updates, args.policy_steps), (1000, 100))
        with patch("sys.stderr", new=io.StringIO()):
            for extra in (("--horizons", "5", "5"), ("--policy-steps", "0"), ("--expert-updates", "0"),
                          ("--candidates", "12"), ("--seed", "-1"), ("--sim-seeds", "1", "1")):
                with self.assertRaises(SystemExit):
                    arguments(["--dataset-root", "/tmp/data", *extra])

    def test_common_cohort_does_not_change_across_horizons(self):
        def row(identifier, h, informative, value):
            selection = {key: {"normalized_return": value} for key in ("latent", "uniform", "reward_oracle")}
            score = {"reward_informative": informative, "selection": selection}
            return {"id": identifier, "horizon": h, "oracle": score, "forecast": score, "forecast_vs_oracle_rank": None}
        rows = [row("a", 5, True, .2), row("b", 5, False, 1.),
                row("a", 15, True, .3), row("b", 15, True, 1.)]
        result = summarize_rankings(rows, [5, 15])
        self.assertEqual(result["5"]["common_ids"], ["a"])
        self.assertEqual(result["15"]["common_ids"], ["a"])
        self.assertEqual(result["5"]["common_returns"]["oracle"], .2)
        self.assertEqual(result["15"]["common_returns"]["oracle"], .3)
        self.assertEqual(result["15"]["all_anchor_returns"]["oracle"], .65)
        self.assertEqual(result["15"]["coverage"], "LOW_CONTRAST")
        rows[2]["oracle"]["reward_informative"] = False
        result = summarize_rankings(rows, [5, 15])
        self.assertEqual(result["5"]["common_count"], 0)
        self.assertIsNone(result["5"]["common_returns"]["forecast"])
        self.assertIsNone(result["5"]["forecast_vs_oracle_rank"])

    def test_forecast_endpoint_indices_match_terminal_cost_without_head_or_model_updates(self):
        case = fake_case()
        case["image"] = case["image"].expand(-1, 2, -1, -1, -1).clone()
        case["states"] = np.zeros((3, 2, 4))
        case["prefix"] = case["goal_image"][None].expand(3, -1, -1, -1).clone()
        case["past_action"] = torch.zeros(2, 1)
        case["action"] = np.random.default_rng(4).uniform(-1, 1, (3, 5, 1)).astype(np.float32)
        for family in FAMILIES:
            with self.subTest(family=family):
                model = load_model_family(family).build_model(tiny_config(family, "cartpole_balance_sparse"))
                model.planner.objective = "last"
                before = tensor_digest(model.state_dict())
                modes = [m.training for m in model.modules()]
                with patch.object(model, "update", side_effect=AssertionError("No model update")), \
                     patch.object(model.state_head, "forward", side_effect=AssertionError("No physical head")), \
                     patch.object(model, "rollout", wraps=model.rollout) as rollout:
                    rows = score_rankings(model, [case], [1, 5], 2, 2)
                self.assertEqual(rollout.call_count, 2)  # Two candidate chunks, not one rollout per horizon.
                self.assertEqual(before, tensor_digest(model.state_dict()))
                self.assertEqual(modes, [m.training for m in model.modules()])
                with readout_mode(model):
                    latent = model.encode({"image": case["prefix"][None]})
                    goal = encode_images(model, case["goal_image"][None], 2)
                    for row in rows:
                        expected = model._goal_cost(latent, case["past_action"][None],
                                                    torch.from_numpy(case["action"])[None, :, :row["horizon"]], goal)[0]
                        torch.testing.assert_close(torch.tensor(row["forecast"]["latent_cost"]), expected, rtol=1e-5, atol=1e-5)
                changed = copy.deepcopy(case)
                changed["rewards"][:] = 0
                changed["states"][:] = 456
                changed["image"][:] = 0
                other = score_rankings(model, [changed], [1, 5], 2, 2)
                self.assertEqual([r["forecast"]["latent_cost"] for r in rows],
                                 [r["forecast"]["latent_cost"] for r in other])


class SimulatorPlanningHorizonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = tiny_config("leworldmodel", "cartpole_balance_sparse")
        cls.config.env.time_limit = 256
        cls.args = arguments(["--dataset-root", "/tmp", "--device", "cpu", "--horizons", "1", "3",
                              "--sim-seeds", "12000000", "--candidates", "13", "--policy-steps", "3"])
        with redirect_stdout(io.StringIO()), patch("scripts.diagnose_goal_objective.PROFILES", {
            "balanced": ("balanced", [0., 0., 0., 0.]), "pole_outside": ("failure", [.02, .12, 0., .2]),
        }):
            cls.cases = collect_objective_cases(cls.config, cls.args, history_size=3)

    def test_real_history_actions_and_branch_endpoints_are_aligned(self):
        for case in self.cases:
            env = make_env(self.config.env, case["seed"])
            try:
                env.reset()
                set_cart_state(env, case["initial_state"])
                prefix = observe_prefix(env, 3)
                np.testing.assert_array_equal(cart_state(env), case["anchor_state"])
                for key in ("prefix", "past_action"):
                    torch.testing.assert_close(prefix[key], case[key], rtol=0, atol=0)
                with simulator_branch(env) as branch:
                    for step, action in enumerate(case["action"][2]):
                        observation, reward, done, _ = branch.step(action)
                        self.assertFalse(done)
                        self.assertEqual(float(reward), case["rewards"][2, step])
                        if step + 1 in self.args.horizons:
                            index = self.args.horizons.index(step + 1)
                            np.testing.assert_array_equal(observation["image"], case["image"][2, index].numpy())
            finally:
                env.close()

    def test_native_policies_pair_starts_reset_caches_restore_settings_and_leave_weights_unchanged(self):
        for family in FAMILIES:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temporary:
                config = tiny_config(family, "cartpole_balance_sparse")
                config.env.time_limit = 256
                model = load_model_family(family).build_model(config)
                before = tensor_digest(model.state_dict())
                original_planner = model.planner
                model._cem_mean = old_cem = torch.tensor([99.])
                model._gradient_actions = old_gradient = torch.tensor([98.])
                modes = [m.training for m in model.modules()]
                torch.manual_seed(42)
                rng = torch.get_rng_state().clone()
                calls = []
                original_act = model.act
                def act(history, past, deterministic=False, first=None, *, calls=calls, model=model, original_act=original_act):
                    calls.append((history["image"].clone(), past.clone(), first.clone()))
                    self.assertEqual(set(history), {"image", "goal_image"})
                    if first.all():
                        self.assertIsNone(model._cem_mean)
                        self.assertIsNone(model._gradient_actions)
                    return original_act(history, past, deterministic=deterministic, first=first)
                with redirect_stdout(io.StringIO()), patch.object(model, "act", side_effect=act), \
                     patch.object(model, "update", side_effect=AssertionError("No online update")):
                    policies = [evaluate_horizon(config, model, self.cases, h, self.args, Path(temporary)) for h in (1, 3, 1)]
                self.assertEqual(before, tensor_digest(model.state_dict()))
                self.assertIs(model.planner, original_planner)
                self.assertIs(model._cem_mean, old_cem)
                self.assertIs(model._gradient_actions, old_gradient)
                self.assertEqual(modes, [m.training for m in model.modules()])
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                self.assertEqual(policies[0]["cases"], policies[2]["cases"])
                self.assertEqual(policies[0]["maximum_return"], 6)
                self.assertEqual(policies[0]["seconds_lookahead"], .02)
                for call in (calls[0], calls[3], calls[6]):
                    torch.testing.assert_close(call[0], torch.stack([c["prefix"] for c in self.cases]), rtol=0, atol=0)
                    torch.testing.assert_close(call[1], torch.stack([c["past_action"] for c in self.cases]), rtol=0, atol=0)
                    self.assertTrue(call[2].all())
                self.assertFalse(calls[1][2].any())

    def test_failure_closes_environments_and_restores_planner(self):
        model = load_model_family("leworldmodel").build_model(self.config)
        original = model.planner
        envs = []
        def tracked_env(*a, **kw):
            env = make_env(*a, **kw)
            env.close = unittest.mock.Mock(wraps=env.close)
            envs.append(env)
            return env
        with tempfile.TemporaryDirectory() as temporary, \
             patch("scripts.diagnose_planning_horizons.make_env", side_effect=tracked_env), \
             patch.object(model, "act", side_effect=RuntimeError("expected failure")), \
             self.assertRaisesRegex(RuntimeError, "expected failure"):
            evaluate_horizon(self.config, model, self.cases, 3, self.args, Path(temporary))
        self.assertIs(model.planner, original)
        self.assertTrue(all(e.close.call_count == 1 for e in envs))

    def test_end_to_end_trains_once_per_model_no_checkpoints_and_preserves_pairing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, "temporal_straightening")
            args = arguments(["--dataset-root", str(root), "--expert-updates", "2", "--device", "cpu",
                              "--horizons", "1", "3", "--policy-steps", "3", "--output", str(root / "output")])
            def config(name, args):
                result = tiny_config(name, args.scenario)
                result.env.dataset_root = str(root)
                result.env.time_limit = 256
                result.jepa_model.planner.horizon = 5
                return result
            with redirect_stdout(io.StringIO()), patch("scripts.diagnose_planning_horizons.arguments", return_value=args), \
                 patch("scripts.diagnose_planning_horizons.build_config", side_effect=config), \
                 patch("scripts.diagnose_planning_horizons.collect_objective_cases", return_value=self.cases) as collect, \
                 patch("torch.set_num_interop_threads"), \
                 patch("torch.save", side_effect=AssertionError("No checkpoint writes")), \
                 patch("torch.load", side_effect=AssertionError("No checkpoint reads")):
                status = main()
            self.assertEqual(status, 0, (args.output / "summary.txt").read_text())
            self.assertEqual(collect.call_count, 1)
            report = json.loads((args.output / "report.json").read_text())
            self.assertEqual(len(report["runs"]), 2)
            for run in report["runs"]:
                self.assertEqual(run["status"], "COMPLETE")
                metrics = (args.output / run["model"] / "offline_metrics.jsonl").read_text().splitlines()
                self.assertEqual(len(metrics), 2)
                for arm in run["arms"]:
                    self.assertEqual(arm["status"], "COMPLETE")
                    self.assertEqual(arm["policy"]["initial_state_sha256"], run["offline"]["state_sha256"])
                    self.assertEqual(arm["policy"]["initial_state_sha256"], arm["policy"]["final_state_sha256"])
                    log = args.output / run["model"] / f"horizon_{arm['horizon']}" / "policy_metrics.jsonl"
                    self.assertEqual(len(log.read_text().splitlines()), 3)
                self.assertEqual(run["config"]["jepa_model"]["planner"]["horizon"], 5)
            self.assertFalse(list(args.output.rglob("*.pt")))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
