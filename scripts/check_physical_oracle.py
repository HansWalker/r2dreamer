"""Simulator parity, action-derivative and CLI contracts for the physical oracle."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from envs.dmc import make_env
from models.planning import LatentPlanner
from scripts.diagnose_goal_objective import cart_state, set_cart_state
from scripts.diagnose_physical_controller import CartpoleCost, physical_labels
from scripts.diagnose_physical_oracle import (
    OraclePlanner, SimulatorFutures, SimulatorRollout, arguments, evaluate, integration_state, main,
)
from scripts.diagnose_planner_oracle import simulator_branch
from scripts.train_planner_check import build_config


class PhysicalOracleTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.config = build_config("leworldmodel", SimpleNamespace(
            scenario="cartpole_balance_sparse", device="cpu", seed=0, dataset_root=Path("/tmp/unused")))
        self.config.env.goal = None
        self.envs = []
        self.cases = []
        for index, state in enumerate(([.18, .08, .1, .15], [-.18, -.08, -.1, -.15])):
            env = make_env(self.config.env, 12000000 + index)
            self.stack.callback(env.close)
            env._observation = lambda time_step: {}
            env.reset()
            set_cart_state(env, state)
            for _ in range(2):
                env.step(np.zeros(1))
            self.envs.append(env)
            self.cases.append({"id": str(index), "seed": 12000000 + index, "cohort": "boundary",
                               "initial_state": state, "anchor_state": cart_state(env).tolist(), "anchor_success": True})
        self.oracle = SimulatorFutures(self.envs, self.stack, 1e-3)

    def test_rollouts_match_wrapped_environment_and_preserve_parents(self):
        rng = np.random.default_rng(6)
        states = [integration_state(env._env.physics) for env in self.envs]
        counts = [(env._episode_step, env._env._step_count) for env in self.envs]
        for horizon in (5, 50):
            with self.subTest(horizon=horizon):
                actions = rng.uniform(-1, 1, (2, 3, horizon, 1))
                actual = self.oracle.rollout(np.arange(2), actions)
                expected = np.empty_like(actual)
                for row, env in enumerate(self.envs):
                    for sample, sequence in enumerate(actions[row]):
                        with simulator_branch(env) as branch:
                            for step, action in enumerate(sequence):
                                branch.step(action)
                                expected[row, sample, step] = cart_state(branch)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
                np.testing.assert_array_equal(actual, self.oracle.rollout(np.arange(2), actions))
        for env, before, count in zip(self.envs, states, counts, strict=True):
            np.testing.assert_array_equal(integration_state(env._env.physics), before)
            self.assertEqual((env._episode_step, env._env._step_count), count)

    def test_repeated_chunk_indices_keep_independent_environments_and_candidates(self):
        actions = np.full((3, 2, 4, 1), .4)
        mixed = self.oracle.rollout(np.array([1, 0, 1]), actions)
        separate = self.oracle.rollout(np.arange(2), actions[:2])
        np.testing.assert_array_equal(mixed[0], separate[1])
        np.testing.assert_array_equal(mixed[1], separate[0])
        np.testing.assert_array_equal(mixed[2], separate[1])

    def test_simulator_state_jacobian_passes_gradcheck(self):
        actions = torch.tensor([[[[.2], [-.3], [.4]]], [[[-.2], [.3], [-.4]]]], dtype=torch.float64, requires_grad=True)
        indices = torch.arange(2)[:, None]
        self.assertTrue(torch.autograd.gradcheck(
            lambda value: SimulatorRollout.apply(value, indices, self.oracle), (actions,),
            eps=1e-4, atol=2e-5, rtol=2e-3))

    def test_cost_gradient_converges_and_respects_tanh_chain_and_saved_anchor(self):
        logits = torch.tensor([[[[.1], [-.2], [.4]]], [[[-.1], [.2], [-.4]]]], requires_grad=True)
        indices = torch.arange(2)[:, None]
        def cost(value):
            return CartpoleCost()(physical_labels(SimulatorRollout.apply(value.tanh(), indices, self.oracle))).sum()
        before = cost(logits)
        gradient = torch.autograd.grad(before, logits)[0]
        self.oracle.epsilon = 5e-4
        half = torch.autograd.grad(cost(logits), logits)[0]
        torch.testing.assert_close(gradient, half, rtol=2e-4, atol=2e-5)
        numerical = torch.empty_like(logits)
        for index in range(logits.numel()):
            plus, minus = logits.detach().clone(), logits.detach().clone()
            plus.view(-1)[index] += .002
            minus.view(-1)[index] -= .002
            numerical.view(-1)[index] = (cost(plus) - cost(minus)) / .004
        torch.testing.assert_close(gradient, numerical, rtol=2e-2, atol=3e-4)
        # An outstanding backward must use its own anchor, not the next real step.
        value = cost(logits)
        for env in self.envs:
            env.step(np.array([.7]))
        self.oracle.refresh()
        old_anchor_gradient = torch.autograd.grad(value, logits)[0]
        torch.testing.assert_close(half, old_anchor_gradient, rtol=0, atol=0)

    def test_native_solver_loops_are_reused_with_finite_bounded_warmstarts(self):
        self.assertIs(OraclePlanner._cem, LatentPlanner._cem)
        self.assertIs(OraclePlanner._gradient_plan, LatentPlanner._gradient_plan)
        for mode in ("cem", "gradient"):
            settings = copy.deepcopy(self.config.jepa_model.planner)
            settings.type, settings.samples, settings.elites = mode, 4, 2
            settings.horizon, settings.iterations, settings.gradient_batch_size = 3, 2, 3
            planner = OraclePlanner(settings, self.oracle, CartpoleCost(), "cpu")
            torch.manual_seed(42)
            first = planner.action(0)
            second = planner.action(1)
            self.assertEqual(first.shape, (2, 1))
            self.assertTrue(torch.isfinite(first).all() and torch.isfinite(second).all())
            self.assertTrue((first.abs() <= 1).all() and (second.abs() <= 1).all())
            cache = planner._cem_mean if mode == "cem" else planner._gradient_actions
            self.assertIsNotNone(cache)
            self.assertEqual(len(list(planner.parameters())), 0)

    def test_long_horizon_cost_derivatives_remain_consistent(self):
        for horizon in (25, 50):
            with self.subTest(horizon=horizon):
                logits = torch.zeros(1, 1, horizon, 1, requires_grad=True)
                def cost(value):
                    future = SimulatorRollout.apply(value.tanh(), torch.tensor([[0]]), self.oracle)
                    return CartpoleCost()(physical_labels(future)).sum()
                self.oracle.epsilon = 1e-3
                gradient = torch.autograd.grad(cost(logits), logits)[0]
                self.oracle.epsilon = 5e-4
                half = torch.autograd.grad(cost(logits), logits)[0]
                self.assertTrue(torch.isfinite(gradient).all())
                torch.testing.assert_close(gradient, half, rtol=2e-3, atol=2e-4)
                for step in (0, horizon // 2, horizon - 1):
                    plus, minus = logits.detach().clone(), logits.detach().clone()
                    plus[0, 0, step, 0] += .002
                    minus[0, 0, step, 0] -= .002
                    numerical = (cost(plus) - cost(minus)) / .004
                    torch.testing.assert_close(gradient[0, 0, step, 0], numerical, rtol=2e-2, atol=3e-3)

    def source_report(self, models=("leworldmodel",)):
        source = {"settings": {"seed": 0, "policy_steps": 2, "horizon": 5},
                  "physical_cost": vars(CartpoleCost()), "cases": self.cases, "runs": []}
        for name in models:
            config = copy.deepcopy(self.config)
            config.model_family = name
            config.jepa_model.planner.type = "cem" if name == "leworldmodel" else "gradient"
            config.jepa_model.planner.samples = 4
            config.jepa_model.planner.elites = 2
            config.jepa_model.planner.iterations = 2
            policy = {"return_mean": 3., "sustained_rate": 0.}
            source["runs"].append({"model": name, "status": "COMPLETE",
                                   "config": OmegaConf.to_container(config, resolve=True),
                                   "snapshots": [{"phase": "offline", "physical_controller": {"policy": policy}}]})
        return source

    def test_cli_only_runs_oracle_and_writes_consistent_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "source.json"
            contents = json.dumps(self.source_report())
            path.write_text(contents)
            output = root / "result"
            with redirect_stdout(io.StringIO()), patch("torch.load", side_effect=AssertionError("Checkpoint read")), \
                 patch("torch.save", side_effect=AssertionError("Checkpoint write")):
                status = main(["--source-report", str(path), "--models", "leworldmodel", "--device", "cpu", "--output", str(output)])
            self.assertEqual(status, 0)
            self.assertEqual(path.read_text(), contents)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["training_updates"], 0)
            run = report["runs"][0]
            self.assertEqual(run["status"], "COMPLETE")
            self.assertTrue(run["matched_policy_length"])
            self.assertEqual(report["settings"]["horizons"], [5])
            traces = [json.loads(line) for line in (output / run["trace_file"]).read_text().splitlines()]
            for index, case in enumerate(run["policy"]["cases"]):
                self.assertEqual(case["return"], sum(row["rewards"][index] for row in traces))
            self.assertTrue((output / "summary.txt").exists())
            self.assertFalse(list(output.rglob("*.pt")))

    def test_horizon_sweep_changes_only_horizon_and_resets_each_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.source_report(("leworldmodel", "temporal_straightening"))
            path = root / "source.json"
            contents = json.dumps(source)
            path.write_text(contents)
            base = ["--source-report", str(path), "--device", "cpu"]
            with redirect_stdout(io.StringIO()), patch("torch.load", side_effect=AssertionError("Checkpoint read")), \
                 patch("torch.save", side_effect=AssertionError("Checkpoint write")):
                self.assertEqual(main([*base, "--horizons", "3", "5", "--output", str(root / "sweep")]), 0)
                self.assertEqual(main([*base, "--horizons", "5", "--output", str(root / "single")]), 0)
            sweep = json.loads((root / "sweep/report.json").read_text())
            single = json.loads((root / "single/report.json").read_text())
            self.assertEqual([(r["model"], r["planner"]["horizon"]) for r in sweep["runs"]],
                             [(name, h) for name in ("leworldmodel", "temporal_straightening") for h in (3, 5)])
            self.assertEqual(sweep["physical_cost"], source["physical_cost"])
            self.assertEqual(path.read_text(), contents)
            for run in sweep["runs"]:
                expected = next(r["config"]["jepa_model"]["planner"] for r in source["runs"] if r["model"] == run["model"]).copy()
                expected["horizon"] = run["planner"]["horizon"]
                self.assertEqual(run["planner"], expected)
                self.assertEqual(run["source_horizon"], 5)
                self.assertEqual(run["status"], "COMPLETE")
                if expected["horizon"] == 5:
                    match = next(r for r in single["runs"] if r["model"] == run["model"])
                    a = [json.loads(line) for line in (root / "sweep" / run["trace_file"]).read_text().splitlines()]
                    b = [json.loads(line) for line in (root / "single" / match["trace_file"]).read_text().splitlines()]
                    for row in a + b:
                        row.pop("policy_seconds")
                    self.assertEqual(a, b)
            self.assertFalse(list(root.rglob("*.pt")))

    def test_sweep_preserves_results_and_continues_after_failed_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.source_report()
            path = root / "source.json"
            path.write_text(json.dumps(source))
            output = root / "result"
            def fail_first(config, *args):
                if config.jepa_model.planner.horizon == 3:
                    raise RuntimeError("Test failure")
                return evaluate(config, *args)
            with redirect_stdout(io.StringIO()), patch("scripts.diagnose_physical_oracle.evaluate", side_effect=fail_first):
                status = main(["--source-report", str(path), "--device", "cpu", "--models", "leworldmodel",
                               "--horizons", "3", "5", "--output", str(output)])
            self.assertEqual(status, 1)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual([r["status"] for r in report["runs"]], ["FAIL", "COMPLETE"])
            self.assertIn("Test failure", (output / "leworldmodel/horizon_3/error.log").read_text())
            self.assertTrue((output / report["runs"][1]["trace_file"]).is_file())

    def test_episode_boundary_checks_the_longest_requested_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.source_report()
            source["runs"][0]["config"]["env"]["time_limit"] = 100
            path = root / "source.json"
            path.write_text(json.dumps(source))
            output = root / "result"
            with self.assertRaisesRegex(ValueError, "episode boundary"):
                main(["--source-report", str(path), "--device", "cpu", "--models", "leworldmodel",
                      "--horizons", "5", "50", "--output", str(output)])
            self.assertFalse(output.exists())

    def test_invalid_cli_options_are_rejected(self):
        with patch("sys.stderr", new=io.StringIO()):
            for extra in (["--fd-epsilon", "0"], ["--fd-epsilon", "nan"], ["--policy-steps", "0"],
                          ["--models", "leworldmodel", "leworldmodel"], ["--horizons", "0"],
                          ["--horizons", "-5"], ["--horizons", "5", "5"]):
                with self.assertRaises(SystemExit):
                    arguments(["--source-report", "unused.json", *extra])


if __name__ == "__main__":
    unittest.main()
