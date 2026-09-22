"""Feedback dynamics, sequence extraction and source-report diagnostic contracts."""

import copy
import hashlib
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
from scripts.diagnose_controller_reference import main, selected_sequence, simulate_sequence
from scripts.diagnose_goal_objective import cart_state, feedback_gain, set_cart_state
from scripts.diagnose_physical_controller import CartpoleCost, physical_labels
from scripts.diagnose_physical_oracle import integration_state
from scripts.train_planner_check import build_config


class ControllerReferenceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.config = build_config("leworldmodel", SimpleNamespace(
            scenario="cartpole_balance_sparse", device="cpu", seed=0, dataset_root=Path("/tmp/unused")))
        self.config.env.goal = None
        self.env = make_env(self.config.env, 12000000)
        self.stack.callback(self.env.close)
        self.env._observation = lambda time_step: {}
        self.env.reset()
        state = [.18, .08, 0., .15]
        set_cart_state(self.env, state)
        for _ in range(int(self.config.jepa_model.history_size) - 1):
            self.env.step(np.zeros(1, dtype=np.float32))
        self.cases = [{"id": "12000000/boundary", "seed": 12000000, "cohort": "boundary",
                       "initial_state": state, "anchor_state": cart_state(self.env).tolist(), "anchor_success": True}]

    def test_feedback_recovers_boundary_and_replay_preserves_true_sequence(self):
        gain = feedback_gain(self.env)
        before = integration_state(self.env._env.physics)
        feedback = simulate_sequence(self.env, 100, CartpoleCost(), gain=gain)
        self.assertTrue(all(feedback["success"][-10:]))
        replay = simulate_sequence(self.env, 100, CartpoleCost(), actions=feedback["actions"])
        self.assertEqual(feedback, replay)
        np.testing.assert_array_equal(integration_state(self.env._env.physics), before)
        expected = CartpoleCost()(physical_labels(torch.tensor(feedback["states"], dtype=torch.float64))).item()
        self.assertEqual(feedback["cost"], expected)

    def test_extracts_executed_cem_mean_and_lowest_cost_gradient_restart(self):
        cem = SimpleNamespace(planner=SimpleNamespace(type="cem"), _cem_mean=torch.tensor([[[.2], [.3]]]))
        np.testing.assert_array_equal(selected_sequence(cem, cem._cem_mean[:, 0]), cem._cem_mean.numpy())
        actions = torch.tensor([[[[.4], [.2]], [[-.7], [.1]]], [[[.5], [.2]], [[-.8], [.3]]]])
        gradient = SimpleNamespace(planner=SimpleNamespace(type="gradient"), device=torch.device("cpu"),
                                   _gradient_actions=actions,
                                   _goal_cost=lambda *args: torch.tensor([[2., 1.], [0., 3.]]))
        selected = torch.stack((actions[0, 1], actions[1, 0]))
        np.testing.assert_array_equal(selected_sequence(gradient, selected[:, 0]), selected.numpy())
        with self.assertRaises(AssertionError):
            selected_sequence(gradient, torch.zeros(2, 1))

    def test_cli_runs_both_solvers_without_checkpoints_and_records_reproducible_costs(self):
        source = {"settings": {"seed": 0, "policy_steps": 3, "horizon": 5},
                  "physical_cost": vars(CartpoleCost()), "cases": self.cases, "runs": []}
        for name, mode in (("leworldmodel", "cem"), ("temporal_straightening", "gradient")):
            config = copy.deepcopy(self.config)
            config.model_family = name
            config.jepa_model.planner.type = mode
            config.jepa_model.planner.samples, config.jepa_model.planner.elites = 4, 2
            config.jepa_model.planner.iterations = 2
            source["runs"].append({"model": name, "status": "COMPLETE", "config": OmegaConf.to_container(config, resolve=True)})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, output = root / "source.json", root / "output"
            contents = json.dumps(source)
            source_path.write_text(contents)
            with redirect_stdout(io.StringIO()), patch("torch.load", side_effect=AssertionError("checkpoint read")), \
                 patch("torch.save", side_effect=AssertionError("checkpoint write")), \
                 patch("envs.dmc.DeepMindControl.render", side_effect=AssertionError("image rendering")):
                self.assertEqual(main(["--source-report", str(source_path), "--output", str(output),
                                       "--horizons", "3", "5"]), 0)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["source_sha256"], hashlib.sha256(contents.encode()).hexdigest())
            self.assertEqual(source_path.read_text(), contents)
            self.assertEqual(report["training_updates"], 0)
            self.assertFalse(report["checkpoint_reads"] or report["checkpoint_writes"])
            self.assertEqual(len(report["runs"]), 4)
            traces = [json.loads(line) for line in (output / "feedback_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(report["feedback"]["cases"][0]["return"], sum(row["rewards"][0] for row in traces))
            for run in report["runs"]:
                case = run["comparison"]["cases"][0]
                for kind in ("feedback", "solver"):
                    result = case[kind]
                    direct = simulate_sequence(self.env, run["horizon"], CartpoleCost(), actions=result["actions"])
                    self.assertEqual(result, direct)
                self.assertEqual(case["solver_minus_feedback_cost"], case["solver"]["cost"] - case["feedback"]["cost"])
            self.assertTrue(report["source_hashes"])


if __name__ == "__main__":
    unittest.main()
