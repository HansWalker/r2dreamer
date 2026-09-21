"""CPU/simulator checks for native learning curves and a non-compressed online continuation."""

import copy
import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

import tools
from scripts.check_online_checkpoint_smoke import ToyEnvironment
from scripts.check_planner_recipe import real_fixture
from scripts.check_state_normalization import tiny_config
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.diagnose_goal_objective import PROFILES, collect_objective_cases
from scripts.train_planner_learning_curve import (
    arguments, latent_errors, main, measure, validate_budget,
)
from scripts.train_planner_check import build_config, new_model
from training import load_model_family
from training.evaluation import StateDataset
from training.trainer import online_update_target


class LearningCurveTests(unittest.TestCase):
    def test_cli_and_default_recipe(self):
        args = arguments(["--dataset-root", "/tmp/data", "--device", "cpu"])
        self.assertEqual(args.eval_updates, [1000, 3000, 5000])
        self.assertEqual(args.online_eval_steps, [])
        for name, objective in (("leworldmodel", "last"), ("temporal_straightening", "ts_mpc")):
            config = build_config(name, args)
            validate_budget(config, args)
            self.assertEqual(config.jepa_model.planner.objective, objective)
            self.assertEqual(list(config.jepa_model.goal.alternatives), [])
            self.assertEqual(config.replay.batch_size, 128)
            self.assertEqual(config.training.online.steps, 80000)
            self.assertEqual(config.training.online.updates, 10000)
            self.assertEqual(online_update_target(config, args.online_steps), 262)
        with redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()):
            for extra in (["--eval-updates", "0"], ["--eval-updates", "2", "2"],
                          ["--online-steps", "-1"], ["--horizon", "0"], ["--policy-steps", "0"],
                          ["--online-eval-steps", "0"], ["--online-eval-steps", "2048", "2048"],
                          ["--online-eval-steps", "4096"], ["--online-eval-steps", "8192"],
                          ["--online-steps", "0", "--online-eval-steps", "32"],
                          ["--models", "leworldmodel", "leworldmodel"]):
                with self.assertRaises(SystemExit):
                    arguments(["--dataset-root", "/tmp/data", *extra])
        self.assertEqual(arguments(["--dataset-root", "/tmp/data", "--eval-updates", "5", "2"]).eval_updates, [2, 5])
        for steps in (1, 2048, 4097, 80032):
            args.online_steps = steps
            with self.assertRaisesRegex(ValueError, "Online steps"):
                validate_budget(config, args)
        args.online_steps = 0
        validate_budget(config, args)
        args.policy_steps = 498
        with self.assertRaisesRegex(ValueError, "one simulator episode"):
            validate_budget(config, args)

    def test_long_online_budget_keeps_the_production_schedule(self):
        args = arguments(["--dataset-root", "/tmp/data", "--device", "cpu", "--eval-updates", "5000",
                          "--online-steps", "32000", "--online-eval-steps", "16000", "4096"])
        self.assertEqual(args.online_eval_steps, [4096, 16000])
        for name in args.models:
            config = build_config(name, args)
            validate_budget(config, args)
            self.assertEqual(config.training.online.steps, 80000)
            self.assertEqual(config.training.online.updates, 10000)
            self.assertEqual([online_update_target(config, step) for step in [4096, 16000, 32000]],
                             [262, 1789, 3842])
            self.assertEqual(args.online_steps // (config.env.env_num * config.env.time_limit), 2)
        for points in ([2048], [4097]):
            args.online_eval_steps = points
            with self.assertRaisesRegex(ValueError, "Online evaluation steps"):
                validate_budget(config, args)

    def test_error_shapes_and_hold_baseline(self):
        for shape in ((4,), (2, 4)):
            truth = torch.ones(3, 5, *shape)
            prediction = truth * 3
            result = latent_errors(prediction, truth, torch.zeros(1, 1, *shape))
            self.assertEqual(result["mse"], [4.] * 5)
            self.assertEqual(result["hold_mse"], [1.] * 5)
            self.assertEqual(result["target_rms_std"], 0.)

    def test_measurement_preserves_training_rng_optimizers_buffers_and_planner_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_config = real_fixture(root)
            args = arguments(["--dataset-root", str(root), "--device", "cpu", "--policy-steps", "2", "--horizon", "3"])
            with patch("scripts.diagnose_goal_objective.PROFILES", dict(list(PROFILES.items())[:1])), redirect_stdout(io.StringIO()):
                cases = collect_objective_cases(fixture_config, SimpleNamespace(
                    sim_seeds=[12000000, 12000001], horizons=[1, 2, 3], candidates=16), history_size=3)
            for name in args.models:
                config = tiny_config(name, args.scenario)
                config.env.dataset_root = str(root)
                config.env.time_limit = 256
                config.jepa_model.planner.horizon = 3
                OmegaConf.resolve(config)
                with load_model_family(name).build_replay(config) as dataset:
                    model = new_model(config, dataset)
                    model.train()
                    model.update(dataset.sample_episode_batch())
                    heldout = StateDataset(dataset.h5, dataset.metadata, config.model_io,
                                           config.state_head.fields, model.state_head.targets)
                    windows = heldout.sample_windows(2, 6, 9, 3, 0, 1)
                    batch = heldout.read_batch(windows, 6)
                    model._cem_mean = torch.ones(1, 3, 1)
                    model._gradient_actions = torch.ones(1, 2, 3, 1)
                    caches = model._cem_mean, model._gradient_actions
                    before = tensor_digest(model.state_dict())
                    optimizer = copy.deepcopy(model.optimizer_state_dict())
                    sampler = copy.deepcopy(dataset.state_dict())
                    modes = [m.training for m in model.modules()]
                    rng = tools.get_rng_state()
                    output = root / name
                    output.mkdir()
                    with redirect_stdout(io.StringIO()):
                        scores = measure(config, model, cases, batch, args, output)
                    after = tools.get_rng_state()
                    self.assertEqual(before, tensor_digest(model.state_dict()))
                    torch.testing.assert_close(model.optimizer_state_dict(), optimizer, rtol=0, atol=0)
                    np.testing.assert_equal(dataset.state_dict(), sampler)
                    self.assertEqual(modes, [m.training for m in model.modules()])
                    self.assertIs(model._cem_mean, caches[0])
                    self.assertIs(model._gradient_actions, caches[1])
                    self.assertEqual(rng["python"], after["python"])
                    np.testing.assert_equal(rng["numpy"], after["numpy"])
                    torch.testing.assert_close(rng["torch"], after["torch"], rtol=0, atol=0)
                    self.assertEqual(len(scores["expert"]["latent"]["mse"]), 3)
                    self.assertEqual(set(scores["expert"]["physical"]["forecast"]), {"1", "3"})
                    self.assertTrue(all(window.episode not in dataset.episodes for window in windows))

    def test_from_scratch_offline_then_online_keeps_one_model_and_original_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_fixture(root)
            args = arguments(["--dataset-root", str(root), "--device", "cpu", "--eval-updates", "2", "4",
                              "--online-steps", "32", "--online-eval-steps", "22", "--policy-steps", "2",
                              "--horizon", "3", "--output", str(root / "output")])
            created, envs, online_initial = {}, [], {}

            def config_for(name, args):
                config = tiny_config(name, args.scenario)
                config.env.dataset_root = str(root)
                config.env.time_limit = 256
                config.state_head.samples_per_update = 2
                config.training.online.steps = 64
                config.training.online.updates = 8
                config.training.online.warmup_transitions = 8
                OmegaConf.resolve(config)
                return config

            def create(config, dataset):
                name = str(config.model_family)
                self.assertNotIn(name, created)
                model = new_model(config, dataset)
                created[name] = model
                if hasattr(model, "configure_pretraining"):
                    original = model.configure_pretraining
                    model.configure_pretraining = unittest.mock.Mock(wraps=original)
                return model

            def environment(config):
                name = next(reversed(created))
                model = created[name]
                online_initial[name] = {
                    "sha256": tensor_digest(model.state_dict()),
                }
                env = ToyEnvironment()
                envs.append(env)
                return env

            with patch("scripts.train_planner_learning_curve.arguments", return_value=args), \
                 patch("scripts.train_planner_learning_curve.build_config", side_effect=config_for), \
                 patch("scripts.train_planner_learning_curve.new_model", side_effect=create), \
                 patch("scripts.train_planner_learning_curve.make_envs", side_effect=environment), \
                 patch("scripts.diagnose_goal_objective.PROFILES", dict(list(PROFILES.items())[:1])), \
                 patch("torch.set_num_interop_threads"), patch("torch.save", side_effect=AssertionError("No checkpoints")), \
                 redirect_stdout(io.StringIO()):
                status = main()
            report = json.loads((args.output / "report.json").read_text())
            self.assertEqual(status, 0, [(r["model"], r.get("error")) for r in report["runs"]])
            self.assertTrue(all(env.closed for env in envs))
            for run in report["runs"]:
                model, name = created[run["model"]], run["model"]
                self.assertEqual(run["offline_updates"], 4)
                self.assertEqual([(s["phase"], s["updates"]) for s in run["snapshots"]],
                                 [("offline", 2), ("offline", 4), ("online", 1), ("online", 2)])
                self.assertEqual([s["env_steps"] for s in run["snapshots"]], [0, 0, 22, 32])
                self.assertEqual(online_initial[name]["sha256"], run["snapshots"][1]["model_sha256"])
                self.assertEqual(run["online"]["schedule_updates"], 8)
                self.assertEqual(run["online"]["schedule_env_steps"], 64)
                self.assertEqual(run["online"]["head_updates_before"], 4)
                self.assertEqual(run["online"]["head_updates_after"], 6)
                self.assertEqual(run["online"]["agent_transitions"], 16)
                for opt in model.optimizers.values():
                    self.assertTrue(all(int(state["step"]) == 6 for state in opt.state.values()))
                if name == "leworldmodel":
                    model.configure_pretraining.assert_called_once_with(4)
                    self.assertGreater(run["snapshots"][0]["learning_rates"]["model"][0], 0)
                    self.assertEqual(run["snapshots"][1]["learning_rates"]["model"][0], 0)
                    self.assertGreater(run["snapshots"][-1]["learning_rates"]["model"][0], 0)
                    self.assertEqual(model.scheduler.last_epoch, 2)
                else:
                    self.assertEqual(run["snapshots"][0]["learning_rates"], run["snapshots"][-1]["learning_rates"])
                rows = [json.loads(line) for line in (args.output / name / "offline_metrics.jsonl").read_text().splitlines()]
                self.assertEqual([row["update"] for row in rows], [1, 2, 3, 4])
                self.assertTrue(all(window["episode"] != 0 for window in run["expert_windows"]))
                self.assertTrue((args.output / name / "online_metrics.jsonl").exists())
            self.assertFalse(list(args.output.rglob("*.pt")))
            self.assertIn("COMPLETE means execution", (args.output / "summary.txt").read_text())
            self.assertIn("online_1[env=22]", (args.output / "summary.txt").read_text())

    def test_snapshot_rng_isolation_even_if_policy_fails(self):
        model = torch.nn.Linear(1, 1)
        before = tools.get_rng_state()
        def fail(*args):
            random.random()
            np.random.rand()
            torch.rand(3)
            raise RuntimeError("injected")
        with patch("scripts.train_planner_learning_curve.score_expert", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                measure(None, model, None, None, None, None)
        after = tools.get_rng_state()
        self.assertEqual(before["python"], after["python"])
        np.testing.assert_equal(before["numpy"], after["numpy"])
        torch.testing.assert_close(before["torch"], after["torch"], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
