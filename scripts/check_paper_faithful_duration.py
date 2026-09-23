"""Simulator isolation, exact resume, and six-fit CPU integration checks."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

import tools
from dmc_expert.storage import DATA_FORMAT, append_episode, dataset_identity, ensure_arrays
from envs.dmc import goal_relation, goal_relation_spec, make_env
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.paper_faithful_duration_support import (
    TaskBranchReplay, collect_bank, evaluate, restore_env, simulate, snapshot_env, validate_bank,
)
from scripts.paper_faithful_followup_eval import _equal
from scripts.paper_faithful_support import _digest
from scripts.train_paper_faithful_check import new_model, update
from scripts.train_paper_faithful_duration import (
    FORMAT, TASKS, MODELS, BudgetInfeasible, arguments, atomic_save, budget_estimate,
    branch_replay, build_config, choose_budget, file_hash, main, reuse_bank,
    restore_checkpoint, runtime_versions, save_checkpoint, source_hashes,
)
from training import load_model_family
from training.protocol import implementation_sha256


def fixture_config(model, task, args):
    config = build_config(model, task, args)
    config.expert_data.train_episodes = 2
    config.expert_data.heldout_episodes = 1
    config.jepa_model.planner.samples = 4
    config.jepa_model.planner.elites = 2
    config.jepa_model.planner.iterations = 1
    config.jepa_model.planner.horizon = 3
    config.state_head.samples_per_update = 4
    return config


def write_fixture(config):
    """Real task images/states; short complete episodes in production-format arrays."""
    path = Path(config.training.expert.data_path)
    path.mkdir(parents=True)
    env = make_env(config.env, 47, include_physical_state=True)
    try:
        env.reset()
        observations = env._env.task.get_observation(env._env.physics)
        domain, task = str(config.scenario.collection_task).split("/")
        dims = int(config.model_io.action.shape[0])
        shapes = {key: list(np.atleast_1d(value).shape) for key, value in observations.items()}
        metadata = {"format": DATA_FORMAT, "domain_name": domain, "task_name": task,
                    "policy": "tdmpc2", "policy_mode": "mpc", "action_repeat": 2,
                    "action_min": [-1] * dims, "action_max": [1] * dims,
                    "obs_dim": sum(np.asarray(v).size for v in observations.values()), "action_dim": dims,
                    "observation_keys": list(observations), "observation_shapes": shapes,
                    "num_episodes": 3, "max_episode_steps": 500, "image_size": 64,
                    "episode_splits": {"train": [0, 2], "heldout": [2, 3]},
                    "goal_relation": goal_relation_spec(env._env.physics, domain, task)}
        (path / "metadata.json").write_text(json.dumps(metadata))
        with h5py.File(path / "data.hdf5", "w") as h5:
            ensure_arrays(h5, metadata)
            for episode in range(3):
                obs = env.reset()
                frames, states = [obs["image"]], []
                relations = [goal_relation(env._env.physics, domain, task)]
                raw = lambda: np.concatenate([np.asarray(value).reshape(-1) for value in
                                               env._env.task.get_observation(env._env.physics).values()])
                states.append(raw())
                actions = np.random.default_rng(episode).uniform(-.4, .4, (12, dims)).astype(np.float32)
                rewards = []
                for action in actions:
                    obs, reward, _, _ = env.step(action)
                    frames.append(obs["image"])
                    states.append(raw())
                    relations.append(goal_relation(env._env.physics, domain, task))
                    rewards.append(reward)
                last = np.zeros((12, 1), np.uint8)
                last[-1] = 1
                append_episode(h5, episode, {"observations": np.asarray(states, np.float32),
                                             "images": np.stack(frames), "actions": actions,
                                             "rewards": np.asarray(rewards, np.float32)[:, None],
                                             "discounts": np.ones((12, 1), np.float32),
                                             "terminations": np.zeros((12, 1), np.uint8), "truncations": last,
                                             "goal_relations": np.asarray(relations, np.float32)},
                               float(np.sum(rewards)))
    finally:
        env.close()


class BudgetTests(unittest.TestCase):
    # A100 timings from paper_faithful_duration_20260923_002626. No ignored report
    # dependency: this regression must also run in a clean checkout.
    rates = dict(zip((f"{task}/{model}" for task in TASKS for model in MODELS),
                     ({"update_seconds": rate, "overhead_seconds": overhead} for rate, overhead in (
                         (.2139900905, 659.4088323), (.1886162360, 427.5546154),
                         (.5595676046, 1226.5173989), (.1862940176, 702.2119759),
                         (.6142636819, 2890.9772884), (.1929666520, 1611.6906522)))))

    def test_recorded_failure_and_realistic_retry_budget(self):
        args = arguments([])
        estimate = budget_estimate(args, self.rates, 1844.3905)
        self.assertEqual(estimate["permitted_updates"], 1544)
        self.assertFalse(estimate["fits_target"])
        self.assertAlmostEqual(estimate["required_total_seconds"] / 3600, 7.05106, places=4)
        self.assertAlmostEqual(estimate["required_total_with_margin_seconds"] / 3600, 8.15296, places=4)
        with self.assertRaisesRegex(BudgetInfeasible, "240-minute target.*1544"):
            choose_budget(args, self.rates, 1844.3905)
        args.minutes = 480
        self.assertGreaterEqual(choose_budget(args, self.rates, 300)["common_updates"], 8192)
        args.minutes, args.updates = 1, 8192
        self.assertFalse(budget_estimate(args, self.rates, 0)["fits_target"])
        self.assertEqual(choose_budget(args, self.rates, 0)["common_updates"], 8192)

    def test_saved_estimate_is_read_only_and_needs_no_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            report = {"format": FORMAT, "run_name": "failed_fixture", "settings": {
                "tasks": list(TASKS), "models": list(MODELS), "minutes": 240, "updates": None,
                "min_updates": 8192, "max_updates": 100000},
                "calibration": self.rates, "runs": [], "seconds": 1844.3905}
            path = source / "report.json"
            path.write_text(json.dumps(report))
            before = file_hash(path)
            with patch("torch.cuda.is_available", side_effect=AssertionError("GPU inspected")), \
                 patch("scripts.train_paper_faithful_duration.build_config", side_effect=AssertionError("Model configured")), \
                 redirect_stdout(io.StringIO()) as captured:
                self.assertEqual(main(["--estimate-from", folder, "--minutes", "480"]), 0)
            estimate = json.loads(captured.getvalue())
            self.assertTrue(estimate["future_work_excluding_preparation"]["fits_target"])
            self.assertFalse(estimate["including_recorded_preparation"]["fits_target"])
            self.assertEqual(file_hash(path), before)
            self.assertEqual(list(source.iterdir()), [path])


class DurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.args = arguments(["--dataset-root", str(cls.root), "--device", "cpu", "--profile", "tiny",
                              "--batch-size", "4", "--sources", "2"])
        cls.banks = {}
        for task in TASKS:
            config = fixture_config("leworldmodel", task, cls.args)
            write_fixture(config)
            cls.banks[task] = collect_bank(config, counts={"train": 2, "validation": 1, "test": 1},
                                          candidates=6, horizon=3, seed=cls.args.data_seed + TASKS.index(task) * 1_000_000)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def bank_source(self, name):
        folder = self.root / name
        folder.mkdir()
        report = {"format": FORMAT, "implementation_sha256": implementation_sha256(),
                  "source_hashes": source_hashes(), "versions": runtime_versions("cpu"),
                  "datasets": {}, "banks": {}}
        report["source_hashes"]["train_paper_faithful_duration.py"] = "older-budget-runner"
        for task, bank in self.banks.items():
            path = folder / f"{task}_bank.pt"
            atomic_save(bank, path)
            report["banks"][task] = {"file": path.name, "file_sha256": file_hash(path),
                                      "sha256": bank["sha256"], "metadata": bank["metadata"]}
            config = fixture_config("leworldmodel", task, self.args)
            metadata = json.loads((Path(config.training.expert.data_path) / "metadata.json").read_text())
            report["datasets"][task] = {"identity": dataset_identity(metadata)}
        (folder / "report.json").write_text(json.dumps(report))
        return folder, report

    def test_reused_banks_reject_incompatible_or_changed_data(self):
        folder, report = self.bank_source("bank_checks")
        args = copy.copy(self.args)
        args.train_anchors, args.validation_anchors, args.test_anchors = 2, 1, 1
        args.candidates, args.forecast_horizons = 6, [1, 3]
        task = "reacher"
        config = fixture_config("leworldmodel", task, args)
        identity = report["datasets"][task]["identity"]
        def reuse():
            return reuse_bank(folder, config, args, identity, runtime_versions("cpu"))
        self.assertEqual(reuse()["sha256"], self.banks[task]["sha256"])
        args.data_seed += 1
        with self.assertRaisesRegex(ValueError, "collection settings"):
            reuse()
        args.data_seed -= 1
        path = folder / "report.json"
        for key, value, message in (
            ("implementation_sha256", "changed", "implementation"),
            ("versions", {**report["versions"], "mujoco": "changed"}, "runtime"),
            ("datasets", {}, "identity"),
            ("source_hashes", {}, "helper"),
        ):
            path.write_text(json.dumps({**report, key: value}))
            with self.assertRaisesRegex(ValueError, message):
                reuse()
        path.write_text(json.dumps(report))
        with (folder / report["banks"][task]["file"]).open("ab") as handle:
            handle.write(b"damaged")
        with self.assertRaisesRegex(ValueError, "file changed"):
            reuse()

    def test_default_recipes_and_budget(self):
        args = arguments([])
        self.assertEqual((args.minutes, args.seed), (240, 1))
        for task, h in zip(TASKS, (5, 10, 25), strict=True):
            for name in MODELS:
                config = build_config(name, task, args)
                self.assertEqual(config.jepa_model.planner.horizon, h)
                self.assertEqual(config.jepa_model.planner.objective, "last" if name == "leworldmodel" else "ts_mpc")
                self.assertEqual(config.training.online.expert_fraction, 0.)
                if name == "temporal_straightening":
                    self.assertEqual(config.jepa_model.curvature_mode, "patch")
                    self.assertEqual(config.jepa_model.planner.aggregate_goal_weight, 0.)
        rates = {str(i): {"update_seconds": .17, "overhead_seconds": 100} for i in range(6)}
        budget = choose_budget(args, rates, 300)
        self.assertGreater(budget["common_updates"], 6822)
        self.assertLess(budget["estimated_remaining_seconds"] + 300, 240 * 60)
        with self.assertRaises(ValueError):
            choose_budget(args, rates, 240 * 60)
        with redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()):
            for bad in (["--minutes", "nan"], ["--sources", "3"], ["--models", "leworldmodel", "leworldmodel"],
                        ["--resume", "/tmp/run", "--updates", "5"]):
                with self.assertRaises(SystemExit):
                    arguments(bad)

    def test_snapshots_reproduce_all_tasks_and_isolate_parent(self):
        for task in TASKS:
            bank = self.banks[task]
            validate_bank(bank)
            config = fixture_config("leworldmodel", task, self.args)
            case = bank["splits"]["validation"][0]
            env = make_env(config.env, 993, include_physical_state=True)
            try:
                restore_env(env, case["snapshot"])
                before = _digest(snapshot_env(env))
                np.testing.assert_array_equal(env.render(), case["prefix"][-1])
                for candidate in (0, 1, 5):
                    result = simulate(env, case["action"][candidate].numpy())
                    for key in result:
                        torch.testing.assert_close(result[key], case[key][candidate], rtol=0, atol=0)
                self.assertEqual(before, _digest(snapshot_env(env)))
                if task == "reacher":
                    np.testing.assert_array_equal(env._env.physics.model.geom_pos, case["snapshot"]["geometry"]["geom_pos"])
                    np.testing.assert_array_equal(env._goal_image, case["goal_image"])
            finally:
                env.close()
            replay = branch_replay(bank, self.args)
            obs, action = replay.sample_training_batch()
            self.assertEqual(action.shape[-1], int(config.model_io.action.shape[0]))
            self.assertEqual(obs["physical_state"].shape[-1], 5 if task.startswith("cartpole") else 8)
            damaged = copy.deepcopy(bank)
            damaged["splits"]["test"][0]["seed"] = damaged["splits"]["train"][0]["seed"]
            with self.assertRaises(ValueError):
                validate_bank(damaged)

    def test_resume_and_evaluation_preserve_next_update_for_both_models(self):
        task = "reacher"
        bank = self.banks[task]
        for name in MODELS:
            config = fixture_config(name, task, self.args)
            config.training.expert.updates = 4
            with load_model_family(name).build_replay(config) as data:
                model = new_model(config, data)
                replay = branch_replay(bank, self.args)
                if hasattr(model, "configure_pretraining"):
                    model.configure_pretraining(4)
                update(model, data, replay, .5, 1, 1)
                row = {"config": OmegaConf.to_container(config, resolve=True), "updates": 1}
                path = self.root / f"{name}_resume.pt"
                identity = dataset_identity(data.metadata)
                save_checkpoint(path, model, data, replay, row, 4, identity, bank["sha256"])
                update(model, data, replay, .5, 2, 1)
                expected = copy.deepcopy(model.state_dict())
                expected_opt = copy.deepcopy(model.optimizer_state_dict())
                expected_sampler = _digest(data.state_dict())
                expected_branch = _digest(replay.state_dict())
            with load_model_family(name).build_replay(config) as data:
                resumed = new_model(config, data)
                replay = branch_replay(bank, self.args)
                restore_checkpoint(path, resumed, data, replay, 4, identity, bank["sha256"], config)
                before = tensor_digest(resumed.state_dict())
                rng = tools.get_rng_state()
                with patch.object(resumed.state_head, "forward", side_effect=AssertionError("Physical control input")):
                    result = evaluate(config, resumed, bank["splits"]["validation"], steps=2,
                                      policy_cases=1, seed=100, horizons=[1, 3])
                self.assertEqual(tensor_digest(resumed.state_dict()), before)
                self.assertTrue(_equal(rng["torch"], tools.get_rng_state()["torch"]))
                for probe in result["policy"]["probes"]:
                    self.assertEqual(tuple(probe["actions"].shape), (7, 3, 2))
                    np.testing.assert_array_equal(probe["actions"][0, 0], result["policy"]["traces"][probe["step"]]["actions"][0])
                update(resumed, data, replay, .5, 2, 1)
                self.assertTrue(_equal(expected, resumed.state_dict()))
                self.assertTrue(_equal(expected_opt, resumed.optimizer_state_dict()))
                self.assertEqual(expected_sampler, _digest(data.state_dict()))
                self.assertEqual(expected_branch, _digest(replay.state_dict()))
                with self.assertRaises(ValueError):
                    restore_checkpoint(path, resumed, data, replay, 5, identity, bank["sha256"], config)

    def test_six_fits_and_completed_resume(self):
        output = self.root / "run"
        source, source_report = self.bank_source("reused_banks")
        source_hash = file_hash(source / "report.json")
        argv = ["--dataset-root", str(self.root), "--output", str(output), "--device", "cpu", "--profile", "tiny",
                "--reuse-banks", str(source),
                "--batch-size", "4", "--sources", "2", "--updates", "2", "--calibration-updates", "1",
                "--train-anchors", "2", "--validation-anchors", "1", "--test-anchors", "1", "--candidates", "6",
                "--forecast-horizons", "1", "3", "--validation-policy-cases", "1", "--validation-policy-steps", "2",
                "--policy-cases", "1", "--policy-steps", "3", "--save-every", "1", "--skip-reference"]
        with patch("scripts.train_paper_faithful_duration.build_config", side_effect=fixture_config), \
             patch("scripts.train_paper_faithful_duration.collect_bank", side_effect=AssertionError("Recollected saved banks")), \
             patch("training.planning.OnlineSession", side_effect=AssertionError("Online training called")), \
             redirect_stdout(io.StringIO()) as captured:
            code = main(argv)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(code, 0, report.get("traceback"))
            self.assertIn("Run | run | status=COMPLETE", captured.getvalue())
            self.assertEqual(len(report["runs"]), 6)
            self.assertEqual(report["online_updates"], 0)
            self.assertEqual(file_hash(source / "report.json"), source_hash)
            self.assertEqual(report["bank_source"]["report_sha256"], source_hash)
            for task in TASKS:
                self.assertEqual(report["banks"][task]["sha256"], source_report["banks"][task]["sha256"])
            for row in report["runs"]:
                self.assertEqual([v["updates"] for v in row["validation"]], [0, 1, 2])
                self.assertEqual(row["test"]["result"]["policy"]["case_ids"], report["controls"][row["task"]]["test"]["case_ids"])
                payload = torch.load(output / row["key"] / "latest.pt", weights_only=False)
                self.assertTrue(payload["resume_supported"])
                self.assertIn("dataset_sampler", payload)
                self.assertTrue((output / row["key"] / "test_plans.pt").is_file())
            with patch("scripts.train_paper_faithful_duration.update", side_effect=AssertionError("Completed fit retrained")), \
                 patch("scripts.train_paper_faithful_duration.evaluate", side_effect=AssertionError("Completed fit reevaluated")):
                self.assertEqual(main(["--resume", str(output)]), 0)

    def test_budget_rejection_keeps_preparation_and_explains_status(self):
        source, _ = self.bank_source("infeasible_source")
        output = self.root / "infeasible"
        argv = ["--dataset-root", str(self.root), "--output", str(output), "--reuse-banks", str(source),
                "--device", "cpu", "--profile", "tiny", "--tasks", "reacher", "--models", "leworldmodel",
                "--minutes", "1", "--batch-size", "4", "--sources", "2", "--train-anchors", "2",
                "--validation-anchors", "1", "--test-anchors", "1", "--candidates", "6",
                "--forecast-horizons", "1", "3", "--validation-policy-cases", "1", "--validation-policy-steps", "1",
                "--policy-cases", "1", "--policy-steps", "1", "--skip-reference"]
        with patch("scripts.train_paper_faithful_duration.build_config", side_effect=fixture_config), \
             patch("scripts.train_paper_faithful_duration.calibrate", return_value={"update_seconds": .2, "overhead_seconds": 100}), \
             patch("scripts.train_paper_faithful_duration.fit", side_effect=AssertionError("Started an infeasible fit")), \
             redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(main(argv), 1)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["status"], "BUDGET_INFEASIBLE")
        self.assertEqual(report["runs"], [])
        self.assertIn("budget_estimate", report)
        self.assertIn("calibration", report)
        self.assertNotIn("budget", report)
        self.assertNotIn("traceback", report)
        self.assertTrue((output / "reacher_bank.pt").is_file())
        self.assertIn("status=BUDGET_INFEASIBLE", captured.getvalue())
        self.assertIn("training", (output / "summary.txt").read_text())

    def test_interrupted_fit_resumes_from_atomic_progress(self):
        output = self.root / "interrupted"
        argv = ["--dataset-root", str(self.root), "--output", str(output), "--device", "cpu", "--profile", "tiny",
                "--tasks", "reacher", "--models", "leworldmodel", "--batch-size", "4", "--sources", "2",
                "--updates", "2", "--calibration-updates", "1", "--train-anchors", "2",
                "--validation-anchors", "1", "--test-anchors", "1", "--candidates", "6",
                "--forecast-horizons", "1", "3", "--validation-policy-cases", "1", "--validation-policy-steps", "1",
                "--policy-cases", "1", "--policy-steps", "1", "--save-every", "1", "--skip-reference"]

        def interrupt(model, dataset, branch, fraction, step, seed):
            if step == 2 and (output / "reacher/leworldmodel/latest.pt").exists():
                raise KeyboardInterrupt()
            return update(model, dataset, branch, fraction, step, seed)

        with patch("scripts.train_paper_faithful_duration.build_config", side_effect=fixture_config), \
             redirect_stdout(io.StringIO()) as captured:
            with patch("scripts.train_paper_faithful_duration.update", side_effect=interrupt):
                self.assertEqual(main(argv), 1)
            self.assertIn("status=INTERRUPTED", captured.getvalue())
            payload = torch.load(output / "reacher/leworldmodel/latest.pt", weights_only=False)
            self.assertEqual(payload["updates"], 1)
            self.assertEqual(main(["--resume", str(output)]), 0)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["runs"][0]["updates"], 2)
            self.assertEqual([x["updates"] for x in report["runs"][0]["validation"]], [0, 1, 2])
            logs = [json.loads(line)["update"] for line in (output / "reacher/leworldmodel/metrics.jsonl").read_text().splitlines()]
            self.assertEqual(logs, [1, 2])


if __name__ == "__main__":
    unittest.main()
