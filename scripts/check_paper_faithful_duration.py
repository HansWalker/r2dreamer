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
    TASKS, MODELS, arguments, branch_replay, build_config, choose_budget, main,
    restore_checkpoint, save_checkpoint,
)
from training import load_model_family


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
                                          candidates=6, horizon=3, seed=80_000_000 + TASKS.index(task) * 100)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

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
        argv = ["--dataset-root", str(self.root), "--output", str(output), "--device", "cpu", "--profile", "tiny",
                "--batch-size", "4", "--sources", "2", "--updates", "2", "--calibration-updates", "1",
                "--train-anchors", "2", "--validation-anchors", "1", "--test-anchors", "1", "--candidates", "6",
                "--forecast-horizons", "1", "3", "--validation-policy-cases", "1", "--validation-policy-steps", "2",
                "--policy-cases", "1", "--policy-steps", "3", "--save-every", "1", "--skip-reference"]
        with patch("scripts.train_paper_faithful_duration.build_config", side_effect=fixture_config), \
             patch("training.planning.OnlineSession", side_effect=AssertionError("Online training called")), \
             redirect_stdout(io.StringIO()) as captured:
            code = main(argv)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(code, 0, report.get("traceback"))
            self.assertIn("Run | run | status=COMPLETE", captured.getvalue())
            self.assertEqual(len(report["runs"]), 6)
            self.assertEqual(report["online_updates"], 0)
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
