"""CPU regression checks for the checkpoint-based online smoke runner.

Run with: python -m scripts.check_online_checkpoint_smoke
"""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import tools
from dmc_expert.storage import (
    DATA_FORMAT,
    append_episode,
    dataset_identity,
    ensure_arrays,
)
from scripts.check_state_normalization import legacy_weights, tiny_config
from scripts.smoke_models import synthetic_batch
from scripts.smoke_online_checkpoints import accuracy_regression, diagnose, run_case, write_summary
from training import load_model_family
from training.protocol import checkpoint_compatibility, run_identity
from training.readout import online_readout
from training.trainer import online_update_target


class ToyEnvironment:
    env_num = 1

    def __init__(self):
        self.closed = False
        self.steps = 0

    def observation(self):
        return TensorDict({
            "image": torch.full((1, 64, 64, 3), self.steps, dtype=torch.uint8),
            "physical_state": torch.tensor([[.01 * self.steps, 1, 0, .01, 0]]),
            "is_terminal": torch.zeros(1, 1),
        }, batch_size=(1,))

    def reset(self):
        self.steps = 0
        return self.observation()

    def step(self, action):
        if action.shape != (1, 1) or not torch.isfinite(action).all():
            raise AssertionError("Policy must supply a finite action.")
        self.steps += 1
        return self.observation(), torch.ones(1, 1), torch.tensor([self.steps == 8])

    def reset_done(self, observation, done):
        return self.reset() if done.any() else observation

    def close(self):
        self.closed = True


def fixture(root, name):
    config = tiny_config(name, "cartpole_balance_sparse")
    config.env.time_limit = 256
    config.training.online.steps = 64
    config.training.online.updates = 8
    config.training.online.warmup_transitions = 8
    config.state_head.samples_per_update = 2
    path = root / str(config.scenario.dataset)
    path.mkdir()
    metadata = {
        "format": DATA_FORMAT, "domain_name": "cartpole", "task_name": "balance_sparse",
        "policy": "tdmpc2", "policy_mode": "actor", "action_repeat": 2,
        "action_min": [-1], "action_max": [1], "obs_dim": 5, "action_dim": 1,
        "observation_keys": ["position", "velocity"], "observation_shapes": {"position": [3], "velocity": [2]},
        "num_episodes": 3, "max_episode_steps": 128, "image_size": 64,
        "episode_splits": {"train": [0, 1], "heldout": [1, 3]},
        "goal_relation": {**OmegaConf.to_container(config.scenario.goal, resolve=True), "shape": [2]},
    }
    (path / "metadata.json").write_text(json.dumps(metadata))
    with h5py.File(path / "data.hdf5", "w") as h5:
        ensure_arrays(h5, metadata)
        for index in range(3):
            observations = np.zeros((129, 5), np.float32)
            observations[:, 0] = np.arange(129) / 100 + index / 100
            observations[:, 1] = 1
            truncation = np.zeros((128, 1), np.uint8)
            truncation[-1] = 1
            append_episode(h5, index, {
                "observations": observations, "images": np.full((129, 64, 64, 3), index, np.uint8),
                "actions": np.zeros((128, 1), np.float32), "rewards": np.ones((128, 1), np.float32),
                "discounts": np.ones((128, 1), np.float32), "terminations": np.zeros((128, 1), np.uint8),
                "truncations": truncation,
                "goal_relations": np.stack((observations[:, 0], np.zeros(129)), axis=-1).astype(np.float32),
            }, 128)
    family = load_model_family(name)
    model = family.build_model(config)
    model.state_head.set_stats([0, 1, 0, 0, 0], [.04, .001, .01, .15, .21])
    batch, _, _ = synthetic_batch(config, model)
    family.expert_update(model, batch)
    checkpoint = family.checkpoint(model)
    key = {"dreamer": "agent_state_dict", "storm": "world_model"}.get(name, "model_state_dict")
    checkpoint[key] = legacy_weights(checkpoint[key], "state_head.")
    checkpoint.update(
        phase="expert", expert_updates=1, checkpoint_id="test-expert", experiment_protocol=config.experiment_protocol,
        training_config=OmegaConf.to_container(config, resolve=True), run_identity=run_identity(config),
        compatibility={**checkpoint_compatibility(config), "recipe_version": 2},
        dataset_identity=dataset_identity(metadata), rng_state=tools.get_rng_state(),
    )
    source = root / "pretrained.pt"
    torch.save(checkpoint, source)
    return config, {
        "checkpoint": str(source), "result_path": str(root / "result.json"), "dataset_root": str(root),
        "model": name, "scenario": "cartpole_balance_sparse", "seed": 0, "device": "cpu",
        "env_steps": 32, "windows": 2, "batch_size": 1, "context_length": 3,
        "horizons": [1, 100], "window_seed": 2000000,
    }


class OnlineCheckpointSmokeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_schedule_matches_production_formula_without_compression(self):
        config = SimpleNamespace(
            env=SimpleNamespace(action_repeat=2),
            training=SimpleNamespace(online=SimpleNamespace(steps=80000, updates=10000, warmup_transitions=1024)),
        )
        for steps in (0, 32, 2048, 2080, 4096, 20000, 80000, 80032):
            eligible = max(0, min(steps // 2, 40000) - 1024)
            self.assertEqual(online_update_target(config, steps), 10000 * eligible // (40000 - 1024))
        self.assertEqual(online_update_target(config, 4096), 262)

    def test_diagnostics_restore_training_mode_and_rng(self):
        model = torch.nn.Linear(1, 1)
        initial = tools.get_rng_state()

        def diagnostic(*args):
            model.eval()
            torch.rand(2)
            np.random.rand(2)
            return {"test": True}

        with patch("scripts.smoke_online_checkpoints.analyze_checkpoint", side_effect=diagnostic):
            self.assertEqual(diagnose(model, None, None, None), {"test": True})
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.get_rng_state(), initial["torch"], rtol=0, atol=0)
        np.testing.assert_equal(np.random.get_state(), initial["numpy"])
        with (
            patch("scripts.smoke_online_checkpoints.analyze_checkpoint", side_effect=ValueError("test")),
            self.assertRaisesRegex(ValueError, "test"),
        ):
            diagnose(model, None, None, None)
        self.assertTrue(model.training)

    def test_nonfinite_diagnostics_fail_without_poisoning_the_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, job = fixture(root, "leworldmodel")
            with (
                patch("scripts.smoke_online_checkpoints.analyze_checkpoint", return_value={"metric": float("inf")}),
                patch("scripts.smoke_online_checkpoints.make_envs") as make_envs,
            ):
                result = run_case(job)
            self.assertEqual(result["status"], "FAIL")
            make_envs.assert_not_called()
            self.assertNotIn("before", result)
            self.assertEqual(json.loads((root / "result.json").read_text())["status"], "FAIL")
            result["log"] = "stdout.log"
            self.assertIn("FAIL", write_summary(root, [result], SimpleNamespace(**job)))

    def test_real_models_reuse_legacy_checkpoints_without_writing_weights(self):
        for name in ("leworldmodel", "temporal_straightening"):
            with self.subTest(family=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, job = fixture(root, name)
                source = Path(job["checkpoint"])
                checksum = hashlib.sha256(source.read_bytes()).hexdigest()
                environment = ToyEnvironment()
                with (
                    patch("scripts.smoke_online_checkpoints.make_envs", return_value=environment),
                    patch("torch.save", side_effect=AssertionError("Smoke must not save checkpoints")),
                    patch("training.planning.expert_update", side_effect=AssertionError("No expert updates")),
                ):
                    result = run_case(job)
                self.assertEqual(result["status"], "PASS", result.get("error"))
                self.assertTrue(environment.closed)
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), checksum)
                self.assertEqual(list(root.glob("*.pt")), [source])
                online = result["online"]
                self.assertEqual(online["updates"], online_update_target(config, 32))
                self.assertEqual(online["replay_rows"], 16)
                self.assertEqual(online["head_updates_before"], 1)
                self.assertEqual(online["head_updates_after"], 3)
                self.assertEqual(online["schedule_updates"], 8)
                self.assertEqual(online["schedule_env_steps"], 64)
                self.assertTrue(result["migration"]["head_optimizer_reset"])
                self.assertLess(result["migration"]["max_prediction_difference"], 1e-6)
                self.assertTrue(all(window["episode"] in (1, 2) for window in result["windows"]))
                self.assertEqual(result["migration"]["output_scale"], result["migration"]["previous_output_scale"])
                self.assertEqual(result["migration"]["loss_scale"], [1] * 5)
                self.assertEqual(online["last_metrics"]["state/expert_examples"], 1)
                before = result["before"]["all"]["physical"]["observed"]["1"]["rmse"]
                after = result["after"]["all"]["physical"]["observed"]["1"]["rmse"]
                self.assertNotEqual(before, after)
                result["log"] = "stdout.log"
                summary = write_summary(root, [result], SimpleNamespace(**job))
                self.assertIn("does NOT establish long-run", summary)
                report = json.loads((root / "report.json").read_text())
                self.assertFalse(report["checkpoint_writes"])
                self.assertFalse(report["evaluation_fitting"])

    def test_large_finite_accuracy_regression_is_not_an_execution_pass(self):
        def diagnostic(error):
            return {"all": {"physical": {"observed": {"1": {"rmse": {"position[0]": error}, "mean_normalized_mse": error**2}}}}}
        result = accuracy_regression(diagnostic(.002), diagnostic(.2))
        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_coordinates"], ["position[0]"])
        before, after = diagnostic(.002), diagnostic(.002)
        after["all"]["physical"]["observed"]["1"]["mean_normalized_mse"] = 10000
        self.assertFalse(accuracy_regression(before, after)["passed"])

    def test_readout_sampler_uses_only_train_split_and_resumes_exactly(self):
        for name in ("dreamer", "storm", "tdmpc2", "leworldmodel", "temporal_straightening"):
            with self.subTest(family=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, _ = fixture(root, name)
                config.training.expert.data_path = str(root / str(config.scenario.dataset))
                original = OmegaConf.to_container(config, resolve=True)
                family = load_model_family(name)
                model = family.build_model(config)
                with patch("dmc_expert.replay.DMCExpertDataset._state_stats", side_effect=AssertionError("No statistics rescan")):
                    with online_readout(config, family, model) as replay:
                        self.assertEqual(replay.episodes.tolist(), [0])
                        saved = copy.deepcopy(replay.state_dict())
                        expected = model.state_head._expert_source()
                        model.state_head.fit(*expected)
                        self.assertEqual(model.state_head.online_updates.item(), 1)
                    self.assertFalse(replay.h5.id.valid)
                    self.assertIsNone(model.state_head._expert_source)
                    with online_readout(config, family, model, {"phase": "online", "readout_replay_state": saved}):
                        self.assertEqual(model.state_head.online_updates.item(), 1)
                        actual = model.state_head._expert_source()
                        for left, right in zip(expected, actual):
                            torch.testing.assert_close(left, right, rtol=0, atol=0)
                self.assertEqual(OmegaConf.to_container(config, resolve=True), original)

    def test_online_or_incompatible_source_is_rejected_before_collection(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, job = fixture(Path(temporary), "leworldmodel")
            path = Path(job["checkpoint"])
            original = torch.load(path, weights_only=False)
            for failure in ("online", "unfinished", "wrong_scenario", "recipe", "warmup", "unaligned"):
                payload, case = copy.deepcopy(original), dict(job)
                if failure == "online":
                    payload["phase"] = "online"
                elif failure == "unfinished":
                    payload["expert_updates"] = 0
                elif failure == "wrong_scenario":
                    case["scenario"] = "reacher"
                elif failure == "recipe":
                    payload["compatibility"]["training_sha256"] = "different"
                elif failure == "warmup":
                    case["env_steps"] = 8
                else:
                    case["env_steps"] = 31
                torch.save(payload, path)
                with self.subTest(failure=failure), patch("scripts.smoke_online_checkpoints.make_envs") as make_envs:
                    result = run_case(case)
                    self.assertEqual(result["status"], "FAIL")
                    make_envs.assert_not_called()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
