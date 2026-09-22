"""Runner contracts, real-simulator end-to-end fitting, and checkpoint evaluation."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.check_planner_recipe import real_fixture
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.train_paper_faithful_check import (
    ARMS, arguments, build_config, choose_updates, common_initial_weights,
    load_snapshot, main, mixed_batch, new_model, save_checkpoint,
)


class PaperFaithfulRunnerTests(unittest.TestCase):
    def test_full_recipe_preserves_native_objectives_and_online_settings(self):
        args = arguments(["--device", "cpu"])
        hashes = {}
        for arm in ARMS:
            config = build_config(arm, args)
            self.assertEqual(config.training.online.updates, 10000)
            self.assertEqual(config.training.online.steps, 80000)
            self.assertEqual(config.training.online.expert_fraction, 0.)
            self.assertEqual(config.jepa_model.planner.objective,
                             "last" if config.model_family == "leworldmodel" else "ts_mpc")
            self.assertEqual(config.replay.batch_size, 128)
            self.assertEqual(config.jepa_model.history_size, 3)
            model = new_model(config)
            digest = tensor_digest(common_initial_weights(model))
            if config.model_family in hashes:
                self.assertEqual(hashes[config.model_family], digest)
            hashes[config.model_family] = digest
            del model

    def test_runtime_budget_is_matched_and_rejects_an_inadequate_run(self):
        count = choose_updates(3600, [.1, .3, .2], 600, 100, 10000)
        self.assertGreaterEqual(count, 100)
        self.assertLessEqual(count * .6 * 1.2 + 600 * 1.25 + 60, 3600)
        with self.assertRaisesRegex(ValueError, "No arms were trained"):
            choose_updates(60, [.1, .3], 60, 100, 10000)

    def test_coverage_replaces_rows_and_keeps_labels_separate(self):
        expert = ({"image": torch.zeros(8, 4, 2, 2, 3), "physical_state": torch.zeros(8, 4, 5)},
                  torch.zeros(8, 3, 1), torch.zeros(8, 3, 1))
        branch = ({"image": torch.ones(4, 4, 2, 2, 3), "physical_state": torch.ones(4, 4, 5)},
                  torch.ones(4, 3, 1))
        obs, action = mixed_batch(expert, branch, .5)
        self.assertEqual(len(action), 8)
        self.assertEqual(set(obs), {"image", "physical_state"})
        self.assertTrue(torch.equal(action[:4], expert[1][:4]))
        self.assertTrue(torch.equal(action[4:], branch[1]))
        self.assertIs(mixed_batch(expert, None, 0), expert)
        with self.assertRaises(ValueError):
            mixed_batch(expert, ({"image": branch[0]["image"]}, branch[1]), .5)

    def test_checkpoint_roundtrip_retains_architecture_and_never_resumes_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(["--device", "cpu", "--profile", "tiny"])
            config = build_config("ts_agg_01", args)
            model = new_model(config)
            source = Path(temporary) / "model.pt"
            digest = save_checkpoint(source, model, config, 7, {"test": True})
            with patch.object(type(model), "load_optimizer_state_dict", side_effect=AssertionError("Resumed")):
                loaded_config, loaded, metadata = load_snapshot(source, "cpu")
            self.assertEqual(metadata["sha256"], digest)
            self.assertFalse(metadata["training_resume"])
            self.assertEqual(loaded_config.jepa_model.curvature_mode, "agg")
            self.assertEqual(tensor_digest(model.state_dict()), tensor_digest(loaded.state_dict()))
            payload = torch.load(source, weights_only=False)
            payload["format"] = "planner_learning_curve_v1"
            payload["training_config"]["jepa_model"]["planner"]["objective"] = "last"
            torch.save(payload, source)
            _, _, metadata = load_snapshot(source, "cpu")
            self.assertEqual(metadata["saved_objective"], "last")
            self.assertEqual(metadata["evaluated_objective"], "ts_mpc")

    def test_real_simulator_offline_run_all_arms_and_evaluation_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = real_fixture(root)
            metadata_path = root / fixture.scenario.dataset / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["episode_splits"] = {"train": [0, 2], "heldout": [2, 3]}
            metadata_path.write_text(json.dumps(metadata))
            original_builder = build_config

            def fixture_builder(arm, args):
                config = original_builder(arm, args)
                config.env.time_limit = 256
                config.expert_data.train_episodes = 2
                config.expert_data.heldout_episodes = 1
                config.expert_data.policy_mode = "actor"
                # This test checks runner wiring; the separate default test guards production budgets.
                config.jepa_model.planner.samples = 4
                config.jepa_model.planner.elites = 2
                config.jepa_model.planner.iterations = 1
                return config

            common = ["--dataset-root", str(root), "--device", "cpu", "--profile", "tiny",
                      "--batch-size", "4", "--sources", "2", "--updates", "2", "--calibration-updates", "1",
                      "--train-anchors", "2", "--validation-anchors", "1", "--test-anchors", "1",
                      "--candidates", "6", "--horizon", "3", "--policy-cases", "1", "--policy-steps", "2",
                      "--skip-reference"]
            with patch("scripts.train_paper_faithful_check.build_config", side_effect=fixture_builder), \
                 patch("training.planning.OnlineSession", side_effect=AssertionError("Online collection")), \
                 redirect_stdout(io.StringIO()):
                result = main([*common, "--output", str(root / "output")])
            report = json.loads((root / "output/report.json").read_text())
            self.assertEqual(result, 0, report.get("traceback"))
            self.assertEqual(report["status"], "COMPLETE")
            self.assertEqual(report["online_updates"], 0)
            self.assertFalse(report["online_schedule_changed"])
            self.assertEqual(report["reference"]["status"], "NOT_RUN")
            self.assertEqual([r["updates"] for r in report["runs"]], [2] * len(ARMS))
            for row in report["runs"]:
                self.assertEqual(row["config"]["training"]["expert"]["updates"], 2)
                self.assertEqual(row["validation"]["objective"], "last" if row["family"] == "leworldmodel" else "ts_mpc")
                self.assertEqual(row["policy"]["steps"], 2)
                self.assertEqual(row["status"], "COMPLETE")
                self.assertTrue((root / "output" / row["arm"] / "native.pt").is_file())
            checkpoint = root / "output/ts_agg_01/native.pt"
            with patch("scripts.train_paper_faithful_check.build_config", side_effect=fixture_builder), \
                 patch("scripts.train_paper_faithful_check.update", side_effect=AssertionError("Evaluation fitted")), \
                 redirect_stdout(io.StringIO()):
                result = main([*common, "--checkpoint", str(checkpoint), "--output", str(root / "evaluate")])
            evaluation = json.loads((root / "evaluate/report.json").read_text())
            self.assertEqual(result, 0, evaluation.get("traceback"))
            self.assertEqual(len(evaluation["runs"]), 1)
            self.assertEqual(evaluation["runs"][0]["checkpoint"]["format"], "paper_faithful_offline_v1")
            self.assertIn("branch scoring only", evaluation["scope"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
