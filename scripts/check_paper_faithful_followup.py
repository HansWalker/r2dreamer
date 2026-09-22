"""Follow-up schedule contracts and all-arm, two-seed simulator integration."""

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
from scripts.train_paper_faithful_check import ARMS as ORIGINAL_ARMS, common_initial_weights, load_snapshot, new_model
from scripts.train_paper_faithful_followup import (
    ARMS, arguments, build_followup_config, estimated_fit_seconds, main, milestone_updates,
)


class FollowupRunnerTests(unittest.TestCase):
    def test_full_models_keep_native_recipes_and_paired_initialization(self):
        args = arguments(["--device", "cpu"])
        self.assertEqual(args.seeds, [1, 2])
        self.assertEqual(args.forecast_horizons, [1, 5, 15, 25])
        self.assertEqual(len(ORIGINAL_ARMS), 6)
        hashes = {}
        for seed in args.seeds:
            for arm in ARMS:
                config = build_followup_config(arm, args, seed)
                self.assertEqual(config.seed, seed)
                self.assertEqual(config.jepa_model.planner.horizon, 5)
                self.assertEqual(config.replay.batch_size, 128)
                self.assertEqual(config.training.online.updates, 10000)
                self.assertEqual(config.training.online.expert_fraction, 0.)
                self.assertEqual(config.jepa_model.planner.objective, "last" if arm == "lewm_coverage" else "ts_mpc")
                self.assertEqual(config.jepa_model.planner.iterations, 30 if arm == "lewm_coverage" else 32)
                self.assertEqual(config.jepa_model.planner.samples, 300 if arm == "lewm_coverage" else 16)
                model = new_model(config)
                if arm.startswith("ts_"):
                    digest = tensor_digest(common_initial_weights(model))
                    if seed in hashes:
                        self.assertEqual(hashes[seed], digest)
                    hashes[seed] = digest
                del model
        self.assertNotEqual(hashes[1], hashes[2])

    def test_budget_separates_setup_and_counts_paired_validation_measurements(self):
        validation = {"setup_seconds": 3., "acting_seconds": 6., "steps": 3}
        test = {"setup_seconds": 4., "acting_seconds": 9., "steps": 3}
        estimate = estimated_fit_seconds(10., 2., 9, 9, validation, test, [.5], 25, 100)
        self.assertEqual(estimate, 443.)
        self.assertEqual(estimated_fit_seconds(10., 2., 9, 9, validation, test, [.5], 25, 110) - estimate, 30.)
        self.assertEqual(estimated_fit_seconds(10., 2., 9, 9, validation, test, [.25, .5], 25, 100) - estimate, 55.)
        self.assertEqual(milestone_updates(2, [.25, .5, .75]), [0, 1, 2])
        self.assertEqual(milestone_updates(1, [.5]), [0, 1])
        self.assertEqual(milestone_updates(100, [.5]), [0, 50, 100])

    def test_real_two_seed_comparison_and_intermediate_native_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = real_fixture(root)
            metadata_path = root / fixture.scenario.dataset / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["episode_splits"] = {"train": [0, 2], "heldout": [2, 3]}
            metadata_path.write_text(json.dumps(metadata))
            original = build_followup_config

            def fixture_config(arm, args, seed):
                config = original(arm, args, seed)
                config.env.time_limit = 256
                config.expert_data.train_episodes = 2
                config.expert_data.heldout_episodes = 1
                config.expert_data.policy_mode = "actor"
                config.jepa_model.planner.samples = 4
                config.jepa_model.planner.elites = 2
                config.jepa_model.planner.iterations = 1
                return config

            argv = ["--dataset-root", str(root), "--output", str(root / "run"), "--device", "cpu", "--profile", "tiny",
                    "--batch-size", "4", "--sources", "2", "--updates", "2", "--calibration-updates", "1",
                    "--train-anchors", "6", "--validation-anchors", "6", "--test-anchors", "6", "--candidates", "6",
                    "--forecast-horizons", "1", "3", "--planner-horizon", "3", "--validation-policy-cases", "1",
                    "--validation-policy-steps", "1", "--policy-cases", "1", "--policy-steps", "2", "--skip-reference"]
            with patch("scripts.train_paper_faithful_followup.build_followup_config", side_effect=fixture_config), \
                 patch("training.planning.OnlineSession", side_effect=AssertionError("Online training called")), \
                 redirect_stdout(io.StringIO()):
                code = main(argv)
            report = json.loads((root / "run/report.json").read_text())
            self.assertEqual(code, 0, report.get("traceback"))
            self.assertEqual(report["status"], "COMPLETE")
            self.assertEqual(report["online_updates"], 0)
            self.assertFalse(report["online_schedule_changed"])
            self.assertEqual(report["reference"]["status"], "NOT_RUN")
            self.assertEqual(len(report["runs"]), 6)
            self.assertEqual(report["budget"]["milestones"], [0, 1, 2])
            for row in report["runs"]:
                self.assertEqual(row["updates"], 2)
                self.assertEqual(row["coverage_fraction"], .5)
                self.assertEqual([x["updates"] for x in row["validation"]], [0, 1, 2])
                self.assertEqual([x["updates"] for x in row["checkpoints"]], [1, 2])
                self.assertIsNone(row["validation"][0]["result"]["policy"])
                for measurement in row["validation"][1:]:
                    policy = measurement["result"]["policy"]
                    self.assertEqual(policy["steps"], 1)
                    self.assertEqual(policy["case_ids"], report["controls"]["validation"]["zero"]["case_ids"])
                    self.assertTrue(measurement["result"]["training_state_preserved"])
                self.assertEqual(row["test"]["result"]["policy"]["case_ids"], report["controls"]["test"]["zero"]["case_ids"])
                self.assertEqual(row["test"]["result"]["policy"]["steps"], 2)
                for checkpoint in row["checkpoints"]:
                    path = root / "run" / checkpoint["file"]
                    config, loaded, identity = load_snapshot(path, "cpu")
                    self.assertEqual(config.training.expert.updates, 2)
                    self.assertEqual(identity["sha256"], checkpoint["sha256"])
                    self.assertFalse(identity["training_resume"])
                    del loaded


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
