"""CPU regression checks for the from-scratch hour-scale planner diagnostic."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.check_online_checkpoint_smoke import ToyEnvironment, fixture
from scripts.train_planner_check import (
    FAMILIES, build_config, calibrate, choose_budget, diagnostic_settings, new_model, run_training, set_budget, write_report,
)
from training import load_model_family
from training.evaluation import StateDataset
from training.trainer import online_update_target


class EvaluationToyEnvironment(ToyEnvironment):
    def step(self, action, reset_mask=None):
        if reset_mask is not None and bool(reset_mask.all()):
            return self.reset(), torch.zeros(1, 1), torch.zeros(1, dtype=torch.bool)
        return super().step(action)


class PlannerTrainingTest(unittest.TestCase):
    def args(self):
        return SimpleNamespace(dataset_root=Path("/tmp/expert"), scenario="cartpole_balance_sparse", device="cpu", seed=0)

    def test_gpu_profile_is_small_but_uses_full_episodes_and_native_losses(self):
        for name in FAMILIES:
            with self.subTest(name=name):
                config = build_config(name, self.args())
                model = new_model(config, SimpleNamespace(state_mean=torch.zeros(5), state_std=torch.ones(5)))
                self.assertLess(sum(p.numel() for p in model.parameters()), 500_000)
                self.assertEqual(int(model.state_head.updates), 0)
                self.assertEqual(int(config.env.time_limit), 1000)
                self.assertEqual(int(config.replay.batch_size), 128)
                self.assertEqual(int(config.replay.episodes_per_batch), 16)
                self.assertEqual(int(config.state_head.samples_per_update), 256)
                self.assertTrue(config.jepa_model.use_amp)
                self.assertEqual(config.jepa_model.optim.encoder_lr, 1e-5)
                self.assertEqual(config.training.online.expert_fraction, 0)
                self.assertEqual(config.state_head.online.expert_fraction, .5)

    def test_common_budget_accounts_for_both_models_and_respects_target(self):
        configs = {name: build_config(name, self.args()) for name in FAMILIES}
        timings = {name: dict(offline=.05, online=.08, collect=.2, diagnostics=2,
                             evaluation_step=.05, evaluation_reset=3) for name in FAMILIES}
        budget = choose_budget(configs, timings, 60, 120)
        updates = budget["updates_per_phase_per_model"]
        self.assertEqual(updates % 64, 0)
        self.assertGreaterEqual(updates, 2048)
        self.assertLessEqual(budget["projected_seconds_total"] + budget["reserve_seconds"], 3600)
        for config in configs.values():
            set_budget(config, updates)
            self.assertEqual(config.training.expert.updates, config.training.online.updates)
            self.assertEqual(online_update_target(config, 0), 0)
            self.assertEqual(online_update_target(config, 2048), 0)
            self.assertEqual(online_update_target(config, config.training.online.steps), updates)
            self.assertEqual((config.training.online.steps // 2 - 1024) / updates, 4)
        with self.assertRaisesRegex(ValueError, "increase --minutes"):
            choose_budget(configs, timings, 1, 120)
        for invalid in (0., float("nan"), float("inf")):
            broken = copy.deepcopy(timings)
            for rate in broken.values():
                rate.update(offline=invalid, online=0., collect=0.)
            with self.assertRaisesRegex(ValueError, "positive, finite"):
                choose_budget(configs, broken, 60, 0)

    def test_scratch_training_calibration_and_reports_with_real_hdf5(self):
        for name in FAMILIES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, _ = fixture(root, name)
                config.env.dataset_root = str(root)
                set_budget(config, 4)
                family = load_model_family(name)
                settings = diagnostic_settings(0)
                settings.context_length = 3
                settings.horizons = [1, 100]
                settings.batch_size = 2
                with family.build_replay(config) as dataset:
                    model = new_model(config, dataset)
                    expected = {k: v.clone() for k, v in model.state_dict().items()}
                    heldout = StateDataset(dataset.h5, dataset.metadata, config.model_io,
                                           config.state_head.fields, model.state_head.targets)
                    windows = heldout.sample_windows(2, 103, settings.window_seed, 3, .5, 2)
                    self.assertTrue(all(w.episode not in dataset.episodes for w in windows))
                    sources = {"heldout_expert": (heldout, windows)}
                    report = {"runs": [{"model": name, "status": "RUNNING"}]}
                    output = root / "reports"
                    output.mkdir()
                    envs = []

                    def environment(*args, **kwargs):
                        envs.append(EvaluationToyEnvironment())
                        return envs[-1]

                    with patch("scripts.train_planner_check.make_envs", side_effect=environment), \
                         patch("scripts.train_planner_check.make_eval_envs", side_effect=environment):
                        rate = calibrate(config, dataset, sources, settings)
                        self.assertTrue(all(rate[key] > 0 for key in ("offline", "online", "collect", "diagnostics")))
                        fresh = new_model(config, dataset)
                        for key, value in expected.items():
                            torch.testing.assert_close(fresh.state_dict()[key], value, rtol=0, atol=0)
                        self.assertEqual(int(fresh.state_head.updates), 0)
                        run_training(config, dataset, sources, settings, output, report["runs"][0],
                                     lambda: write_report(output, report))
                    self.assertTrue(all(env.closed for env in envs))
                    result = report["runs"][0]
                    self.assertIn(result["status"], {"PASS", "REGRESSION"})
                    self.assertEqual(result["phases"]["offline"]["updates"], 4)
                    self.assertEqual(result["phases"]["online"]["updates"], 4)
                    self.assertEqual(result["phases"]["online"]["agent_transitions"], 24)
                    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
                    self.assertEqual(len(rows), 8)
                    self.assertFalse(list(output.glob("*.pt")))
                    summary = write_report(output, report)
                    self.assertIn("4/4", summary)
                    self.assertEqual(json.loads((output / "report.json").read_text())["runs"][0]["status"], result["status"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
