"""Small CPU harness checks, not the Lambda training experiment."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.check_online_checkpoint_smoke import fixture
from scripts.diagnose_fixed_replay import FixedBatches
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.train_ts_ablation import (
    VARIANTS, arguments, build_config, collect_shared, comparison, diagnostic_settings,
    legacy_loss, new_model, physical_checks, run_variant, write_report,
)
from training import load_model_family
from training.evaluation import StateDataset


class TSAblationTest(unittest.TestCase):
    def test_legacy_formula_is_flattened_and_production_remains_patchwise(self):
        args = arguments(["--dataset-root", "/tmp/expert", "--device", "cpu"])
        config = build_config("temporal_straightening", args)
        model = new_model(config, SimpleNamespace(state_mean=torch.zeros(5), state_std=torch.ones(5)))
        self.assertLess(sum(p.numel() for p in model.parameters()), 500_000)
        self.assertEqual(config.replay.batch_size, 128)
        self.assertEqual(config.replay.episodes_per_batch, 16)
        self.assertEqual(config.state_head.online.expert_fraction, .5)
        self.assertEqual(config.training.online.expert_fraction, 0)
        latent = torch.tensor([[[[0., 0.], [0., 0.]], [[100., 0.], [1., 0.]],
                                [[200., 0.], [0., 0.]], [[300., 0.], [1., 0.]]]], requires_grad=True)
        with patch.object(model, "predict", return_value=latent[:, 1:]):
            old, old_metrics = legacy_loss(model, {}, latent, None)
            new, new_metrics = model.representation_loss({}, latent, None)
        self.assertLess(float(old_metrics["curvature_loss"].detach()), .001)
        self.assertAlmostEqual(float(new_metrics["curvature_loss"].detach()), 1.)
        self.assertGreater(float(new.detach()), float(old.detach()))
        self.assertTrue(torch.isfinite(torch.autograd.grad(new, latent)[0]).all())

    def test_shared_collection_has_disjoint_validation_and_no_planner_calls(self):
        args = arguments(["--dataset-root", "/tmp/expert", "--device", "cpu"])
        config = build_config("temporal_straightening", args)
        original_limit = config.env.time_limit

        def episode(settings, model, seed, mode):
            self.assertIn(mode, {"zero", "random"})
            return {"seed": seed, "policy": mode, "time_limit": settings.env.time_limit}

        with patch("scripts.train_ts_ablation.collect_episode", side_effect=episode) as collect:
            training, validation = collect_shared(config, None, 42)
        self.assertEqual(collect.call_count, 18)
        self.assertEqual(len(training), 16)
        self.assertTrue(all(ep["time_limit"] == 400 for ep in training))
        self.assertTrue(all(ep["time_limit"] == original_limit for ep in validation.episodes))
        self.assertEqual(config.env.time_limit, original_limit)
        self.assertFalse({ep["seed"] for ep in training} & {ep["seed"] for ep in validation.episodes})

    def test_original_unit_guards_and_honest_verdicts(self):
        metric = {"physical_rmse": {"position": .02, "angle": .005}, "mean_normalized_mse": .1}
        before = {"expert": {"all": {"physical": {
            "observed": {"1": metric}, "forecast": {"5": metric, "100": metric},
        }}}}
        after = copy.deepcopy(before)
        after["expert"]["all"]["physical"]["observed"]["1"]["mean_normalized_mse"] = 1e9
        self.assertTrue(all(v["passed"] for check in physical_checks(before, after).values() for v in check.values()))
        after["expert"]["all"]["physical"]["observed"]["1"]["physical_rmse"]["angle"] = .031
        self.assertFalse(physical_checks(before, after)["expert/observed/h1"]["angle"]["passed"])
        trial = {"status": "NO_REGRESSION", "initial_state_sha256": "initial",
                 "phases": {phase: {"sampler_end_sha256": phase} for phase in ("offline", "adaptation")}}
        runs = [copy.deepcopy(trial), copy.deepcopy(trial)]
        self.assertTrue(comparison(runs).startswith("INCONCLUSIVE"))
        runs[0]["status"] = "REGRESSION"
        self.assertTrue(comparison(runs).startswith("ENCOURAGING"))
        runs[1]["status"] = "REGRESSION"
        self.assertTrue(comparison(runs).startswith("REGRESSION REMAINS"))
        runs[1]["initial_state_sha256"] = "different"
        self.assertTrue(comparison(runs).startswith("INVALID"))

    def test_real_updates_match_inputs_and_keep_validation_out_of_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = fixture(root, "temporal_straightening")
            config.env.dataset_root = str(root)
            args = SimpleNamespace(seed=0, expert_updates=2, online_updates=4)
            settings = diagnostic_settings(0)
            settings.context_length, settings.horizons, settings.batch_size = 3, [1, 5, 100], 2
            family = load_model_family("temporal_straightening")
            with family.build_replay(config) as dataset:
                sampler_start = copy.deepcopy(dataset.state_dict())
                model = new_model(config, dataset)
                heldout = StateDataset(dataset.h5, dataset.metadata, config.model_io,
                                       config.state_head.fields, model.state_head.targets)
                windows = heldout.sample_windows(2, 103, settings.window_seed, 3, .5, 2)
                self.assertTrue(all(w.episode not in dataset.episodes for w in windows))
                generator = torch.Generator().manual_seed(21)
                episodes = [{"image": torch.randint(256, (12, 64, 64, 3), dtype=torch.uint8, generator=generator),
                             "state": torch.randn(12, 5, generator=generator),
                             "action": torch.rand(11, 1, generator=generator) * 2 - 1}]
                replay = FixedBatches(episodes, config, 7)
                plans = [replay.draw() for _ in range(args.online_updates)]
                sources = {"expert": (heldout, windows)}
                report = {"runs": []}
                signatures = []
                sample = dataset.sample_episode_batch

                def tracked_sample():
                    batch = sample()
                    signatures.append(tensor_digest({**batch[0], "action": batch[1]}))
                    return batch

                def persist(result):
                    report["runs"][-1] = result
                    write_report(root, report)

                with patch.object(dataset, "sample_episode_batch", side_effect=tracked_sample):
                    for variant in VARIANTS:
                        report["runs"].append({"variant": variant, "status": "RUNNING"})
                        result = run_variant(config, dataset, sampler_start, replay, plans, sources, settings,
                                             args, variant, root / variant, persist)
                        self.assertNotEqual(result["status"], "FAIL", result.get("error"))
                        self.assertEqual(len(result["snapshots"]), 3)
                        rows = [json.loads(line) for line in (root / variant / "metrics.jsonl").read_text().splitlines()]
                        self.assertEqual(len(rows), 6)
                        self.assertEqual(rows[-1]["state/updates"], 6)
                        self.assertEqual([row["state/expert_examples"] for row in rows[:2]], [0, 0])
                        self.assertTrue(all(row["state/expert_examples"] > 0 for row in rows[2:]))
                        self.assertFalse(list((root / variant).glob("*.pt")))
                self.assertEqual(signatures[:6], signatures[6:])
                self.assertFalse(comparison(report["runs"]).startswith(("INVALID", "INCOMPLETE")))
                saved = json.loads((root / "report.json").read_text())
                self.assertEqual(saved["runs"][0]["initial_state_sha256"], saved["runs"][1]["initial_state_sha256"])
                self.assertIn("NOT own-policy", (root / "summary.txt").read_text())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
