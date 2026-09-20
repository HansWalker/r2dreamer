"""CPU regression checks for the TS fitting harness; not the Lambda learning test."""

import copy
import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.check_online_checkpoint_smoke import fixture
from scripts.check_state_normalization import tiny_config
from scripts.check_ts_mechanisms import fake_cases
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.train_ts_ablation import arguments
from scripts.ts_fit_isolation import (
    action_controls, branch_bank, branch_scores, head_controls, head_gradients,
    head_windows, run_fit_isolation,
)
from training import load_model_family


class TSFitIsolationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.family = load_model_family("temporal_straightening")
        config = tiny_config("temporal_straightening", "cartpole_balance_sparse")
        config.state_head.samples_per_update = 4
        self.model = self.family.build_model(config)

    def test_cli_mode_budgets_and_mutual_exclusion(self):
        args = arguments(["--dataset-root", "/tmp/expert", "--fit-isolation"])
        self.assertTrue(str(args.output).startswith("runs/ts_fit_isolation_"))
        self.assertEqual((args.expert_updates, args.action_updates, args.online_updates), (1000, 2000, 512))
        with redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()):
            for extra in (("--mechanisms", "--fit-isolation"), ("--action-updates", "0")):
                with self.assertRaises(SystemExit):
                    arguments(["--dataset-root", "/tmp/expert", *extra])

    def test_causal_head_windows_keep_labels_and_clip_boundaries(self):
        features = torch.arange(2 * 4 * 2 * 3).reshape(2, 4, 2, 3).float()
        labels = torch.arange(2 * 4 * 5).reshape(2, 4, 5).float()
        f, t = head_windows(features, labels, 3)
        self.assertEqual(f.shape, (4, 3, 2, 3))
        for index, (episode, start) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            torch.testing.assert_close(f[index], features[episode, start:start + 3])
            torch.testing.assert_close(t[index], labels[episode, start:start + 3])
        with self.assertRaises(ValueError):
            head_windows(features, labels[:, :2], 3)

    def test_branch_windows_align_outgoing_actions_and_true_successors(self):
        case = fake_cases()[0]
        case["prefix"] = torch.arange(3, dtype=torch.uint8)[:, None, None, None].expand(3, 64, 64, 3)
        case["image"] = torch.arange(3, 18, dtype=torch.uint8).reshape(3, 5, 1, 1, 1).expand(3, 5, 64, 64, 3)
        case["past_action"] = torch.tensor([[.1], [.2]])
        with patch.object(self.model, "encode", side_effect=lambda obs: obs["image"][..., 0, 0, :1].float().unsqueeze(-2)):
            bank = branch_bank(self.model, [case])
        self.assertEqual(bank["latent"].shape, (15, 4, 1, 1))
        for step in range(5):
            for candidate in range(3):
                row = 3 * step + candidate
                sequence = torch.cat((torch.arange(3), torch.arange(3 + 5 * candidate, 8 + 5 * candidate))).float()
                torch.testing.assert_close(bank["latent"][row, :, 0, 0], sequence[step:step + 4])
                actions = torch.cat((case["past_action"], case["action"][candidate]))
                torch.testing.assert_close(bank["action"][row], actions[step:step + 3])
        self.assertFalse(bank["latent"].requires_grad)

    def test_action_blind_predictor_cannot_pass_fitting_criterion(self):
        bank = branch_bank(self.model, fake_cases())
        rng, state = torch.get_rng_state(), tensor_digest(self.model.state_dict())
        original = self.model.action_encoder.forward
        with patch.object(self.model.action_encoder, "forward", side_effect=lambda a: original(torch.zeros_like(a))):
            result = branch_scores(self.model, bank)["fp32"]
        self.assertFalse(result["fits_train_criterion"])
        for case in result["cases"]:
            for score in case["horizons"].values():
                self.assertEqual(score["predicted_delta_rms"], 0)
                self.assertGreaterEqual(score["mse_over_blind_floor"], 1 - 1e-6)
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        self.assertEqual(state, tensor_digest(self.model.state_dict()))

    def test_gradient_probe_measures_conflict_without_changing_existing_gradients(self):
        class LinearHead(torch.nn.Module):
            samples_per_update = 4
            coordinates = ("x", "y")
            loss_scale = torch.ones(2)

            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)
                torch.nn.init.zeros_(self.linear.weight)

            def _examples(self, features, labels, count):
                return self.linear(features[:count, -1]), labels[:count, -1]

        head = LinearHead()
        head.linear.weight.grad = torch.ones_like(head.linear.weight)
        features = torch.tensor([[[1., 0.]]]).expand(4, 1, 2)
        online = (features, torch.tensor([[[2., 0.]]]).expand(4, 1, 2))
        expert = (features, torch.tensor([[[-.25, 0.]]]).expand(4, 1, 2))
        before = tensor_digest(head.state_dict())
        result = head_gradients(head, online, expert)
        self.assertAlmostEqual(result["norm_ratio"], 4)
        self.assertAlmostEqual(result["cosine"], -1)
        self.assertLess(result["mixed_dot_expert"], 0)
        self.assertEqual(result["online"]["coordinates"]["y"]["gradient_norm"], 0)
        self.assertEqual(before, tensor_digest(head.state_dict()))
        torch.testing.assert_close(head.linear.weight.grad, torch.ones_like(head.linear.weight))
        head.samples_per_update = 1
        with self.assertRaises(ValueError):
            head_gradients(head, online, expert)

    def test_controls_share_start_and_never_touch_protected_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary), seed=0, action_updates=2, online_updates=4)
            shared = copy.deepcopy(self.family.checkpoint(self.model))
            bank = branch_bank(self.model, fake_cases())
            report = {"actions": [], "heads": []}
            with redirect_stdout(io.StringIO()), \
                 patch.object(self.model, "act", side_effect=AssertionError("No planner")), \
                 patch.object(self.model.state_head, "fit", side_effect=AssertionError("No head fitting in action controls")):
                action_controls(self.model, shared, {"train": bank, "validation": bank}, args, report)
            self.assertEqual([r["mode"] for r in report["actions"]], ["native_dropout", "dropout_off"])
            self.assertTrue(all(r["protected_state_unchanged"] for r in report["actions"]))
            self.assertTrue(all(r["parameter_delta_rms"]["action_encoder"] > 0 for r in report["actions"]))
            labels = torch.randn(len(bank["latent"]), 4, 5)
            labels[..., 1] = 1
            features, labels = head_windows(bank["latent"], labels, self.model.state_head.history)
            banks = {name: (features, labels) for name in ("expert_train", "simulator_train", "expert_validation", "simulator_validation")}
            with redirect_stdout(io.StringIO()), \
                 patch.object(self.model, "encode", side_effect=AssertionError("Head controls must use cached features")), \
                 patch.object(self.model, "predict", side_effect=AssertionError("No predictor training in head controls")):
                head_controls(self.model, shared, banks, args, report)
            initial = tensor_digest(shared["model_state_dict"])
            self.assertTrue(all(r["initial_state_sha256"] == initial for r in report["actions"] + report["heads"]))
            self.assertTrue(all(r["native_state_unchanged"] for r in report["heads"]))
            for mode, count in (("expert_only", 0), ("mixed_50_50", self.model.state_head.samples_per_update // 2)):
                metrics = [json.loads(line) for line in (args.output / f"head_{mode}" / "metrics.jsonl").read_text().splitlines()]
                self.assertEqual(len(metrics), 4)
                self.assertTrue(all(row["state/expert_examples"] == count for row in metrics))
            self.assertFalse(list(args.output.rglob("*.pt")))

    def test_small_end_to_end_harness_has_no_checkpoint_io_or_policy_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = fixture(root, "temporal_straightening")
            config.env.dataset_root = str(root)
            args = SimpleNamespace(output=root / "reports", seed=0, expert_updates=2, action_updates=2, online_updates=2)
            args.output.mkdir()

            def cases(config, settings):
                result = fake_cases()
                result[0].update(seed=settings.sim_seeds[0], rollin_policy="zero", anchor_agent_step=2,
                                 relation=torch.zeros(3, 5, 2), id=str(settings.sim_seeds[0]))
                return result

            def episode(config, model, seed, mode):
                images = fake_cases()[0]["image"].flatten(0, 1)
                labels = torch.zeros(len(images), 5)
                labels[:, 1] = 1
                return {"image": images, "state": labels, "seed": seed, "policy": mode,
                        "action": torch.zeros(len(images) - 1, 1), "return": 0, "agent_steps": len(images) - 1}

            report = {"runs": []}
            with self.family.build_replay(config) as dataset, redirect_stdout(io.StringIO()), \
                 patch("scripts.ts_fit_isolation.collect_cases", side_effect=cases), \
                 patch("scripts.ts_fit_isolation.collect_episode", side_effect=episode), \
                 patch("torch.save", side_effect=AssertionError("No checkpoint writes")), \
                 patch("torch.load", side_effect=AssertionError("No checkpoint reads")):
                status = run_fit_isolation(config, dataset, args, report, time.monotonic())
            self.assertEqual(status, 0)
            self.assertEqual(report["status"], "COMPLETE")
            self.assertEqual(report["shared_offline"]["updates"], 2)
            self.assertEqual(len(report["heads"]), 2)
            self.assertEqual(len(report["actions"]), 2)
            self.assertNotEqual(report["action_cases"]["train"][0]["seed"], report["action_cases"]["validation"][0]["seed"])
            self.assertTrue(all(w["episode"] != 0 for w in report["expert_validation_windows"]))
            self.assertEqual(len((args.output / "offline_metrics.jsonl").read_text().splitlines()), 2)
            json.loads((args.output / "report.json").read_text())
            self.assertIn("not that either problem is repaired", (args.output / "summary.txt").read_text())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
