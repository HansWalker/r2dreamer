"""CPU harness checks for the TS mechanisms diagnostic; not the Lambda experiment."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.temporal_straightening import TemporalStraightening
from scripts.check_online_checkpoint_smoke import fixture
from scripts.check_state_normalization import tiny_config
from scripts.diagnose_fixed_replay import FixedBatches, batchnorm_state
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.diagnose_planner_oracle import collect_cases
from scripts.online_validation import TrajectoryDataset
from scripts.smoke_models import synthetic_batch
from scripts.train_ts_ablation import arguments
from scripts.train_planner_check import diagnostic_settings, new_model
from scripts.ts_mechanisms import (
    MODES, action_probes, case_metadata, crossed_readouts, observed_bank,
    response_metrics, train_controls, write_report,
)
from training import load_model_family
from training.evaluation import StateDataset


def fake_cases():
    generator = torch.Generator().manual_seed(107)
    return [{"id": "fake", "prefix": torch.randint(256, (3, 64, 64, 3), dtype=torch.uint8, generator=generator),
             "past_action": torch.zeros(2, 1),
             "action": torch.tensor([0., -1., 1.])[:, None, None].expand(3, 5, 1).clone(),
             "image": torch.randint(256, (3, 5, 64, 64, 3), dtype=torch.uint8, generator=generator)}]


class TSMechanismsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(93)

    def test_cli_is_opt_in_and_separate_from_curvature_comparison(self):
        default = arguments(["--dataset-root", "/tmp/expert"])
        mechanisms = arguments(["--dataset-root", "/tmp/expert", "--mechanisms"])
        self.assertFalse(default.mechanisms)
        self.assertTrue(mechanisms.mechanisms)
        self.assertTrue(str(default.output).startswith("runs/ts_ablation_"))
        self.assertTrue(str(mechanisms.output).startswith("runs/ts_mechanisms_"))
        self.assertEqual((mechanisms.expert_updates, mechanisms.online_updates), (1000, 512))

    def test_action_response_distinguishes_correct_ignored_and_inverted_controls(self):
        actual = torch.tensor([[0., 0.], [-1., -2.], [1., 2.]])
        correct = response_metrics(actual, actual)
        self.assertEqual(correct["matched_action_rmse"], 0)
        self.assertAlmostEqual(correct["response_ratio"], 1)
        self.assertAlmostEqual(correct["response_cosine"], 1, places=6)
        self.assertGreater(correct["wrong_action_rmse"], 0)
        ignored = response_metrics(torch.zeros_like(actual), actual)
        self.assertEqual(ignored["response_ratio"], 0)
        self.assertIsNone(ignored["response_cosine"])
        self.assertAlmostEqual(ignored["matched_action_rmse"], ignored["wrong_action_rmse"])
        inverted = response_metrics(-actual, actual)
        self.assertAlmostEqual(inverted["response_cosine"], -1, places=6)
        self.assertIsNone(response_metrics(actual, torch.zeros_like(actual))["response_ratio"])

    def test_action_probes_preserve_weights_gradients_modes_and_rng(self):
        config = tiny_config("temporal_straightening", "cartpole_balance_sparse")
        model = load_model_family("temporal_straightening").build_model(config)
        batch, _, _ = synthetic_batch(config, model)
        model.update(batch)
        model.set_adaptation_mode("frozen_bn")
        before = tensor_digest(model.state_dict())
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
        rng, amp = torch.get_rng_state(), model.use_amp
        modes = [module.training for module in model.modules()]
        with patch.object(model.state_head, "forward", side_effect=AssertionError("No head in action probe")), \
             patch.object(model, "act", side_effect=AssertionError("No planner")):
            scores = action_probes(model, fake_cases())["fp32"][0]
        self.assertGreater(scores["gradients"]["current_action_jacobian_projection_norm"], 0)
        self.assertTrue(scores["gradients"]["input_connected"])
        self.assertEqual(scores["gradients"]["action_encoder_parameters_with_gradient"],
                         scores["gradients"]["action_encoder_parameter_tensors"])
        self.assertEqual(before, tensor_digest(model.state_dict()))
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertEqual(amp, model.use_amp)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(gradients.keys(), {name for name, p in model.named_parameters() if p.grad is not None})
        for name, gradient in gradients.items():
            torch.testing.assert_close(dict(model.named_parameters())[name].grad, gradient, rtol=0, atol=0)
        original = model.action_encoder.forward
        with patch.object(model.action_encoder, "forward", side_effect=lambda action: original(torch.zeros_like(action))):
            disconnected = action_probes(model, fake_cases())["fp32"][0]
        self.assertFalse(disconnected["gradients"]["input_connected"])
        self.assertEqual(disconnected["5"]["predicted_delta_rms"], 0)

    def test_crossed_readouts_isolate_head_and_running_statistics_without_mutation(self):
        config = tiny_config("temporal_straightening", "cartpole_balance_sparse")
        model = load_model_family("temporal_straightening").build_model(config)
        episode = {"image": fake_cases()[0]["image"].flatten(0, 1), "state": torch.randn(15, 5),
                   "action": torch.randn(14, 1), "seed": 71, "policy": "random"}
        data = TrajectoryDataset([episode])
        settings = SimpleNamespace(context_length=3, batch_size=2)
        sources = {"simulator": (data, data.sample_windows(2, 4, 93))}
        old_head = copy.deepcopy(model.state_head)
        old_bank = observed_bank(model, sources, settings)
        old_bn = {key: value.clone() for key, value in batchnorm_state(model).items()}
        with torch.no_grad():
            model.state_head.readout[-1].bias.add_(.25)
        cross = crossed_readouts(model, old_head, old_bank, old_bn, sources, settings)["simulator"]
        self.assertEqual(cross["E0_D0"], cross["Et_D0"])
        self.assertEqual(cross["E0_Dt"], cross["Et_Dt"])
        self.assertNotEqual(cross["E0_D0"], cross["E0_Dt"])
        with torch.no_grad():
            next(module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)).running_mean.add_(3)
        before = tensor_digest(model.state_dict())
        cross = crossed_readouts(model, old_head, old_bank, old_bn, sources, settings)["simulator"]
        self.assertGreater(cross["feature_drift_rms"], 0)
        self.assertEqual(cross["feature_drift_with_old_bn_rms"], 0)
        self.assertEqual(cross["E0_Dt"], cross["Et_BN0_Dt"])
        self.assertEqual(before, tensor_digest(model.state_dict()))

    def test_controls_pretrain_once_share_weights_moments_and_batches_and_do_not_save(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = fixture(root, "temporal_straightening")
            config.env.dataset_root = str(root)
            args = SimpleNamespace(seed=0, expert_updates=2, online_updates=4, output=root / "reports")
            args.output.mkdir()
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
                episodes = [{"image": fake_cases()[0]["image"].flatten(0, 1), "state": torch.randn(15, 5),
                             "action": torch.rand(14, 1) * 2 - 1}]
                replay = FixedBatches(episodes, config, 7)
                plans = [replay.draw() for _ in range(args.online_updates)]
                sources = {"expert": (heldout, windows)}
                signatures, expert_signatures = [], []
                update, sample = TemporalStraightening.update, dataset.sample_episode_batch

                def tracked_update(model, batch):
                    signatures.append(tensor_digest({**batch[0], "action": batch[1]}))
                    return update(model, batch)

                def tracked_sample():
                    batch = sample()
                    expert_signatures.append(tensor_digest({**batch[0], "action": batch[1]}))
                    return batch

                report = {"runs": []}
                with patch.object(TemporalStraightening, "update", tracked_update), \
                     patch.object(TemporalStraightening, "act", side_effect=AssertionError("No planner")), \
                     patch.object(dataset, "sample_episode_batch", side_effect=tracked_sample), \
                     patch("torch.save", side_effect=AssertionError("No checkpoint write")), \
                     patch("torch.load", side_effect=AssertionError("No checkpoint read")), redirect_stdout(io.StringIO()):
                    train_controls(config, dataset, sampler_start, replay, plans, sources, settings,
                                   fake_cases(), args, report)
                self.assertEqual(len(signatures), 2 + 3 * 4)
                self.assertEqual(len(expert_signatures), 2 + 3 * 4)
                for hashes in (signatures, expert_signatures):
                    self.assertEqual(hashes[2:6], hashes[6:10])
                    self.assertEqual(hashes[2:6], hashes[10:14])
                self.assertEqual([r["mode"] for r in report["runs"]], list(MODES))
                for run in report["runs"]:
                    self.assertNotEqual(run["status"], "FAIL", run.get("error"))
                    self.assertEqual(run["updates"], 4)
                    self.assertEqual(run["initial_state_sha256"], report["shared_offline"]["state_sha256"])
                    self.assertEqual(run["batchnorm_unchanged"], run["mode"] != "native")
                    self.assertEqual(run["encoder_unchanged"], run["mode"] == "frozen_encoder")
                    self.assertGreater(run["parameter_delta_rms"]["action_encoder"], 0)
                    self.assertEqual(run["parameter_delta_rms"]["encoder"] == 0, run["mode"] == "frozen_encoder")
                    rows = [json.loads(line) for line in (args.output / run["mode"] / "metrics.jsonl").read_text().splitlines()]
                    self.assertEqual(rows[-1]["state/updates"], 6)
                    self.assertEqual(len(run["snapshots"]), 2)
                self.assertEqual(len((args.output / "offline_metrics.jsonl").read_text().splitlines()), 2)
                self.assertFalse(list(args.output.rglob("*.pt")))
                summary = write_report(args.output, report)
                self.assertIn("E0_D0", summary)
                self.assertIn("frozen_encoder/fp32", summary)
                self.assertEqual(len(json.loads((args.output / "report.json").read_text())["runs"]), 3)

    def test_counterfactual_angles_wrap_without_changing_true_relations(self):
        case = fake_cases()[0]
        case.update(seed=73, rollin_policy="random", anchor_agent_step=2, relation=torch.zeros(3, 5, 2))
        case["relation"][1, -1, 1] = 3.1
        case["relation"][2, -1, 1] = -3.1
        before = case["relation"].clone()
        result = case_metadata([case])[0]
        self.assertLess(abs(result["terminal_physical_delta_plus_minus"][1]), .1)
        torch.testing.assert_close(case["relation"], before, rtol=0, atol=0)

    def test_real_simulator_counterfactuals_have_physical_and_pixel_effects(self):
        config = tiny_config("temporal_straightening", "cartpole_balance_sparse")
        config.env.time_limit = 200
        config.jepa_model.planner.horizon = 5
        cases = collect_cases(config, SimpleNamespace(sim_seeds=[10_000_000], rollin_steps=[0], candidates=3))
        metadata = case_metadata(cases)
        self.assertEqual(len(metadata), 2)
        for case in metadata:
            self.assertGreater(case["terminal_pixel_delta_rms_0_255"], 0)
            self.assertGreater(abs(case["terminal_physical_delta_plus_minus"][0]), 0)
        model = load_model_family("temporal_straightening").build_model(config)
        scores = action_probes(model, cases)
        self.assertEqual(len(scores["fp32"]), 2)
        self.assertTrue(all(case["5"]["actual_encoded_delta_rms"] > 0 for case in scores["fp32"]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
