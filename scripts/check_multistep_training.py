"""Recursive-loss contracts and a real two-model offline-to-online CPU run."""

import io
import json
import math
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
from torch import nn

from models.planning import LatentPlanner
from scripts import check_forecast_online as fixtures
from scripts.check_state_normalization import tiny_config
from scripts.evaluate_goal_maintenance import load_source
from scripts.multistep_training_support import ONE_STEP_SOURCE_IMPLEMENTATION
from scripts.smoke_models import synthetic_batch
from scripts.train_forecast_online import arguments, configure, main
from scripts.train_paper_faithful_duration import file_hash
from training import load_model_family
from training.planning import OnlineSession
from training.protocol import validate_training_recipe


FAMILIES = ("temporal_straightening", "leworldmodel")


def model_for(family, task="cartpole_balance_sparse", horizon=5):
    config = tiny_config(family, task)
    config.jepa_model.training_horizon = horizon
    config.replay.sequence_length = 3 + horizon
    config.state_head.samples_per_update = 4
    return config, load_model_family(family).build_model(config)


class LinearPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(2.))
        self.inputs = []

    def forward(self, state, action):
        self.inputs.append((state.detach().clone(), action.detach().clone()))
        return self.gain * state + action


class ToyPlanner(LatentPlanner):
    def __init__(self):
        nn.Module.__init__(self)
        self.history_size, self.training_horizon, self.sequence_length = 3, 5, 8
        self.action_dim, self.use_amp = 1, False
        self.predictor = LinearPredictor()
        self.action_encoder = self.pred_projector = nn.Identity()


class RecursiveLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_recursive_inputs_action_alignment_future_exclusion_and_long_gradient(self):
        model = ToyPlanner()
        latent = torch.arange(8.).reshape(1, 8, 1).requires_grad_()
        action = torch.arange(10., 17.).reshape(1, 7, 1)
        prediction, target = model.training_predictions(latent, action)
        value, expected = latent[:, 2].detach(), []
        for step in range(5):
            value = 2 * value + action[:, step + 2]
            expected.append(value)
            torch.testing.assert_close(model.predictor.inputs[step][1], action[:, step:step + 3])
            if step:
                torch.testing.assert_close(model.predictor.inputs[step][0][:, -1], expected[step - 1])
        torch.testing.assert_close(prediction, torch.stack(expected, 1))
        torch.testing.assert_close(target, latent[:, 3:])
        perturbed = latent.detach().clone()
        perturbed[:, 3:] += 1000
        torch.testing.assert_close(model.training_predictions(perturbed, action)[0], prediction)
        prediction[:, -1].sum().backward()
        self.assertEqual(latent.grad[0, 2, 0].item(), 32.)
        self.assertEqual(latent.grad[:, 3:].abs().sum().item(), 0.)
        self.assertGreater(model.predictor.gain.grad.abs().item(), 0.)

    def test_actual_rollout_matches_uncached_predictions_and_gradients(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                config, model = model_for(family)
                model.eval()
                batch, _, _ = synthetic_batch(config, model, length=8)
                latent = model.encode(batch[0]).detach().requires_grad_()
                action = batch[1].clone().requires_grad_()
                predicted, _ = model.training_predictions(latent, action)
                state, expected = latent[:, :3], []
                for step in range(5):
                    future = model.predict(state, action[:, step:step + 3])[:, -1]
                    expected.append(future)
                    state = torch.cat((state[:, 1:], future[:, None]), 1)
                expected = torch.stack(expected, 1)
                torch.testing.assert_close(predicted, expected, atol=1e-5, rtol=1e-5)
                left = torch.autograd.grad(predicted[:, -1].square().mean(), (latent, action), retain_graph=True)
                right = torch.autograd.grad(expected[:, -1].square().mean(), (latent, action))
                for a, b in zip(left, right, strict=True):
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
                self.assertGreater(left[0][:, :3].abs().sum().item(), 0.)
                self.assertEqual(left[0][:, 3:].abs().sum().item(), 0.)

    def test_default_one_step_is_exact_and_target_gradient_rules_are_preserved(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                config, model = model_for(family, horizon=1)
                model.eval()
                batch, _, _ = synthetic_batch(config, model, length=4)
                latent = model.encode(batch[0])
                actual, target = model.training_predictions(latent, batch[1])
                torch.testing.assert_close(actual, model.predict(latent[:, :-1], batch[1]), rtol=0, atol=0)
                torch.testing.assert_close(target, latent[:, 1:], rtol=0, atol=0)
                default_shapes = {key: value.shape for key, value in model.state_dict().items()}
                config, model = model_for(family)
                self.assertEqual(default_shapes, {key: value.shape for key, value in model.state_dict().items()})
                model.eval()
                model.curvature_weight = model.sigreg_weight = 0.
                batch, _, _ = synthetic_batch(config, model, length=8)
                latent = model.encode(batch[0]).detach().requires_grad_()
                loss, metrics = model.representation_loss(batch[0], latent, batch[1])
                gradient, = torch.autograd.grad(loss, (latent,))
                if family == "temporal_straightening":
                    self.assertEqual(gradient[:, 3:].abs().sum().item(), 0.)
                else:
                    self.assertGreater(gradient[:, 3:].abs().sum().item(), 0.)
                torch.testing.assert_close(metrics["prediction_loss"],
                                           torch.stack([metrics[f"prediction_step_{i}"] for i in range(1, 6)]).mean())

    def test_both_phases_work_on_all_three_tasks_without_extra_labels(self):
        for family in FAMILIES:
            for task in ("cartpole_balance_sparse", "reacher", "ball_in_cup"):
                with self.subTest(family=family, task=task):
                    config, model = model_for(family, task)
                    batch, _, _ = synthetic_batch(config, model, batch_size=4, length=8)
                    if hasattr(model, "configure_pretraining"):
                        model.configure_pretraining(4, resumed=False)
                    offline = model.update(batch)
                    session = OnlineSession(config, model, None)
                    # Production OnlineSession strips the action leaving the last
                    # observation. Its real collection/replay path is tested below.
                    action = torch.cat((batch[1], torch.zeros_like(batch[1][:, :1])), 1)
                    rewards = torch.zeros(4, 8, 1)
                    with patch.object(session.replay, "sample", return_value=(batch[0], action, rewards, rewards)):
                        online = session.update(1)
                    for metrics in (offline, online):
                        self.assertTrue(all(math.isfinite(float(value)) for value in metrics.values()))
                        self.assertIn("prediction_step_5", metrics)
                    self.assertEqual(int(model.state_head.updates), 2)
                    self.assertEqual(int(model.state_head.examples), 8)
                    self.assertEqual(model.history_size, 3)
                    self.assertEqual(model.sequence_length, 8)

    def test_incomplete_or_misaligned_windows_fail(self):
        model = ToyPlanner()
        with self.assertRaisesRegex(ValueError, "aligned actions"):
            model.training_predictions(torch.zeros(2, 7, 1), torch.zeros(2, 6, 1))
        with self.assertRaisesRegex(ValueError, "aligned actions"):
            model.training_predictions(torch.zeros(2, 8, 1), torch.zeros(2, 6, 1))
        for family in FAMILIES:
            with self.assertRaisesRegex(ValueError, "positive"):
                model_for(family, horizon=0)

    def test_production_recipe_accepts_multistep_and_rejects_mismatched_replay(self):
        for family in FAMILIES:
            config, _ = model_for(family)
            validate_training_recipe(config)
            config.replay.sequence_length = 4
            with self.assertRaisesRegex(ValueError, "history_size"):
                validate_training_recipe(config)

    def test_optional_ts_decoder_uses_the_five_corresponding_future_images(self):
        config, _ = model_for("temporal_straightening")
        config.jepa_model.decoder.enabled = True
        model = load_model_family(config.model_family).build_model(config).eval()
        batch, _, _ = synthetic_batch(config, model, length=8)
        latent = model.encode(batch[0])
        prediction, _ = model.training_predictions(latent, batch[1])
        _, metrics = model.representation_loss(batch[0], latent, batch[1])
        expected = torch.nn.functional.mse_loss(model.decoder(prediction.detach()), model.encoder.target(batch[0])[:, 3:])
        torch.testing.assert_close(metrics["decoder_prediction_loss"], expected)
        model.train()
        self.assertTrue(math.isfinite(model.update(batch)["loss"]))


class MultiStepExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ForecastOnlineTests.setUpClass.__func__(cls)
        for row in cls.source_report["runs"]:
            row["coverage_fraction"] = .5
        cls.source_report["implementation_sha256"] = ONE_STEP_SOURCE_IMPLEMENTATION
        (cls.source / "report.json").write_text(json.dumps(cls.source_report))
        cls.hashes = {str(p.relative_to(cls.source)): file_hash(p) for p in cls.source.rglob("*") if p.is_file()}

    @classmethod
    def tearDownClass(cls):
        fixtures.ForecastOnlineTests.tearDownClass.__func__(cls)

    def command(self, name):
        return fixtures.ForecastOnlineTests.command(self, name, "--online-steps", "64", "--retain-offline",
                                          "--training-horizon", "5", "--offline-updates", "4")

    def test_two_model_real_offline_online_run_and_fixed_evaluation(self):
        log = io.StringIO()
        with redirect_stdout(log):
            status = main(self.command("multistep"))
        self.assertEqual(status, 0, log.getvalue())
        output = self.root / "multistep"
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(len(report["runs"]), 2)
        self.assertTrue(report["training_adaptation"]["paper_objective_changed"])
        for row in report["runs"]:
            self.assertEqual((row["offline_updates"], row["updates"]), (4, 7))
            self.assertEqual(row["budget"]["sequence_length"], 8)
            self.assertEqual(row["config"]["jepa_model"]["history_size"], 3)
            snapshots = row["snapshots"]
            self.assertEqual([s["name"] for s in snapshots], ["before", "after_offline", "midpoint", "after"])
            self.assertEqual([s["updates"] for s in snapshots], [0, 0, 2, 7])
            self.assertEqual([s["offline_updates"] for s in snapshots], [0, 4, 4, 4])
            self.assertEqual(len({s["evaluation"]["result"]["fixed_initial_plans"]["data_sha256"] for s in snapshots}), 1)
            self.assertNotEqual(snapshots[0]["model_state_sha256"], snapshots[1]["model_state_sha256"])
            for item in snapshots:
                result = item["evaluation"]["result"]
                self.assertTrue(result["training_state_preserved"])
                self.assertEqual(result["policy"]["case_ids"], report["control"]["case_ids"])
            folder = output / row["task"] / row["model"]
            offline = [json.loads(line) for line in (folder / "offline_metrics.jsonl").read_text().splitlines()]
            online = [json.loads(line) for line in (folder / "online_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(offline), 4)
            self.assertTrue(all("prediction_step_5" in entry["metrics"] for entry in offline))
            active = [entry["metrics"] for entry in online if entry["metrics"]]
            self.assertTrue(all("prediction_step_5" in metrics for metrics in active))
            self.assertEqual(active[-1]["native/offline_sequences"], 0)
            self.assertTrue(all(metrics["state/expert_examples"] == 2 for metrics in active))
            saved = torch.load(folder / "offline_latest.pt", weights_only=False)
            source = torch.load(self.source / row["task"] / row["model"] / "latest.pt", weights_only=False)
            predictor_names = [key for key in source["model_state_dict"] if key.startswith("predictor.")]
            self.assertTrue(any(not torch.equal(saved["model_state_dict"][key], source["model_state_dict"][key])
                                for key in predictor_names), "Offline adaptation must update the predictor itself")
            self.assertEqual(saved["updates"], 4)
            self.assertFalse(saved["resume_supported"])
            if row["model"] == "leworldmodel":
                self.assertEqual(saved["optimizer_state_dict"]["scheduler"]["last_epoch"], 4)
                self.assertGreater(offline[0]["metrics"]["lr"], 0.)
                self.assertEqual(offline[-1]["metrics"]["lr"], 0.)
            latest = torch.load(folder / "latest.pt", weights_only=False)
            self.assertTrue(any(not torch.equal(latest["model_state_dict"][key], saved["model_state_dict"][key])
                                for key in predictor_names), "Online learning must update the predictor itself")
            self.assertEqual((latest["offline_updates"], latest["updates"]), (4, 7))
            self.assertEqual(latest["counters"]["_gradient_updates"], saved["counters"]["_gradient_updates"] + 7)
            self.assertEqual(sum(row["online"]["replay_sampling"]["total_samples"].values()), 28)
            self.assertEqual(row["online"]["head_updates_after"] - row["online"]["head_updates_before"], 7)
            for entry in (folder / "online_data.pt", folder / "step_0.pt"):
                self.assertTrue(entry.is_file())
        self.assertIn("Run | multistep | status=COMPLETE", log.getvalue())
        self.assertEqual(self.hashes, {str(p.relative_to(self.source)): file_hash(p) for p in self.source.rglob("*") if p.is_file()})

    def test_legacy_source_allowance_is_explicit_and_narrow(self):
        args = arguments(self.command("unused"))
        with self.assertRaisesRegex(ValueError, "differ"):
            load_source(args)
        _, banks, records = load_source(args, compatible_implementations=(ONE_STEP_SOURCE_IMPLEMENTATION,))
        for record in records:
            config, _, budget = configure(record, banks[record["task"]], args)
            self.assertEqual(config.replay.sequence_length, 8)
            self.assertEqual(budget["offline_adaptation"]["branch_fraction"], .5)
        with self.assertRaisesRegex(ValueError, "differ"):
            load_source(args, compatible_implementations=("unrecognized",))
        with patch("scripts.evaluate_goal_maintenance.source_hashes", return_value={}):
            with self.assertRaisesRegex(ValueError, "differ"):
                load_source(args, compatible_implementations=(ONE_STEP_SOURCE_IMPLEMENTATION,))
        args.training_horizon = 20
        with self.assertRaises(ValueError):
            configure(records[0], banks[records[0]["task"]], args)


if __name__ == "__main__":
    unittest.main()
