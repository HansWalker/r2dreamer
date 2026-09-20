"""CPU correctness tests for the short-rollout candidate; not a learning-quality test."""

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

from models.planning import LatentPlanner
from scripts.check_online_checkpoint_smoke import fixture
from scripts.check_state_normalization import tiny_config
from scripts.check_ts_mechanisms import fake_cases
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.smoke_tiny_planners import FAMILIES, native_control
from scripts.train_rollout_check import arguments, cache_branches, fit_arm, fitting_loss, main, score_bank, write_report
from training import load_model_family


def cases():
    values = fake_cases()
    for case in values:
        case.update(goal_image=case["prefix"][-1].clone(), reward=torch.zeros(3, 5), relation=torch.zeros(3, 5, 2))
    return values


class RolloutTrainingTest(unittest.TestCase):
    def test_cli_is_small_explicit_and_does_not_accept_repeated_work(self):
        args = arguments(["--dataset-root", "/tmp/expert"])
        self.assertEqual(args.models, list(FAMILIES))
        self.assertEqual((args.expert_updates, args.fit_updates, args.scenario), (1000, 2000, "cartpole_balance_sparse"))
        with patch("sys.stderr", new=io.StringIO()):
            for extra in (("--fit-updates", "0"), ("--models", "leworldmodel", "leworldmodel")):
                with self.assertRaises(SystemExit):
                    arguments(["--dataset-root", "/tmp/expert", *extra])

    def test_recursive_loss_keeps_all_step_gradients_and_detaches_targets(self):
        class Predictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gain = torch.nn.Parameter(torch.tensor(2.))

            def forward(self, state, action):
                return self.gain * state + action

        class Model:
            history_size, action_dim, prediction_weight = 3, 1, .75
            _rollout = LatentPlanner._rollout
            _predict_rollout = LatentPlanner._predict_rollout
            rollout = LatentPlanner._rollout
            predictor = Predictor()
            action_encoder = torch.nn.Identity()
            pred_projector = torch.nn.Identity()

            def representation_loss(self, obs, latent, action):
                prediction = self.predictor.gain.square()
                return self.prediction_weight * prediction + 7, {"prediction_loss": prediction}

        model = Model()
        target = torch.zeros(1, 1, 3, 1, requires_grad=True)
        bank = {"latent": None, "action": None, "prefix": torch.ones(1, 3, 1),
                "past_action": torch.zeros(1, 2, 1), "future_action": torch.zeros(1, 1, 3, 1), "target": target}
        loss, metrics = fitting_loss(model, bank, 1.)
        self.assertEqual(metrics["rollout_loss"].item(), 28.)
        self.assertEqual(loss.item(), 28 * .75 + 7)
        loss.backward()
        self.assertAlmostEqual(model.predictor.gain.grad.item(), 76 * .75)
        self.assertIsNone(target.grad)
        mixed, _ = fitting_loss(model, bank, .5)
        self.assertEqual(mixed.item(), .75 * (4 + 28) / 2 + 7)
        # Changing future labels cannot alter the rollout input or prediction.
        prediction = model.rollout(bank["prefix"], bank["past_action"], bank["future_action"]).detach()
        bank["target"] = torch.ones_like(target) * 100
        changed, _ = fitting_loss(model, bank, 1.)
        self.assertNotEqual(changed.item(), loss.item())
        torch.testing.assert_close(prediction, model.rollout(bank["prefix"], bank["past_action"], bank["future_action"]))
        for invalid in (-1, 2, float("nan")):
            with self.assertRaises(ValueError):
                fitting_loss(model, bank, invalid)

    def test_planner_rollout_matches_uncached_recursion_and_action_gradients(self):
        for name in FAMILIES:
            with self.subTest(model=name):
                torch.manual_seed(8)
                config = tiny_config(name, "cartpole_balance_sparse")
                model = load_model_family(name).build_model(config).eval()
                bank = cache_branches(model, cases())
                action = bank["future_action"].detach().requires_grad_()
                predicted = native_control(model, lambda: model.rollout(bank["prefix"], bank["past_action"], action))
                count, horizon = action.shape[1:3]
                state = bank["prefix"].expand(count, *bank["prefix"].shape[1:])
                controls = torch.cat((bank["past_action"].expand(count, -1, -1), action[0]), dim=1)
                manual = []
                for step in range(horizon):
                    output = model.predict(state, controls[:, step:step + model.history_size])[:, -1:]
                    manual.append(output[:, 0])
                    state = torch.cat((state[:, 1:], output), dim=1)
                manual = torch.stack(manual, dim=1)[None]
                torch.testing.assert_close(predicted, manual, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(torch.autograd.grad(predicted.square().mean(), action)[0],
                                           torch.autograd.grad(manual.square().mean(), action)[0], rtol=1e-4, atol=1e-6)

    def test_zero_weight_is_native_loss_without_extra_rollout(self):
        for name in FAMILIES:
            with self.subTest(model=name):
                model = load_model_family(name).build_model(tiny_config(name, "cartpole_balance_sparse")).eval()
                bank = cache_branches(model, cases())
                torch.manual_seed(1)
                expected, _ = model.representation_loss({}, bank["latent"], bank["action"])
                torch.manual_seed(1)
                with patch.object(model, "rollout", side_effect=AssertionError("Unnecessary rollout")):
                    actual, _ = fitting_loss(model, bank, 0.)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_corrupt_physical_head_cannot_change_either_planner_actions(self):
        for name in FAMILIES:
            with self.subTest(model=name):
                config = tiny_config(name, "cartpole_balance_sparse")
                config.jepa_model.planner.samples = 4
                config.jepa_model.planner.elites = 2
                config.jepa_model.planner.iterations = 2
                config.jepa_model.planner.horizon = 2
                model = load_model_family(name).build_model(config)
                case = cases()[0]
                history = {"image": case["prefix"][None], "goal_image": case["goal_image"][None]}
                def act():
                    model._cem_mean = model._gradient_actions = None
                    torch.manual_seed(13)
                    return native_control(model, lambda: model.act(history, case["past_action"][None], deterministic=True))
                before = act()
                with torch.no_grad():
                    for parameter in model.state_head.parameters():
                        parameter.fill_(float("nan"))
                after = act()
                self.assertTrue(torch.isfinite(after).all())
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_both_arms_restore_same_weights_keep_heads_fixed_and_write_no_checkpoints(self):
        for name in FAMILIES:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as temporary:
                config = tiny_config(name, "cartpole_balance_sparse")
                family = load_model_family(name)
                model = family.build_model(config)
                bank = cache_branches(model, cases())
                shared = copy.deepcopy(family.checkpoint(model))
                args = SimpleNamespace(seed=0, fit_updates=3)
                result = {"model": name, "status": "RUNNING", "arms": []}
                report = {"runs": [result]}
                output = Path(temporary)
                persist = lambda: write_report(output, report)
                rng, state = torch.get_rng_state(), tensor_digest(model.state_dict())
                scored = score_bank(model, bank)
                self.assertEqual(scored["cases"][0]["return_informative"], False)
                torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
                self.assertEqual(state, tensor_digest(model.state_dict()))
                with redirect_stdout(io.StringIO()), \
                     patch.object(model.state_head, "fit", side_effect=AssertionError("No head fitting")), \
                     patch("torch.save", side_effect=AssertionError("No checkpoints")):
                    for weight in (0., .5):
                        fit_arm(model, config, shared, {"train": bank, "validation": bank}, weight, args, output, result, persist)
                for arm in result["arms"]:
                    self.assertEqual(arm["initial_state_sha256"], state)
                    self.assertTrue(arm["protected_state_unchanged"])
                    self.assertEqual(arm["status"], "COMPLETE")
                    rows = [json.loads(line) for line in (output / arm["name"] / "metrics.jsonl").read_text().splitlines()]
                    self.assertEqual(len(rows), args.fit_updates)
                    self.assertEqual("rollout_loss" in rows[-1], bool(arm["rollout_weight"]))
                self.assertFalse(list(output.rglob("*.pt")))
                self.assertIn("not own-policy online", (output / "summary.txt").read_text())

    def test_end_to_end_two_models_pretrain_once_and_use_disjoint_validation_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, "temporal_straightening")
            args = arguments(["--dataset-root", str(root), "--device", "cpu", "--expert-updates", "2",
                              "--fit-updates", "2", "--output", str(root / "reports")])

            def config(name, args):
                value = tiny_config(name, "cartpole_balance_sparse")
                value.env.dataset_root = str(root)
                value.env.time_limit = 256
                value.jepa_model.planner.horizon = 5
                return value

            def collect(config, settings):
                values = cases()
                values[0].update(seed=settings.sim_seeds[0], id=str(settings.sim_seeds[0]),
                                 anchor_agent_step=2, rollin_policy="zero")
                return values

            with redirect_stdout(io.StringIO()), patch("scripts.train_rollout_check.arguments", return_value=args), \
                 patch("scripts.train_rollout_check.build_config", side_effect=config), \
                 patch("scripts.train_rollout_check.collect_cases", side_effect=collect) as collected, \
                 patch("torch.set_num_interop_threads"), \
                 patch("torch.save", side_effect=AssertionError("No checkpoint writes")), \
                 patch("torch.load", side_effect=AssertionError("No checkpoint reads")):
                status = main()
            self.assertEqual(status, 0, (args.output / "summary.txt").read_text())
            self.assertEqual(collected.call_count, 2)
            report = json.loads((args.output / "report.json").read_text())
            self.assertNotEqual(report["cases"]["train"][0]["seed"], report["cases"]["validation"][0]["seed"])
            self.assertEqual(len(report["runs"]), 2)
            for run in report["runs"]:
                self.assertEqual(run["status"], "COMPLETE")
                self.assertEqual(run["offline"]["updates"], 2)
                self.assertEqual(len(run["arms"]), 2)
                self.assertTrue(all(a["initial_state_sha256"] == run["offline"]["state_sha256"] for a in run["arms"]))
            self.assertFalse(list(args.output.rglob("*.pt")))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
