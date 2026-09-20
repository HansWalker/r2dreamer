"""Standalone native-contract and real-simulator checks; no production-loop hooks."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from envs.dmc import make_env
from models.temporal_straightening.model import TemporalStraightening
from models.shared.physical_state import readout_mode
from scripts.check_state_normalization import tiny_config
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.diagnose_planner_oracle import (
    action_candidates, collect_cases, load_case, main, native_probe, rank_correlation,
    ranking_summary, score_case, selection, simulate_candidates, simulator_branch, write_report,
)
from scripts.smoke_models import synthetic_batch
from training import load_model_family
from training.protocol import checkpoint_compatibility, run_identity, validate_checkpoint


def checkpoint_fixture(config, model):
    return {"experiment_protocol": config.experiment_protocol, "checkpoint_id": "oracle-test",
            "run_identity": run_identity(config), "compatibility": checkpoint_compatibility(config),
            "training_config": OmegaConf.to_container(config, resolve=True),
            "model_state_dict": model.state_dict()}


def patch_curvature_reference(latent):
    """Independent scalar formulation of upstream TS `cos`, including moving-patch masking."""
    values = []
    for sequence in latent:
        for time in range(len(sequence) - 2):
            for patch_index in range(sequence.shape[1]):
                first = sequence[time + 1, patch_index] - sequence[time, patch_index]
                second = sequence[time + 2, patch_index] - sequence[time + 1, patch_index]
                if first.norm() > 1e-6 and second.norm() > 1e-6:
                    values.append(1 - torch.dot(first, second) / (first.norm() * second.norm()))
    return torch.stack(values).mean() if values else latent.new_zeros(())


class NativeContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def test_ts_curvature_is_patchwise_not_weighted_by_other_patches_motion(self):
        latent = torch.tensor([[[[0., 0.], [0., 0.]],
                                [[100., 0.], [1., 0.]],
                                [[200., 0.], [0., 0.]]]], requires_grad=True)
        model = SimpleNamespace(predict=lambda state, action: state, prediction_weight=0., curvature_weight=1., decoder=None)
        loss, _ = TemporalStraightening.representation_loss(model, {}, latent, None)
        self.assertAlmostEqual(loss.item(), 1.)
        flattened = latent.flatten(-2).diff(dim=1)
        legacy = 1 - torch.nn.functional.cosine_similarity(flattened[:, 0], flattened[:, 1]).mean()
        self.assertLess(legacy.item(), .001)

    def test_native_losses_and_gradients_match_independent_reference_formulas(self):
        # Sources: LeWM train.py:lejepa_forward; TS visual_world_model.py:forward/total_curvature.
        # This checks the adapted visual-only objectives, not complete upstream architecture parity.
        for family in ("leworldmodel", "temporal_straightening"):
            with self.subTest(family=family):
                config = tiny_config(family, "cartpole_balance_sparse")
                model = load_model_family(family).build_model(config).eval()
                if family == "leworldmodel":
                    for block in model.predictor.blocks:
                        torch.nn.init.normal_(block.modulation[-1].weight, std=.02)
                batch, _, _ = synthetic_batch(config, model)
                actual = model.encode(batch[0]).detach().requires_grad_()
                reference = actual.detach().clone().requires_grad_()
                params = [p for name, p in model.named_parameters()
                          if p.requires_grad and name.startswith(("predictor.", "pred_projector.", "action_encoder."))]
                torch.manual_seed(38)
                loss, _ = model.representation_loss(batch[0], actual, batch[1])
                gradients = torch.autograd.grad(loss, [actual, *params])
                torch.manual_seed(38)
                predicted = model.predict(reference[:, :-1], batch[1])
                target = reference[:, 1:] if family == "leworldmodel" else reference[:, 1:].detach()
                mse = torch.sum((predicted - target)**2) / predicted.numel()
                if family == "leworldmodel":
                    expected = mse + model.sigreg_weight * model.sigreg(reference.transpose(0, 1))
                else:
                    expected = model.prediction_weight * mse + model.curvature_weight * patch_curvature_reference(reference)
                expected_gradients = torch.autograd.grad(expected, [reference, *params])
                torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
                for actual_grad, reference_grad in zip(gradients, expected_gradients, strict=True):
                    torch.testing.assert_close(actual_grad, reference_grad, rtol=2e-4, atol=2e-5)

    def test_ts_stationary_patches_are_finite_and_excluded(self):
        value = torch.ones(2, 4, 3, 5, requires_grad=True)
        model = SimpleNamespace(predict=lambda state, action: state, prediction_weight=1., curvature_weight=.1, decoder=None)
        loss, metrics = TemporalStraightening.representation_loss(model, {}, value, None)
        self.assertEqual(metrics["curvature_loss"].item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_rollout_matches_manual_recursive_predict_and_is_causal(self):
        for family in ("leworldmodel", "temporal_straightening"):
            with self.subTest(family=family):
                config = tiny_config(family, "cartpole_balance_sparse")
                model = load_model_family(family).build_model(config).eval()
                if family == "leworldmodel":
                    for block in model.predictor.blocks:
                        torch.nn.init.normal_(block.modulation[-1].weight, std=.02)
                batch, _, _ = synthetic_batch(config, model)
                with torch.no_grad():
                    latent = model.encode(batch[0])[:, :model.history_size]
                    past = batch[1][:, :model.history_size - 1]
                    actions = torch.rand(latent.shape[0], 2, 5, model.action_dim) * 2 - 1
                    actual = model.rollout(latent, past, actions)
                    for candidate in range(2):
                        states = latent.clone()
                        controls = torch.cat((past, actions[:, candidate]), dim=1)
                        for step in range(5):
                            expected = model.predict(states, controls[:, step:step + model.history_size])[:, -1]
                            torch.testing.assert_close(actual[:, candidate, step], expected, rtol=1e-4, atol=1e-5)
                            states = torch.cat((states[:, 1:], expected[:, None]), dim=1)
                    original = model.predict(latent, batch[1])
                    perturbed = latent.clone()
                    perturbed[:, -1] += 10
                    changed_actions = batch[1].clone()
                    changed_actions[:, -1] *= -1
                    changed = model.predict(perturbed, changed_actions)
                    torch.testing.assert_close(original[:, :-1], changed[:, :-1], rtol=1e-5, atol=1e-6)

    def test_previous_recipe_cannot_silently_resume_training(self):
        config = tiny_config("temporal_straightening", "cartpole_balance_sparse")
        model = load_model_family("temporal_straightening").build_model(config)
        payload = checkpoint_fixture(config, model)
        validate_checkpoint(payload, config, training=True)
        payload["compatibility"]["recipe_version"] = 8
        validate_checkpoint(payload, config, training=False)
        with self.assertRaisesRegex(ValueError, "recipe_version"):
            validate_checkpoint(payload, config, training=True)

    def test_legacy_weights_require_explicit_goal_override_and_remain_read_only(self):
        config = tiny_config("leworldmodel", "cartpole_balance_sparse")
        config.jepa_model.goal.source = "legacy_physical_head"
        model = load_model_family("leworldmodel").build_model(config)
        payload = checkpoint_fixture(config, model)
        payload["compatibility"]["recipe_version"] = 7
        args = SimpleNamespace(device="cpu", seed=config.seed, latent_goals=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrained.pt"
            torch.save(payload, path)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "--latent-goals"):
                load_case(path, "cartpole_balance_sparse", "leworldmodel", args)
            args.latent_goals = True
            loaded, _, metadata = load_case(path, "cartpole_balance_sparse", "leworldmodel", args)
            self.assertEqual(loaded.env.goal.source, "physical_render_v1")
            self.assertTrue(metadata["legacy_goal_override"])
            self.assertEqual(metadata["checkpoint_recipe"], 7)
            self.assertEqual(metadata["native_probe_recipe"], 9)
            self.assertEqual(before, path.read_bytes())


class RankingTest(unittest.TestCase):
    def test_separates_objective_errors_from_dynamics_errors(self):
        wrong_objective = ranking_summary([0, 1, 2], [0, 1, 2], [0, 5, 10])
        self.assertEqual(wrong_objective["forecast_vs_oracle_cost_rank"], 1.)
        self.assertEqual(wrong_objective["oracle_cost_vs_return_rank"], -1.)
        self.assertEqual(wrong_objective["oracle_latent_selection"]["regret"], 10.)
        wrong_dynamics = ranking_summary([0, 1, 2], [2, 1, 0], [0, 5, 10])
        self.assertEqual(wrong_dynamics["oracle_latent_selection"]["regret"], 0.)
        self.assertEqual(wrong_dynamics["learned_selection"]["regret"], 10.)

    def test_ties_are_not_arbitrarily_counted_as_success(self):
        self.assertIsNone(rank_correlation([0, 0, 0], [1, 2, 3]))
        result = selection([0, 0, 0], [0, 0, 9])
        self.assertEqual(result["return_mean"], 3.)
        self.assertEqual(result["regret"], 6.)
        self.assertFalse(ranking_summary([0, 1], [1, 0], [1, 1])["return_informative"])
        self.assertEqual(rank_correlation([0, 0, 2], [1, 1, 3]), 1.)


class SimulatorOracleTest(unittest.TestCase):
    def test_cli_loads_two_checkpoints_and_collects_simulator_data_only_once(self):
        config = tiny_config("leworldmodel", "cartpole_balance_sparse")
        config.env.time_limit = 200
        config.jepa_model.planner.horizon = 1
        model = load_model_family("leworldmodel").build_model(config)
        payload = checkpoint_fixture(config, model)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "cartpole_balance_sparse" / "leworldmodel" / "default" / f"seed_{config.seed}"
            run.mkdir(parents=True)
            for filename in ("pretrained.pt", "final.pt"):
                torch.save(payload, run / filename)
            original = {path.name: path.read_bytes() for path in run.iterdir()}
            args = SimpleNamespace(run_root=root, scenarios=["cartpole_balance_sparse"], models=["leworldmodel"],
                                   checkpoints=["pretrained.pt", "final.pt"], sim_seeds=[73], rollin_steps=[0],
                                   candidates=3, batch_size=4, device="cpu", seed=config.seed,
                                   latent_goals=False, output=root / "report")
            with patch("scripts.diagnose_planner_oracle.parse_args", return_value=args), \
                 patch("scripts.diagnose_planner_oracle.collect_cases", wraps=collect_cases) as collect:
                self.assertEqual(main(), 0)
                self.assertEqual(collect.call_count, 1)
            report = json.loads((args.output / "report.json").read_text())
            self.assertEqual(len(report["results"]), 2)
            self.assertTrue(all(result["model_unchanged"] for result in report["results"]))
            self.assertEqual(original, {path.name: path.read_bytes() for path in run.iterdir()})

    def test_branches_match_real_steps_preserve_parent_rng_state_and_geometry(self):
        for scenario in ("cartpole_balance_sparse", "reacher", "ball_in_cup"):
            with self.subTest(scenario=scenario):
                config = tiny_config("leworldmodel", scenario)
                env, reference = make_env(config.env, 93), make_env(config.env, 93)
                try:
                    env.reset()
                    reference.reset()
                    actions = action_candidates(5, 3, env.action_space.shape[0], 29)
                    before = env._env.physics.get_state().copy()
                    data = simulate_candidates(env, actions)
                    np.testing.assert_array_equal(before, env._env.physics.get_state())
                    reverse = simulate_candidates(env, actions[::-1].copy())
                    torch.testing.assert_close(data["image"], reverse["image"].flip(0), rtol=0, atol=0)
                    torch.testing.assert_close(data["reward"], reverse["reward"].flip(0), rtol=0, atol=0)
                    for step, action in enumerate(actions[1]):
                        obs, reward, done, _ = env.step(action)
                        ref, ref_reward, ref_done, _ = reference.step(action)
                        np.testing.assert_array_equal(data["image"][1, step], obs["image"])
                        self.assertEqual(float(data["reward"][1, step]), float(reward))
                        np.testing.assert_array_equal(obs["image"], ref["image"])
                        self.assertEqual((reward, done), (ref_reward, ref_done))
                    np.testing.assert_array_equal(env.reset()["image"], reference.reset()["image"])
                    with simulator_branch(env) as branch:
                        branch._env.physics.model.geom_pos[:] += 1
                        self.assertFalse(np.array_equal(branch._env.physics.model.geom_pos, env._env.physics.model.geom_pos))
                finally:
                    env.close()
                    reference.close()

    def test_real_cases_scoring_native_probes_and_report_do_not_fit_or_mutate(self):
        cases = None
        for family in ("leworldmodel", "temporal_straightening"):
            with self.subTest(family=family):
                config = tiny_config(family, "cartpole_balance_sparse")
                config.jepa_model.planner.horizon = 2
                config.env.time_limit = 200
                args = SimpleNamespace(sim_seeds=[73], rollin_steps=[0, 2], candidates=3, batch_size=4)
                if cases is None:
                    cases = collect_cases(config, args)
                model = load_model_family(family).build_model(config)
                before, rng = tensor_digest(model.state_dict()), torch.get_rng_state()
                with patch.object(model.state_head, "forward", side_effect=AssertionError("Head used")), \
                     patch.object(model, "update", side_effect=AssertionError("Optimizer update")):
                    scores = [score_case(model, case, 4) for case in cases]
                    probe = native_probe(model, cases, 4)
                    case = cases[0]
                    with readout_mode(model):
                        prefix = model.encode({"image": case["prefix"][None]})
                        goal = model.encode({"image": case["goal_image"][None, None]})[:, 0]
                        cost = model._goal_cost(prefix, case["past_action"][None], case["action"][None], goal)[0]
                    torch.testing.assert_close(cost, torch.tensor(scores[0]["predicted_cost"]), rtol=1e-4, atol=1e-5)
                self.assertEqual(tensor_digest(model.state_dict()), before)
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertTrue(model.training)
                for item in probe.values():
                    # LeWM AdaLN-zero initially blocks upstream action gradients, legitimately.
                    self.assertGreaterEqual(item["gradient_l2"]["action_encoder"], 0)
                    self.assertGreater(item["gradient_l2"]["predictor"], 0)
                with tempfile.TemporaryDirectory() as directory:
                    result = {"scenario": "cartpole_balance_sparse", "model": family, "checkpoint": "test",
                              "status": "COMPLETE", "cases": scores, "native_probe": probe, "elapsed_seconds": 1}
                    write_report(Path(directory), [result], {}, args)
                    report = json.loads((Path(directory) / "report.json").read_text())
                    self.assertEqual(report["optimizer_updates"], 0)
                    self.assertFalse(report["head_used"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
