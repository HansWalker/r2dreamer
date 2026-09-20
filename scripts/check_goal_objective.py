"""CPU and real-simulator contracts for the true-future goal diagnostic."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from envs.dmc import make_env
from models.shared.physical_state import readout_mode
from scripts.check_online_checkpoint_smoke import fixture
from scripts.check_state_normalization import tiny_config
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.diagnose_goal_objective import (
    aggregate,
    arguments,
    candidate_bank,
    cart_state,
    collect_objective_cases,
    feedback_gain,
    main,
    score_horizon,
    score_objective,
    set_cart_state,
    simulate_case,
)
from scripts.diagnose_planner_oracle import encode_images, simulator_branch
from scripts.smoke_tiny_planners import FAMILIES
from training import load_model_family
from training.protocol import checkpoint_compatibility


def fake_case():
    generator = torch.Generator().manual_seed(3)
    return {"id": "fake", "seed": 12_000_000, "profile": "pole_outside", "cohort": "failure",
            "anchor_state": [0., .12, 0., 0.], "anchor_success": False,
            "action": np.zeros((3, 5, 1), dtype=np.float32),
            "candidate_labels": ["zero", "left", "right"], "tolerance": [.25, .1],
            "image": torch.randint(256, (3, 1, 64, 64, 3), generator=generator, dtype=torch.uint8),
            "goal_image": torch.randint(256, (64, 64, 3), generator=generator, dtype=torch.uint8),
            "rewards": np.array([[0.] * 5, [1.] * 5, [2.] * 5]),
            "successes": np.array([[False] * 5, [False] * 5, [True] * 5]),
            "states": np.zeros((3, 1, 4))}


class GoalObjectiveTest(unittest.TestCase):
    def test_upstream_input_dropout_is_separate_from_transformer_dropout(self):
        for family in FAMILIES:
            with self.subTest(model=family):
                config = tiny_config(family, "cartpole_balance_sparse")
                model = load_model_family(family).build_model(config)
                self.assertEqual(model.predictor.dropout.p, 0.)
                value = torch.randn(2, 3, 16)
                model.predictor.train()
                torch.testing.assert_close(model.predictor.dropout(value), value, rtol=0, atol=0)
                for block in model.predictor.blocks:
                    self.assertEqual(block.attention.dropout, .1)
                    self.assertEqual(block.attention.to_out[-1].p, .1)
                    self.assertEqual(block.feed_forward.net[3].p, .1)
                    self.assertEqual(block.feed_forward.net[-1].p, .1)

    def test_legacy_configs_keep_their_behavior_and_weights_costs_remain_compatible(self):
        for family in FAMILIES:
            with self.subTest(model=family):
                config = tiny_config(family, "cartpole_balance_sparse")
                legacy = copy.deepcopy(config)
                del legacy.jepa_model.predictor.emb_dropout
                corrected = load_model_family(family).build_model(config)
                original = load_model_family(family).build_model(legacy)
                self.assertEqual(original.predictor.dropout.p, .1)
                original.load_state_dict(corrected.state_dict(), strict=True)
                self.assertEqual(tensor_digest(original.state_dict()), tensor_digest(corrected.state_dict()))
                self.assertNotEqual(checkpoint_compatibility(config), checkpoint_compatibility(legacy))
                case = fake_case()
                self.assertEqual(score_objective(original, [case], [5], 2, 2),
                                 score_objective(corrected, [case], [5], 2, 2))
                with readout_mode(corrected), readout_mode(original):
                    image = case["image"][:2, 0, None].expand(-1, 3, -1, -1, -1)
                    latent = corrected.encode({"image": image})
                    action = torch.randn(2, 3, corrected.action_dim)
                    torch.testing.assert_close(corrected.predict(latent, action), original.predict(latent, action), rtol=0, atol=0)
                    candidates = action[:, None, :2]
                    goal = latent[:, -1]
                    torch.testing.assert_close(corrected._goal_cost(latent, action[:, :2], candidates, goal),
                                               original._goal_cost(latent, action[:, :2], candidates, goal), rtol=0, atol=0)

    def test_cli_rejects_invalid_or_repeated_work(self):
        args = arguments(["--dataset-root", "/tmp/expert"])
        self.assertEqual(args.models, list(FAMILIES))
        self.assertEqual((args.expert_updates, args.horizons, args.candidates), (1000, [5, 25, 100], 21))
        with patch("sys.stderr", new=io.StringIO()):
            for extra in (("--expert-updates", "0"), ("--candidates", "12"), ("--horizons", "5", "5"),
                          ("--sim-seeds", "-1"), ("--models", "leworldmodel", "leworldmodel")):
                with self.assertRaises(SystemExit):
                    arguments(["--dataset-root", "/tmp/expert", *extra])

    def test_candidates_are_reproducible_bounded_and_include_neutral_opposing_and_feedback(self):
        first, labels = candidate_bank(21, 50, 93)
        second, _ = candidate_bank(21, 50, 93)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (21, 50, 1))
        self.assertLessEqual(abs(first).max(), 1.)
        self.assertEqual(len(set(labels)), 21)
        np.testing.assert_array_equal(first[0], np.zeros((50, 1)))
        np.testing.assert_array_equal(first[1], -first[2])
        self.assertEqual(labels[-1], "simulator_feedback")

    def test_reward_ranking_detects_objective_failure_without_a_predictor(self):
        case = fake_case()
        correct = score_horizon(case, np.array([2., 1., 0.]), 5, 0, 2)
        wrong = score_horizon(case, np.array([0., 1., 2.]), 5, 0, 2)
        tied = score_horizon(case, np.zeros(3), 5, 0, 2)
        self.assertTrue(correct["reward_informative"])
        self.assertTrue(correct["recovery_available"])
        self.assertEqual(correct["selection"]["latent"]["regret_fraction"], 0.)
        self.assertEqual(wrong["selection"]["latent"]["regret_fraction"], 1.)
        self.assertEqual(correct["selection"]["latent"]["sustained_rate"], 1.)
        self.assertEqual(wrong["selection"]["latent"]["sustained_rate"], 0.)
        self.assertEqual(tied["selection"]["latent"]["regret_fraction"], .5)
        self.assertEqual(tied["selection"]["latent"]["sustained_rate"], 1 / 3)
        self.assertEqual(tied["selection"]["latent"]["tied_candidates"], [0, 1, 2])

    def test_flat_rewards_and_unrecoverable_anchors_cannot_be_reported_as_evidence(self):
        case = fake_case()
        case["rewards"][:] = 0
        case["successes"][:] = False
        row = score_horizon(case, np.array([2., 1., 0.]), 5, 0, 2)
        result = aggregate([row])
        self.assertFalse(row["reward_informative"])
        self.assertFalse(row["recovery_available"])
        self.assertEqual(result["coverage"], "LOW_CONTRAST")
        self.assertEqual(result["recoverable_failures"], 0)
        self.assertIsNone(result["selection"]["latent"]["recovery_rate"])
        self.assertIsNone(result["selection"]["latent"]["normalized_return"])
        case["rewards"][2, 0] = 1
        self.assertFalse(score_horizon(case, np.zeros(3), 5, 0, 2)["reward_informative"])
        case["anchor_success"] = True
        case["successes"][2] = True
        self.assertFalse(score_horizon(case, np.zeros(3), 5, 0, 2)["recovery_available"])

    def test_reward_alignment_and_sustained_success_use_the_requested_horizon(self):
        case = fake_case()
        case["states"] = np.zeros((3, 2, 4))
        case["rewards"] = np.tile([0., 0., 2., 2., 2.], (3, 1))
        case["successes"] = case["rewards"] == 2
        early = score_horizon(case, np.zeros(3), 2, 0, 2)
        late = score_horizon(case, np.zeros(3), 5, 1, 2)
        self.assertEqual(early["returns"], [0.] * 3)
        self.assertEqual(late["returns"], [6.] * 3)
        self.assertFalse(any(late["sustained"]))
        self.assertEqual(late["selection"]["latent"]["terminal_success_rate"], 1.)
        case["rewards"] = np.tile([0.] * 10 + [2.] * 10, (3, 1))
        case["successes"] = case["rewards"] == 2
        recovered = score_horizon(case, np.zeros(3), 20, 1, 2)
        self.assertTrue(recovered["recovery_available"])
        self.assertEqual(recovered["tail_agent_steps"], 10)
        self.assertEqual(recovered["selection"]["latent"]["sustained_rate"], 1.)

    def test_native_cost_parity_without_predictor_head_planner_or_model_mutation(self):
        case = fake_case()
        for family in FAMILIES:
            with self.subTest(model=family):
                model = load_model_family(family).build_model(tiny_config(family, "cartpole_balance_sparse"))
                before = tensor_digest(model.state_dict())
                modes = [module.training for module in model.modules()]
                with patch.object(model, "rollout", side_effect=AssertionError("No learned dynamics")), \
                     patch.object(model, "predict", side_effect=AssertionError("No predictor")), \
                     patch.object(model, "act", side_effect=AssertionError("No planner optimization")), \
                     patch.object(model.state_head, "forward", side_effect=AssertionError("No physical head")):
                    rows = score_objective(model, [case], [5], 2, 2)
                    modified = copy.deepcopy(case)
                    modified["rewards"][:] = 0
                    modified["states"][:] = 123
                    changed = score_objective(model, [modified], [5], 2, 2)
                self.assertEqual(rows[0]["latent_cost"], changed[0]["latent_cost"])
                self.assertEqual(before, tensor_digest(model.state_dict()))
                self.assertEqual(modes, [module.training for module in model.modules()])
                with readout_mode(model):
                    actual = encode_images(model, case["image"][:, 0], 2)
                    goal = encode_images(model, case["goal_image"][None], 2)
                    with patch.object(model, "rollout", return_value=actual[None, :, None]):
                        expected = model._goal_cost(None, None, None, goal)[0]
                torch.testing.assert_close(torch.tensor(rows[0]["latent_cost"]), expected, rtol=1e-6, atol=1e-6)

    def test_end_to_end_pretrains_once_per_model_and_collects_shared_cases_once(self):
        self._end_to_end(compare=False)

    def test_paired_end_to_end_matches_initialization_and_windows_without_checkpoints(self):
        self._end_to_end(compare=True)

    def _end_to_end(self, compare):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, "temporal_straightening")
            args = arguments(["--dataset-root", str(root), "--expert-updates", "2", "--device", "cpu",
                              "--horizons", "5", "--output", str(root / "output"),
                              *(["--compare-embedding-dropout"] if compare else [])])

            def config(name, args):
                result = tiny_config(name, args.scenario)
                result.env.dataset_root = str(root)
                result.env.time_limit = 256
                return result

            with redirect_stdout(io.StringIO()), patch("scripts.diagnose_goal_objective.arguments", return_value=args), \
                 patch("scripts.diagnose_goal_objective.build_config", side_effect=config), \
                 patch("scripts.diagnose_goal_objective.collect_objective_cases", return_value=[fake_case()]) as collect, \
                 patch("torch.set_num_interop_threads"), \
                 patch("torch.save", side_effect=AssertionError("No checkpoints")), \
                 patch("torch.load", side_effect=AssertionError("No checkpoints")):
                status = main()
            self.assertEqual(status, 0, (args.output / "summary.txt").read_text())
            self.assertEqual(collect.call_count, 1)
            report = json.loads((args.output / "report.json").read_text())
            self.assertEqual(len(report["runs"]), 4 if compare else 2)
            for result in report["runs"]:
                self.assertEqual(result["status"], "COMPLETE")
                self.assertTrue(result["model_unchanged_during_scoring"])
                self.assertEqual(result["offline"]["updates"], 2)
                self.assertEqual(result["summary"]["5"]["coverage"], "LOW_CONTRAST")
            self.assertFalse(list(args.output.rglob("*.pt")))
            if compare:
                for name in FAMILIES:
                    old, new = [run for run in report["runs"] if run["model"] == name]
                    self.assertEqual(old["offline"]["initial_state_sha256"], new["offline"]["initial_state_sha256"])
                    self.assertEqual(old["sampler_sha256_after_training"], new["sampler_sha256_after_training"])
                    self.assertNotEqual(old["offline"]["state_sha256"], new["offline"]["state_sha256"])
                    self.assertEqual(old["config"]["jepa_model"]["predictor"]["emb_dropout"], .1)
                    self.assertEqual(new["config"]["jepa_model"]["predictor"]["emb_dropout"], 0.)


class SimulatorGoalObjectiveTest(unittest.TestCase):
    def test_default_bank_has_reward_contrast_and_multiple_recoverable_failure_cases(self):
        config = tiny_config("leworldmodel", "cartpole_balance_sparse")
        config.env.time_limit = 256
        args = arguments(["--dataset-root", "/tmp", "--device", "cpu"])
        # Coverage depends only on physics/rewards, not pixels; real rendering is
        # exercised separately by the exact branch-replay test below.
        with redirect_stdout(io.StringIO()), patch("envs.dmc.DeepMindControl.render", return_value=np.zeros((64, 64, 3), np.uint8)):
            cases = collect_objective_cases(config, args)
        for index, horizon in enumerate(args.horizons):
            rows = [score_horizon(c, np.zeros(len(c["action"])), horizon, index, int(config.env.action_repeat)) for c in cases]
            summary = aggregate(rows)
            self.assertGreaterEqual(summary["reward_informative"], 4, (horizon, summary))
            if horizon == 100:
                self.assertGreaterEqual(summary["recoverable_failures"], 4, summary)
                self.assertLess(summary["recoverable_failures"], summary["failure_cases"])

    def test_real_branches_feedback_and_rewards_replay_exactly_without_changing_parent(self):
        config = tiny_config("leworldmodel", "cartpole_balance_sparse")
        config.env.time_limit = 256
        env = make_env(config.env, 93)
        try:
            env.reset()
            gain = feedback_gain(env)
            set_cart_state(env, np.array([.02, .12, 0., .2]))
            before = env._env.physics.get_state().copy()
            actions, _ = candidate_bank(13, 5, 91)
            result = simulate_case(env, actions, [1, 5], gain)
            np.testing.assert_array_equal(before, env._env.physics.get_state())
            self.assertEqual(env._episode_step, 0)
            self.assertEqual(result["image"].shape[:2], (13, 2))
            self.assertGreater(abs(actions[-1]).max(), 0)
            for index in (1, 12):
                with simulator_branch(env) as branch:
                    for step, action in enumerate(actions[index]):
                        observation, reward, _, _ = branch.step(action)
                        self.assertEqual(result["rewards"][index, step], float(reward))
                        if step in (0, 4):
                            slot = 0 if step == 0 else 1
                            np.testing.assert_array_equal(observation["image"], result["image"][index, slot].numpy())
                            np.testing.assert_array_equal(cart_state(branch), result["states"][index, slot])
        finally:
            env.close()

    def test_collection_has_both_success_and_failure_anchors_with_no_state_leak_between_cases(self):
        config = tiny_config("leworldmodel", "cartpole_balance_sparse")
        config.env.time_limit = 256
        args = SimpleNamespace(horizons=[5], candidates=13, sim_seeds=[12_000_000])
        with redirect_stdout(io.StringIO()):
            cases = collect_objective_cases(config, args)
        self.assertEqual(len(cases), 6)
        self.assertEqual(sum(case["anchor_success"] for case in cases), 3)
        self.assertEqual({case["cohort"] for case in cases}, {"balanced", "boundary", "failure"})
        for case in cases:
            self.assertTrue(np.isfinite(case["rewards"]).all())
            torch.testing.assert_close(case["goal_image"], cases[0]["goal_image"], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
