"""CPU and simulator contracts for the shared native JEPA diagnostic support.

Run: MUJOCO_GL=egl python -m scripts.check_paper_faithful_support
"""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from envs.dmc import make_env
from models.shared.physical_state import readout_mode
from scripts.check_state_normalization import tiny_config
from scripts.diagnose_goal_objective import cart_state
from scripts.diagnose_planner_oracle import simulator_branch
from scripts.paper_faithful_support import (
    BranchReplay, collect_anchor, collect_branch_bank, make_split_manifest,
    score_branches, validate_manifest,
)
from training import load_model_family


def fake_case(spec, *, horizon=5, images=False):
    generator = torch.Generator().manual_seed(int(spec["seed"]))
    candidates, prefix = 6, 3
    actions = torch.linspace(-1, 1, candidates * horizon).reshape(candidates, horizon, 1)
    actions[0] = 0
    states = torch.arange(candidates * horizon * 4).reshape(candidates, horizon, 4).float() / 100
    if images:
        shape = (64, 64, 3)
        first = torch.randint(256, (prefix, *shape), dtype=torch.uint8, generator=generator)
        future = torch.randint(256, (candidates, horizon, *shape), dtype=torch.uint8, generator=generator)
    else:
        shape = (1, 1, 1)
        first = torch.arange(prefix, dtype=torch.uint8).reshape(prefix, *shape)
        future = torch.arange(10, 10 + candidates * horizon, dtype=torch.uint8).reshape(candidates, horizon, *shape)
    return {**spec, "prefix": first, "past_action": torch.tensor([[-.2], [.3]]),
            "prefix_state": torch.arange(12).reshape(3, 4).float() / 100,
            "past_reward": torch.tensor([1., 2.]), "action": actions, "image": future,
            "states": states, "rewards": torch.arange(candidates * horizon).reshape(candidates, horizon).float(),
            "goal_image": torch.zeros(shape, dtype=torch.uint8)}


class PaperFaithfulSupportTest(unittest.TestCase):
    def test_anchor_splits_precede_branches_and_reject_leakage(self):
        manifest = make_split_manifest(6, 3, 3, seed=100)
        self.assertEqual(manifest, make_split_manifest(6, 3, 3, seed=100))
        seeds = [{case["seed"] for case in rows} for rows in manifest.values()]
        self.assertEqual(len(set.union(*seeds)), 12)
        self.assertEqual({c["cohort"] for c in manifest["train"]}, {"balanced", "boundary", "recoverable"})
        corrupted = copy.deepcopy(manifest)
        corrupted["test"][0]["seed"] = corrupted["train"][0]["seed"]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_manifest(corrupted)
        cases = [fake_case(case) for case in manifest["train"]]
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            BranchReplay([*cases, fake_case(manifest["test"][0])], batch_size=4, episodes_per_batch=2)
        with self.assertRaisesRegex(ValueError, "independent"):
            BranchReplay([cases[0], copy.deepcopy(cases[0])], batch_size=4, episodes_per_batch=2)

    def test_sampler_preserves_action_alignment_anchor_diversity_and_restart(self):
        cases = [fake_case(spec) for spec in make_split_manifest(6, 1, 1, seed=100)["train"]]
        replay = BranchReplay(cases, batch_size=12, episodes_per_batch=3, seed=8)
        saved = replay.state_dict()
        obs, action, reward, terminal = replay.sample_training_batch()
        plan = copy.deepcopy(replay.last_plan)
        selected = [item["anchor"] for item in plan]
        self.assertEqual(len(set(selected)), 3)
        self.assertTrue(all(selected.count(anchor) == 4 for anchor in set(selected)))
        by_id = {case["id"]: case for case in cases}
        for index, item in enumerate(plan):
            case = by_id[item["anchor"]]
            start, branch = item["start"], item["branch"]
            frames = torch.cat((case["prefix"], case["image"][branch]))
            controls = torch.cat((case["past_action"], case["action"][branch]))
            rewards = torch.cat((case["past_reward"], case["rewards"][branch]))
            states = torch.cat((case["prefix_state"], case["states"][branch]))[start:start + 4]
            torch.testing.assert_close(obs["image"][index], frames[start:start + 4])
            torch.testing.assert_close(action[index], controls[start:start + 3])
            torch.testing.assert_close(reward[index, :, 0], rewards[start:start + 3])
            torch.testing.assert_close(obs["physical_state"][index, :, 0], states[:, 0])
            torch.testing.assert_close(obs["physical_state"][index, :, 1], states[:, 1].cos())
            torch.testing.assert_close(obs["physical_state"][index, :, 2], states[:, 1].sin())
        self.assertFalse(terminal.any())
        replay.load_state_dict(saved)
        native_obs, native_action = replay.sample()
        self.assertEqual(set(native_obs), {"image"})
        self.assertEqual(replay.last_plan, plan)
        torch.testing.assert_close(native_obs["image"], obs["image"])
        torch.testing.assert_close(native_action, action)
        replay.reset(8)
        replay.sample()
        self.assertEqual(replay.last_plan, plan)

    def test_native_scoring_matches_production_including_ts_prefix_and_preserves_model(self):
        for family in ("leworldmodel", "temporal_straightening"):
            with self.subTest(family=family):
                torch.manual_seed(31)
                config = tiny_config(family, "cartpole_balance_sparse")
                model = load_model_family(family).build_model(config)
                case = fake_case(make_split_manifest(1, 1, 1, seed=20)["validation"][0], images=True)
                model.predictor.eval()  # Preserve mixed module modes, not just the root flag.
                before = copy.deepcopy(model.state_dict())
                modes = [module.training for module in model.modules()]
                random_state = torch.get_rng_state().clone()
                with patch.object(model.state_head, "forward", side_effect=AssertionError("Readout must not be used")):
                    result = score_branches(model, [case], horizons=(1, 3), encode_batch_size=8)
                self.assertFalse(result["physical_head_used"])
                self.assertEqual([module.training for module in model.modules()], modes)
                torch.testing.assert_close(torch.get_rng_state(), random_state, rtol=0, atol=0)
                for name, value in before.items():
                    torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
                with readout_mode(model):
                    history = model.encode({"image": case["prefix"][None]})
                    goal = model.encode({"image": case["goal_image"][None, None]})[:, 0]
                    expected = model._goal_cost(history, case["past_action"][None], case["action"][None, :, :3], goal)
                actual = result["cases"][0]["metrics"]["3"]["predicted_cost"]
                torch.testing.assert_close(torch.tensor(actual), expected[0], rtol=1e-5, atol=1e-5)
                if family == "temporal_straightening":
                    self.assertEqual(result["objective"], "ts_mpc")

    def test_auxiliary_physical_labels_do_not_change_native_parameter_update(self):
        specs = make_split_manifest(2, 1, 1, seed=41)["train"]
        replay = BranchReplay([fake_case(spec, images=True) for spec in specs], batch_size=4, episodes_per_batch=2)
        batch = replay.sample_training_batch()
        for family in ("leworldmodel", "temporal_straightening"):
            with self.subTest(family=family):
                config = tiny_config(family, "cartpole_balance_sparse")
                first = load_model_family(family).build_model(config)
                second = load_model_family(family).build_model(config)
                second.load_state_dict(copy.deepcopy(first.state_dict()))
                changed = copy.deepcopy(batch)
                changed[0]["physical_state"].add_(10)
                torch.manual_seed(73)
                first.update(batch)
                torch.manual_seed(73)
                second.update(changed)
                for name, value in first.state_dict().items():
                    if not name.startswith("state_head."):
                        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)

    def test_scoring_requires_actual_zero_reference_and_separates_informative_anchors(self):
        config = tiny_config("temporal_straightening", "cartpole_balance_sparse")
        model = load_model_family("temporal_straightening").build_model(config)
        specs = make_split_manifest(1, 2, 1, seed=35)["validation"]
        cases = [fake_case(spec, images=True) for spec in specs]
        cases[1]["rewards"].zero_()
        result = score_branches(model, cases, horizons=(3,))
        self.assertEqual(result["aggregate"]["all"]["3"]["anchors"], 2)
        self.assertEqual(result["aggregate_informative"]["all"]["3"]["anchors"], 1)
        self.assertEqual(result["aggregate_informative"]["all"]["3"]["predicted_selected_return"],
                         result["cases"][0]["metrics"]["3"]["predicted_selected_return"])
        empty = result["aggregate_informative"][cases[1]["cohort"]]["3"]
        self.assertEqual(empty["anchors"], 0)
        self.assertIsNone(empty["predicted_selected_return"])
        self.assertEqual(empty["valid_counts"]["predicted_selected_return"], 0)
        # Candidate order and stale display labels must not define the physical reference.
        permuted = copy.deepcopy(cases[0])
        for key in ("action", "image", "rewards"):
            permuted[key] = permuted[key].roll(2, 0)
        reordered = score_branches(model, [permuted], horizons=(3,))["cases"][0]["metrics"]["3"]
        expected = result["cases"][0]["metrics"]["3"]
        self.assertAlmostEqual(reordered["true_action_response_rms"], expected["true_action_response_rms"], places=6)
        self.assertAlmostEqual(reordered["action_response_mse"], expected["action_response_mse"], places=6)
        # A single optimized sequence has no counterfactual simulator zero branch.
        single = copy.deepcopy(cases[0])
        for key in ("action", "image", "rewards"):
            single[key] = single[key][1:2]
        only = score_branches(model, [single], horizons=(3,))["cases"][0]["metrics"]["3"]
        self.assertFalse(only["actual_zero_action_reference_available"])
        for key in ("true_action_response_rms", "action_response_mse", "action_response_ratio"):
            self.assertIsNone(only[key])
        self.assertGreaterEqual(only["predicted_action_response_rms"], 0.)

    def test_real_simulator_branches_restore_parent_and_reproduce_every_frame(self):
        config = tiny_config("temporal_straightening", "cartpole_balance_sparse")
        config.env.time_limit = 1000
        spec = make_split_manifest(1, 1, 1, seed=71)["train"][0]
        env = make_env(config.env, spec["seed"], include_physical_state=False)
        try:
            case = collect_anchor(env, spec, candidates=6, horizon=3)
            self.assertEqual(case["image"].shape, (6, 3, 64, 64, 3))
            self.assertEqual(case["prefix"].shape, (3, 64, 64, 3))
            self.assertEqual(case["past_action"].shape, (2, 1))
            before, step = env._env.physics.get_state().copy(), env._episode_step
            task_rng = copy.deepcopy(env._env.task.random.get_state())
            with simulator_branch(env) as branch:
                for index, action in enumerate(case["action"][2].numpy()):
                    observation, reward, done, _ = branch.step(action)
                    self.assertFalse(done)
                    np.testing.assert_array_equal(observation["image"], case["image"][2, index].numpy())
                    np.testing.assert_allclose(cart_state(branch), case["states"][2, index].numpy(), atol=0, rtol=0)
                    self.assertEqual(float(reward), float(case["rewards"][2, index]))
                branch.reset()
            np.testing.assert_array_equal(before, env._env.physics.get_state())
            self.assertEqual(step, env._episode_step)
            after_rng = env._env.task.random.get_state()
            self.assertEqual(task_rng[0], after_rng[0])
            np.testing.assert_array_equal(task_rng[1], after_rng[1])
            self.assertEqual(task_rng[2:], after_rng[2:])
        finally:
            env.close()

    def test_bank_metadata_hashes_are_reproducible_and_training_sources_exclude_validation(self):
        config = tiny_config("leworldmodel", "cartpole_balance_sparse")
        config.env.time_limit = 1000
        manifest = make_split_manifest(1, 1, 1, seed=91)
        bank = collect_branch_bank(config, manifest, candidates=6, horizon=2)
        self.assertEqual(bank["metadata"]["training_branch_transitions"], 12)
        self.assertEqual(len(bank["sha256"]), 64)
        self.assertEqual(len(set(bank["metadata"]["split_hashes"].values())), 3)
        replay = BranchReplay(bank, batch_size=2, episodes_per_batch=1)
        replay.sample()
        self.assertEqual({row["anchor"] for row in replay.last_plan}, {manifest["train"][0]["id"]})
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            BranchReplay(bank["splits"]["validation"], batch_size=2, episodes_per_batch=1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
