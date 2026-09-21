"""CPU/model and real DMC goal tests: MUJOCO_GL=egl python -m scripts.check_physical_goals."""

import copy
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from envs.dmc import make_env, make_envs, make_eval_envs
from models.planning import LatentPlanner
from scripts.check_state_normalization import tiny_config
from scripts.smoke_models import synthetic_batch
from training import load_model_family
from training.planning import OnlineSession, build_context, evaluate
from training.protocol import checkpoint_compatibility, validate_training_recipe

FAMILIES = ("leworldmodel", "temporal_straightening")
SCENARIOS = ("cartpole_balance_sparse", "reacher", "ball_in_cup", "point_mass")


class LatentGoalTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_head_weights_and_labels_do_not_change_native_updates_or_gradients(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                config = tiny_config(family, "cartpole_balance_sparse")
                left = load_model_family(family).build_model(config)
                right = copy.deepcopy(left)
                batch, _, _ = synthetic_batch(config, left)
                other_batch = copy.deepcopy(batch)
                other_batch[0]["physical_state"] += 100
                with torch.no_grad():
                    for parameter in right.state_head.parameters():
                        parameter.normal_()
                head_before = [p.clone() for p in left.state_head.parameters()]
                torch.manual_seed(22)
                left_metrics = left.update(batch)
                left_rng = torch.get_rng_state()
                torch.manual_seed(22)
                right_metrics = right.update(other_batch)
                torch.testing.assert_close(torch.get_rng_state(), left_rng, rtol=0, atol=0)
                self.assertEqual(left_metrics["loss"], right_metrics["loss"])
                self.assertNotEqual(float(left_metrics["state/loss"]), float(right_metrics["state/loss"]))
                for key, value in left.state_dict().items():
                    if not key.startswith("state_head."):
                        torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)
                for (name, a), (_, b) in zip(left.named_parameters(), right.named_parameters()):
                    if not name.startswith("state_head.") and a.grad is not None:
                        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(head_before, left.state_head.parameters())))

    def test_both_planners_ignore_head_and_encode_fresh_goal_without_training_bn(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                config = tiny_config(family, "cartpole_balance_sparse")
                model = load_model_family(family).build_model(config)
                _, obs, actions = synthetic_batch(config, model)
                history = {"image": obs["image"][:, :3], "goal_image": obs["image"][:, -1]}
                past = actions[:, :2]
                model.train()
                buffers = {k: v.clone() for k, v in model.named_buffers()}
                torch.manual_seed(24)
                baseline = model.act(history, past)
                with torch.no_grad():
                    for parameter in model.state_head.parameters():
                        parameter.fill_(float("nan"))
                model._cem_mean = model._gradient_actions = None
                torch.manual_seed(24)
                with patch.object(model.state_head, "forward", side_effect=AssertionError("Readout used for control")), \
                     patch.object(model, "encode", wraps=model.encode) as encode:
                    actual = model.act(history, past)
                torch.testing.assert_close(baseline, actual, rtol=0, atol=0)
                goal_calls = [call for call in encode.call_args_list if call.args[0]["image"].shape[1] == 1]
                self.assertEqual(len(goal_calls), 1)
                torch.testing.assert_close(goal_calls[0].args[0]["image"][:, 0], history["goal_image"])
                self.assertTrue(model.training)
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                for key, value in model.named_buffers():
                    torch.testing.assert_close(value, buffers[key], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "goal_image"):
                    model.act({"image": history["image"]}, past)

    def test_per_environment_goals_and_patch_tokens_have_native_reductions(self):
        for reduction in ("sum", "mean"):
            for shape in ((4,), (3, 4)):
                prediction = torch.randn(2, 5, 7, *shape, requires_grad=True)
                goal = torch.randn(2, *shape, requires_grad=True)
                model = SimpleNamespace(goal_reduction=reduction, planner={"objective": "last"}, rollout=lambda *args: prediction)
                cost = LatentPlanner._goal_cost(model, None, None, None, goal)
                expected = torch.stack([
                    getattr((prediction[b, :, -1] - goal[b]).square().flatten(1), reduction)(1)
                    for b in range(2)
                ])
                torch.testing.assert_close(cost, expected)
                cost.sum().backward()
                self.assertIsNone(goal.grad)
                self.assertTrue((prediction.grad[:, :, :-1] == 0).all())

    def test_context_and_replay_exclude_goal_history_and_physical_labels(self):
        image = torch.zeros(2, 64, 64, 3, dtype=torch.uint8)
        goal = image + 10
        history, past = build_context(
            {"image": image, "goal_image": goal, "physical_state": torch.ones(2, 5)},
            [deque(), deque()], [deque(), deque()], 3, 1,
        )
        self.assertEqual(set(history), {"image", "goal_image"})
        self.assertEqual(history["image"].shape, (2, 3, 64, 64, 3))
        self.assertIs(history["goal_image"], goal)
        self.assertEqual(past.shape, (2, 2, 1))
        self.assertEqual(set(LatentPlanner.replay_observation(history)), {"image"})

    def test_gradient_chunks_keep_distinct_goals_attached_to_the_correct_environment(self):
        baseline = None
        goals = torch.tensor([[-.5], [0.], [.5]])
        for chunk in (1, 4, 99):
            model = SimpleNamespace(
                device=torch.device("cpu"), action_dim=1, _gradient_actions=None,
                planner=SimpleNamespace(samples=3, horizon=2, iterations=5, lr=.1,
                                        action_noise=0., gradient_batch_size=chunk),
                encode=lambda obs: obs["image"],
                _first_mask=lambda *args: torch.ones(3, dtype=torch.bool),
                _goal_cost=lambda history, past, actions, goal: (actions - goal[:, None, None]).square().sum((-1, -2)),
            )
            with patch("torch.randn", side_effect=lambda *shape, **kwargs: torch.zeros(shape, **kwargs)):
                action = LatentPlanner._gradient_plan(
                    model, {"image": torch.zeros(3, 3, 1)}, torch.zeros(3, 2, 1), True, None, goals
                )
            if baseline is None:
                baseline = action
            torch.testing.assert_close(action, baseline, atol=1e-7, rtol=0)
            self.assertLess(float(action[0]), 0)
            self.assertEqual(float(action[1]), 0)
            self.assertGreater(float(action[2]), 0)

    def test_goal_settings_are_versioned_and_required(self):
        config = tiny_config("leworldmodel", "cartpole_balance_sparse")
        self.assertEqual(checkpoint_compatibility(config)["recipe_version"], 9)
        original = checkpoint_compatibility(config)
        config.jepa_model.goal.observation.cart_position = .1
        self.assertNotEqual(original, checkpoint_compatibility(config))
        config.jepa_model.goal.source = "obsolete_physical_head"
        with self.assertRaisesRegex(ValueError, "physical_render_v1"):
            validate_training_recipe(config)


class RenderedGoalTest(unittest.TestCase):
    def test_valid_visible_goals_do_not_change_live_states_images_rewards_or_rng(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                config = tiny_config("leworldmodel", scenario)
                plain = OmegaConf.create(OmegaConf.to_container(config.env, resolve=True))
                plain.goal = None
                live = make_env(config.env, 7, True)
                reference = make_env(plain, 7, True)
                try:
                    for _ in range(2):
                        obs, ref = live.reset(), reference.reset()
                        for key in ref:
                            np.testing.assert_array_equal(obs[key], ref[key])
                        goal = obs["goal_image"]
                        self.assertEqual(goal.shape, (64, 64, 3))
                        self.assertEqual(goal.dtype, np.uint8)
                        self.assertGreater(float(goal.std()), 1)
                        p = live._goal_renderer.physics
                        self.assertAlmostEqual(float(live._env.task.get_reward(p)), 1)
                        np.testing.assert_array_equal(p.data.qvel, 0)
                        # Goal rendering cannot depend on the current robot pose or velocity.
                        probe = live._env.physics.copy(share_model=False)
                        try:
                            probe.data.qpos[:] += .1
                            probe.data.qvel[:] += 1
                            probe.forward()
                            live._goal_renderer._image = None
                            np.testing.assert_array_equal(goal, live._goal_renderer.render(probe))
                        finally:
                            probe.free()
                        for _ in range(3):
                            action = np.full(live.action_space.shape, .1, dtype=np.float32)
                            obs, reward, done, _ = live.step(action)
                            ref, ref_reward, ref_done, _ = reference.step(action)
                            for key in ref:
                                np.testing.assert_allclose(obs[key], ref[key], atol=1e-12, rtol=0)
                            self.assertEqual((reward, done), (ref_reward, ref_done))
                            np.testing.assert_array_equal(obs["goal_image"], goal)
                finally:
                    live.close()
                    reference.close()

    def test_reacher_goals_follow_episode_target_and_match_between_families(self):
        envs = [make_env(tiny_config(family, "reacher").env, 19) for family in FAMILIES]
        try:
            previous = None
            for _ in range(3):
                images = [env.reset()["goal_image"] for env in envs]
                np.testing.assert_array_equal(*images)
                if previous is not None:
                    self.assertFalse(np.array_equal(previous, images[0]))
                previous = images[0].copy()
                for env in envs:
                    p = env._goal_renderer.physics
                    np.testing.assert_allclose(p.finger_to_target(), 0, atol=1e-10)
                    np.testing.assert_array_equal(p.named.model.geom_pos["target"],
                                                  env._env.physics.named.model.geom_pos["target"])
        finally:
            for env in envs:
                env.close()

    def test_non_planning_models_do_not_receive_goal_images(self):
        env = make_env(tiny_config("tdmpc2", "reacher").env, 7)
        try:
            self.assertNotIn("goal_image", env.reset())
            self.assertIsNone(env._goal_renderer)
        finally:
            env.close()

    def test_nonzero_physical_goals_are_honored(self):
        config = tiny_config("leworldmodel", "ball_in_cup")
        config.jepa_model.goal.observation.cup_displacement = [.1, -.05]
        config.jepa_model.goal.observation.ball_to_target = [.01, .01]
        env = make_env(config.env, 7)
        try:
            env.reset()
            physics = env._goal_renderer.physics
            np.testing.assert_allclose(physics.named.data.qpos[["cup_x", "cup_z"]], [.1, -.05])
            np.testing.assert_allclose(physics.ball_to_target(), [.01, .01])
        finally:
            env.close()

    def test_real_parallel_collection_update_reset_and_evaluation(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                config = tiny_config(family, "reacher")
                config.env.env_num = config.env.eval_episode_num = 2
                config.env.time_limit = 8
                config.replay.batch_size, config.replay.episodes_per_batch = 4, 2
                model = load_model_family(family).build_model(config)
                envs, evaluation = make_envs(config.env), make_eval_envs(config.env)
                try:
                    session = OnlineSession(config, model, envs)
                    session.start()
                    goals = session.obs["goal_image"].clone()
                    for _ in range(4):
                        session.collect()
                    self.assertFalse(torch.equal(goals, session.obs["goal_image"]))
                    metrics = session.update(1)
                    self.assertTrue(torch.isfinite(torch.as_tensor(metrics["loss"])))
                    self.assertEqual(metrics["state/updates"], 1)
                    self.assertEqual(set(session.replay._obs_keys), {"image", "physical_state"})
                    result = evaluate(config, model, evaluation)
                    self.assertTrue(np.isfinite(result[0]))
                finally:
                    envs.close()
                    evaluation.close()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
