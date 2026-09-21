"""CPU/simulator regression tests for objective corrections and explicit task adaptations."""

import copy
import io
import json
import os
import tempfile
import unittest
from collections import deque
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from envs.dmc import make_env
from models.planning import LatentPlanner
from models.shared.latent_goal import latent_goal_cost
from models.shared.physical_state import readout_mode
from scripts.check_planner_recipe import real_fixture
from scripts.check_state_normalization import tiny_config
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.diagnose_goal_objective import PROFILES
from scripts.diagnose_planner_fixes import arguments, main
from scripts.smoke_models import synthetic_batch
from scripts.upstream_ts_probe import COMMIT, full_precision, sources, training_parity
from training import load_model_family
from training.planning import build_context
from training.protocol import checkpoint_compatibility, validate_training_recipe


class ObjectiveTests(unittest.TestCase):
    def test_native_costs_and_gradients(self):
        for shape in ((4,), (3, 4)):
            history = torch.randn(2, 3, *shape)
            prediction = torch.randn(2, 5, 7, *shape, requires_grad=True)
            goal = torch.randn(2, *shape, requires_grad=True)
            terminal = latent_goal_cost(prediction, goal, reduction="sum")
            torch.testing.assert_close(terminal, (prediction[:, :, -1] - goal[:, None]).square().flatten(2).sum(-1))
            actual = latent_goal_cost(prediction, goal, reduction="mean", mode="ts_mpc", history=history)
            path = torch.cat((history[:, None].expand(-1, 5, -1, *shape), prediction), dim=2)
            weights = 2. ** torch.arange(10)
            expected = ((path - goal[:, None, None]).square().flatten(3).mean(-1) * weights / weights.sum()).mean(-1)
            torch.testing.assert_close(actual, expected)
            a = torch.autograd.grad(actual.sum(), prediction, retain_graph=True)[0]
            b = torch.autograd.grad(expected.sum(), prediction)[0]
            torch.testing.assert_close(a, b)
            self.assertIsNone(goal.grad)

    def test_bank_chooses_one_goal_for_whole_tail_and_supports_singletons(self):
        path = torch.tensor([[[[0.], [1.], [0.]]]], requires_grad=True)
        bank = torch.tensor([[[0.], [1.]]], requires_grad=True)
        cost = latent_goal_cost(path, bank, reduction="mean", mode="tail", tail_steps=3)
        torch.testing.assert_close(cost, torch.tensor([[1 / 3]]))
        cost.sum().backward()
        self.assertIsNone(bank.grad)
        self.assertGreater(path.grad.abs().sum(), 0)
        stable = torch.tensor([[[[0.], [0.], [0.]]]])
        self.assertLess(latent_goal_cost(stable, bank, reduction="mean", mode="tail").item(), cost.item())
        torch.testing.assert_close(latent_goal_cost(path, bank[:, 0], reduction="sum"),
                                   latent_goal_cost(path, bank[:, :1], reduction="sum"))
        with self.assertRaisesRegex(ValueError, "tail_steps"):
            latent_goal_cost(path, bank, reduction="sum", mode="tail", tail_steps=4)

    def test_precision_restores_flags_even_on_error(self):
        before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                  torch.backends.cudnn.allow_tf32, torch.backends.cuda.flash_sdp_enabled())
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with full_precision():
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
                raise RuntimeError("injected")
        self.assertEqual(before, (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                                 torch.backends.cudnn.allow_tf32, torch.backends.cuda.flash_sdp_enabled()))

    def test_configuration_is_explicit_versioned_and_legacy_falls_back_to_terminal(self):
        for name, mode in (("leworldmodel", "last"), ("temporal_straightening", "ts_mpc")):
            config = tiny_config(name, "cartpole_balance_sparse")
            self.assertEqual(config.jepa_model.planner.objective, mode)
            old = checkpoint_compatibility(config)
            config.jepa_model.planner.objective = "tail"
            config.jepa_model.planner.horizon = 5
            self.assertNotEqual(old, checkpoint_compatibility(config))
            validate_training_recipe(config)
            with open_dict(config.jepa_model.planner):
                del config.jepa_model.planner["objective"]
            model = load_model_family(name).build_model(config)
            goal = torch.randn(1, 2)
            prediction = torch.randn(1, 3, 4, 2)
            with patch.object(model, "rollout", return_value=prediction):
                torch.testing.assert_close(model._goal_cost(None, None, None, goal),
                                           latent_goal_cost(prediction, goal, reduction=model.goal_reduction))


class GoalSetTests(unittest.TestCase):
    def test_goal_banks_render_valid_targets_without_live_state_or_replay_changes(self):
        for scenario in ("cartpole_balance_sparse", "reacher", "ball_in_cup"):
            config = tiny_config("leworldmodel", scenario)
            config.jepa_model.goal.alternatives = config.scenario.goal_alternatives
            plain = copy.deepcopy(config.env)
            plain.goal = None
            env, reference = make_env(config.env, 7), make_env(plain, 7)
            try:
                for _ in range(2):
                    obs, original = env.reset(), reference.reset()
                    for key in original:
                        np.testing.assert_array_equal(obs[key], original[key])
                    bank = obs["goal_images"]
                    self.assertEqual(bank.shape, (1 + len(config.scenario.goal_alternatives), 64, 64, 3))
                    self.assertEqual(bank.dtype, np.uint8)
                    self.assertEqual(env.observation_space["goal_images"].shape, bank.shape)
                    self.assertFalse(np.array_equal(bank[0], bank[1]))
                    state = env._env.physics.get_state().copy()
                    for index, spec in enumerate(env._goal_renderer.specs):
                        image = env._goal_renderer._render_one(spec, env._goal_renderer._target)
                        np.testing.assert_array_equal(image, bank[index])
                        self.assertEqual(env._env.task.get_reward(env._goal_renderer.physics), 1)
                        np.testing.assert_array_equal(env._goal_renderer.physics.data.qvel, 0)
                    np.testing.assert_array_equal(env._env.physics.get_state(), state)
                    tensors = {k: torch.as_tensor(np.ascontiguousarray(v))[None] for k, v in obs.items()}
                    context, _ = build_context(tensors, [deque()], [deque()], 3, env.action_space.shape[0])
                    self.assertIn("goal_images", context)
                    self.assertEqual(set(LatentPlanner.replay_observation(context)), {"image"})
            finally:
                env.close()
                reference.close()

    def test_both_planners_use_goal_sets_without_readout(self):
        for name in ("leworldmodel", "temporal_straightening"):
            config = tiny_config(name, "cartpole_balance_sparse")
            model = load_model_family(name).build_model(config)
            _, obs, actions = synthetic_batch(config, model)
            history = {"image": obs["image"][:, :3], "goal_images": obs["image"][:, :2]}
            with patch.object(model.state_head, "forward", side_effect=AssertionError("No head")):
                action = model.act(history, actions[:, :2], deterministic=True)
            self.assertTrue(torch.isfinite(action).all())


class TrainingParityTests(unittest.TestCase):
    def test_upstream_loss_gradients_and_optimizer_step_on_copies(self):
        cache = Path(os.environ.get("TS_UPSTREAM_CACHE", str(Path("local/upstream_ts") / COMMIT)))
        if not (cache / "objectives.py").exists():
            self.skipTest("Set TS_UPSTREAM_CACHE to pinned upstream files.")
        model = load_model_family("temporal_straightening").build_model(tiny_config("temporal_straightening", "cartpole_balance_sparse"))
        batch, _, _ = synthetic_batch(tiny_config("temporal_straightening", "cartpole_balance_sparse"), model)
        model.update(batch)  # Exercise nonempty optimizer moments too.
        with readout_mode(model):
            latent = model.encode(batch[0]).detach()
        before = tensor_digest(model.state_dict())
        state = copy.deepcopy(model.optimizer_state_dict())
        with readout_mode(model):
            result = training_parity(model, sources(cache), latent, batch[1])
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(before, tensor_digest(model.state_dict()))
        torch.testing.assert_close(state, model.optimizer_state_dict(), rtol=0, atol=0)
        # A deliberately wrong coefficient must be caught by the upstream loss check.
        original = type(model).representation_loss
        def wrong(self, *args):
            loss, metrics = original(self, *args)
            return loss * 2, metrics
        with patch.object(type(model), "representation_loss", wrong):
            result = training_parity(model, sources(cache), latent, batch[1])
        self.assertEqual(result["status"], "MISMATCH")


class EndToEndTests(unittest.TestCase):
    def test_small_run_fits_once_and_never_saves_or_uses_head_for_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_fixture(root)
            args = arguments(["--dataset-root", str(root), "--models", "leworldmodel", "--device", "cpu",
                              "--expert-updates", "2", "--fit-updates", "2", "--policy-steps", "2", "--output", str(root / "output")])
            cache = Path(os.environ.get("TS_UPSTREAM_CACHE", str(Path("local/upstream_ts") / COMMIT)))
            if (cache / "objectives.py").exists():
                args.models.append("temporal_straightening")
                args.upstream_cache = cache
            def base(name, args):
                config = tiny_config(name, args.scenario)
                config.env.dataset_root = str(root)
                config.env.time_limit = 256
                OmegaConf.resolve(config)
                return config
            with patch("scripts.diagnose_planner_fixes.arguments", return_value=args), \
                 patch("scripts.diagnose_planner_fixes.build_config", side_effect=base), \
                 patch("scripts.diagnose_goal_objective.PROFILES", dict(list(PROFILES.items())[:1])), \
                 patch("torch.set_num_interop_threads"), patch("torch.save", side_effect=AssertionError("No checkpoints")), \
                 redirect_stdout(io.StringIO()):
                status = main()
            report = json.loads((args.output / "report.json").read_text())
            self.assertEqual(status, 0, report)
            for run in report["runs"]:
                self.assertEqual(run["offline"]["updates"], 2)
                expected = [1, 3, 3] if run["model"] == "leworldmodel" else [1, 1, 3, 3]
                self.assertEqual([a["goal_count"] for a in run["arms"]], expected)
                self.assertEqual(run["fit_control"]["updates"], 2)
                self.assertTrue(run["fit_control"]["weights_changed"])
                self.assertTrue(any(lr > 0 for row in run["fit_control"]["learning_rates"] for lr in row))
                self.assertFalse(set(run["fit_control"]["train_episodes"]) & set(run["fit_control"]["validation_episodes"]))
            self.assertFalse(list(args.output.rglob("*.pt")))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
