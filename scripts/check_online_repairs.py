"""CPU checks for planner invariance, independent replay budgets, and diagnostic isolation."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from models.planning import LatentPlanner
from models.shared.physical_state import PhysicalStateHead
from scripts.check_online_checkpoint_smoke import ToyEnvironment, fixture
from scripts.check_state_normalization import settings, tiny_config
from scripts.online_validation import (
    TrajectoryDataset,
    calibrate_readout,
    collect_episode,
)
from scripts.smoke_models import synthetic_batch
from scripts.smoke_online_checkpoints import run_case
from training import load_model_family
from training.planning import OnlineSession
from training.protocol import checkpoint_compatibility, validate_checkpoint
from training.readout import native_mixture, online_readout


class LongToyEnvironment(ToyEnvironment):
    def step(self, action):
        observation, reward, _ = super().step(action)
        return observation, reward, torch.tensor([self.steps == 128])


class OnlineRepairsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def test_candidate_actions_are_invariant_to_environment_restart_and_chunk_counts(self):
        reference = None
        for environments, restarts, chunk in ((1, 1, 1), (5, 3, 4), (50, 7, 256), (5, 3, 99)):
            model = SimpleNamespace(
                device=torch.device("cpu"), action_dim=1, _gradient_actions=None,
                planner=SimpleNamespace(samples=restarts, horizon=2, iterations=4, lr=.1, action_noise=0,
                                        gradient_batch_size=chunk),
                encode=lambda observation: observation["image"],
                _first_mask=lambda first, batch: torch.ones(batch, dtype=torch.bool),
                _goal_cost=lambda history, past, actions, goal: 1e-9 * (actions - .7).square().sum((-1, -2)),
            )
            # Identical initial candidates, including when the overall random tensor shape changes.
            with patch("torch.randn", side_effect=lambda *shape, **kwargs: torch.full(shape, .2, **kwargs)):
                action = LatentPlanner._gradient_plan(model, {"image": torch.ones(environments, 3, 1)},
                                                     torch.zeros(environments, 2, 1), True, None,
                                                     torch.zeros(environments, 1))
            if reference is None:
                reference = action[0]
            torch.testing.assert_close(action, reference.expand_as(action), atol=1e-7, rtol=0)

    def test_native_goal_gradient_uses_only_terminal_latent_and_detached_goal(self):
        for reduction, expected in (("sum", .08), ("mean", .04)):
            model = SimpleNamespace(
                goal_reduction=reduction, planner={"objective": "last"},
                rollout=lambda history, past, actions: actions,
            )
            for name in ("planning_cost", "_aggregate_goal_weight"):
                setattr(model, name, getattr(LatentPlanner, name).__get__(model))
            actions = torch.full((2, 3, 4, 2), .2, requires_grad=True)
            goal = torch.zeros(2, 2, requires_grad=True)
            cost = LatentPlanner._goal_cost(model, torch.zeros(2, 1, 2), None, actions, goal)
            torch.testing.assert_close(cost, torch.full((2, 3), expected))
            gradient = torch.autograd.grad(cost.sum(), actions)[0]
            self.assertTrue((gradient[:, :, -1:] > 0).all())
            self.assertTrue((gradient[:, :, :-1] == 0).all())
            self.assertIsNone(goal.grad)

    def test_mixed_native_forward_keeps_online_readout_input_and_total_budget(self):
        config = tiny_config("leworldmodel", "cartpole_balance_sparse")
        config.replay.batch_size, config.replay.episodes_per_batch = 8, 4
        captured = []
        def update(batch, **kwargs):
            captured.append((batch, kwargs["readout_batch"]))
            return {}
        expert_obs = {"image": torch.full((4, 4, 2, 2, 3), 100, dtype=torch.uint8),
                      "physical_state": torch.full((4, 4, 5), 100.)}
        expert_batch = (expert_obs, *(torch.full((4, 3, 1), 100.) for _ in range(3)))
        model = SimpleNamespace(sequence_length=4, update=update,
                                _online_expert_replay=SimpleNamespace(batch_size=4, episodes_per_batch=2,
                                                                     sample_episode_batch=lambda: expert_batch))
        session = OnlineSession(config, model, None)
        session.replay.start(4)
        for step in range(6):
            session.replay.append({"image": torch.full((4, 2, 2, 3), step, dtype=torch.uint8),
                                   "physical_state": torch.zeros(4, 5)}, torch.zeros(4, 1),
                                  torch.zeros(4, 1), torch.zeros(4), torch.zeros(4, dtype=torch.bool))
        with patch.object(session.replay, "sample_groups", wraps=session.replay.sample_groups) as groups:
            metrics = session.update(1)
        groups.assert_called_once_with(4, 4, 2)
        self.assertEqual(len(captured), 1)
        mixed, readout = captured[0]
        self.assertEqual(mixed[0]["image"].shape[0], 8)
        self.assertEqual(readout[0]["image"].shape[0], 4)
        self.assertTrue((readout[0]["image"] < 100).all())
        self.assertTrue((mixed[0]["image"][4:] == 100).all())
        self.assertEqual(metrics["native/expert_sequences"], 4)
        self.assertEqual(metrics["native/online_sequences"], 4)

    def test_mixture_rejects_fractional_episode_or_reduced_readout_budgets(self):
        config = tiny_config("leworldmodel", "cartpole_balance_sparse")
        config.replay.batch_size, config.replay.episodes_per_batch = 336, 16
        config.state_head.samples_per_update = 256
        config.training.online.expert_fraction = .5
        self.assertEqual(native_mixture(config), (168, 8))
        for fraction in (.1, .9, 1., float("nan")):
            config.training.online.expert_fraction = fraction
            with self.assertRaises(ValueError):
                native_mixture(config)

    def test_both_real_models_use_one_native_forward_and_independent_head_labels(self):
        for name in ("leworldmodel", "temporal_straightening"):
            with self.subTest(model=name):
                config = tiny_config(name, "cartpole_balance_sparse")
                config.state_head.samples_per_update = 8
                model = load_model_family(name).build_model(config)
                batch, _, _ = synthetic_batch(config, model, batch_size=8)
                def subset(start, stop):
                    return ({key: value[start:stop] for key, value in batch[0].items()},
                            *(value[start:stop] for value in batch[1:]))
                online, expert = subset(0, 4), subset(4, 8)
                model.state_head.configure_online(lambda: model.readout_features(expert))
                calls = []
                hook = model.encoder.register_forward_pre_hook(
                    lambda module, args: calls.append((module.training, next(iter(args[0].values())).shape[0])))
                metrics = model.update(batch, readout_batch=online)
                hook.remove()
                self.assertEqual([size for training, size in calls if training], [8])
                self.assertEqual([size for training, size in calls if not training], [4, 4])
                self.assertEqual(metrics["state/expert_examples"], 4)
                self.assertEqual(metrics["state/examples"], 8)
                self.assertEqual(model.state_head.updates.item(), 1)

    def test_conditioning_preserves_predictions_derivatives_stats_and_resets_only_head_moments(self):
        head = PhysicalStateHead(8, settings(), history=3)
        head.set_stats([0, 1, 0, 0, 0], [.03, .001, .01, .15, .21])
        features = torch.randn(2, 4, 8, requires_grad=True)
        head.fit(features, torch.randn(2, 4, 5))
        before = head(features)
        derivative = torch.autograd.grad(before.sum(), features)[0]
        stats = head.mean.clone(), head.std.clone()
        head.set_conditioning([.1, 1, 1, 2, 3])
        torch.testing.assert_close(head(features), before, rtol=2e-5, atol=1e-6)
        torch.testing.assert_close(torch.autograd.grad(head(features).sum(), features)[0], derivative, rtol=2e-5, atol=1e-6)
        self.assertFalse(head.optimizer.state)
        for actual, expected in zip((head.mean, head.std), stats):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_native_sampler_resumes_exactly_and_reads_training_split_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, _ = fixture(Path(temporary), "leworldmodel")
            config.training.expert.data_path = str(Path(temporary) / config.scenario.dataset)
            config.replay.batch_size = config.training.expert.batch_size = 4
            config.replay.episodes_per_batch = 2
            config.training.online.expert_fraction = .5
            config.state_head.online.expert_fraction = 0
            family = load_model_family("leworldmodel")
            model = family.build_model(config)
            with online_readout(config, family, model) as readout:
                self.assertIsNone(readout)
                replay = model._online_expert_replay
                self.assertEqual(replay.episodes.tolist(), [0])
                self.assertEqual(replay.batch_size, 2)
                replay.sample_episode_batch()
                state = copy.deepcopy(replay.state_dict())
                expected = replay.sample_episode_batch()
            self.assertIsNone(model._online_expert_replay)
            with online_readout(config, family, model, {"phase": "online", "native_expert_replay_state": state}):
                actual = model._online_expert_replay.sample_episode_batch()
            for key in expected[0]:
                torch.testing.assert_close(actual[0][key], expected[0][key], rtol=0, atol=0)
            for left, right in zip(actual[1:], expected[1:]):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "native expert sampler"):
                with online_readout(config, family, model, {"phase": "online"}):
                    self.fail("Missing sampler state must reject resume")

    def test_validation_collection_preserves_model_rng_and_planner_state(self):
        config = tiny_config("leworldmodel", "cartpole_balance_sparse")
        config.env.time_limit = 16
        model = load_model_family("leworldmodel").build_model(config)
        model.train()
        model.encoder.eval()
        modes = [module.training for module in model.modules()]
        weights = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        sentinel = torch.ones(1)
        model._cem_mean = sentinel
        env = ToyEnvironment()
        with patch("scripts.online_validation.make_envs", return_value=env):
            episode = collect_episode(config, model, 6000000, "random")
        self.assertTrue(env.closed)
        self.assertEqual(episode["agent_steps"], 8)
        self.assertEqual(len(episode["image"]), 9)
        self.assertIs(model._cem_mean, sentinel)
        self.assertEqual(modes, [module.training for module in model.modules()])
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key, value in weights.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            TrajectoryDataset([episode], forbidden_seeds=[6000000])

    def test_frozen_native_calibration_does_not_change_native_weights_bn_or_rng(self):
        config = tiny_config("leworldmodel", "cartpole_balance_sparse")
        model = load_model_family("leworldmodel").build_model(config)
        model.state_head.expert_fraction = 0
        model.state_head.configure_online(None)
        native = {key: value.clone() for key, value in model.state_dict().items() if not key.startswith("state_head.")}
        episodes = [{"image": torch.zeros(8, 64, 64, 3, dtype=torch.uint8), "state": torch.randn(8, 5),
                     "action": torch.zeros(7, 1), "seed": 9, "policy": "random"}]
        rng = torch.get_rng_state().clone()
        report = calibrate_readout(model, episodes, 2, 42)
        self.assertEqual(report["updates"], 2)
        self.assertEqual(report["native_updates"], 0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key, value in native.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_old_expert_and_online_weights_are_evaluable_but_not_new_recipe_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, job = fixture(Path(temporary), "leworldmodel")
            checkpoint = torch.load(job["checkpoint"], weights_only=False)
            # Reproduce a v5 checkpoint written before the optional native setting existed.
            saved = OmegaConf.create(checkpoint["training_config"])
            del saved.training.online["expert_fraction"]
            checkpoint["training_config"] = OmegaConf.to_container(saved, resolve=True)
            checkpoint["compatibility"] = {**checkpoint_compatibility(saved), "recipe_version": 5}
            for phase in ("expert", "online"):
                checkpoint["phase"] = phase
                validate_checkpoint(checkpoint, saved, training=False)
                with self.assertRaisesRegex(ValueError, "recipe_version"):
                    validate_checkpoint(checkpoint, saved, training=True)
            saved.jepa_model.planner.horizon += 1
            with self.assertRaisesRegex(ValueError, "incompatible"):
                validate_checkpoint(checkpoint, saved, training=True)

    def test_extended_smoke_reuses_fixed_validation_and_never_writes_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, job = fixture(Path(temporary), "leworldmodel")
            job.update(native_expert_fractions=[0.0], calibration_updates=2, policy_episodes=1,
                       max_false_success_rate=.2, windows=2)
            before = Path(job["checkpoint"]).read_bytes()
            with (patch("scripts.smoke_online_checkpoints.make_envs", side_effect=lambda *a, **kw: ToyEnvironment()),
                  patch("scripts.online_validation.make_envs", side_effect=lambda *a, **kw: LongToyEnvironment()),
                  patch("torch.save", side_effect=AssertionError("No checkpoint writes"))):
                result = run_case(job)
            self.assertTrue(result.get("execution_passed"), result.get("error"))
            self.assertEqual(len(result["trials"]), 2)
            self.assertEqual(result["trials"][1]["calibration"]["updates"], 2)
            validation_seeds = {episode["seed"] for episode in result["validation_data"]["episodes"]}
            train_seeds = {episode["seed"] for episode in result["trials"][1]["calibration"]["training_episodes"]}
            self.assertFalse(validation_seeds & train_seeds)
            self.assertEqual(Path(job["checkpoint"]).read_bytes(), before)
            for trial in result["trials"]:
                self.assertEqual(trial["online"]["updates"], 2)
                self.assertEqual(len(trial["policy_after"]), 1)
                self.assertIn("random/forecast/h100", trial["final_checks"]["validation_guard"]["checks"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
