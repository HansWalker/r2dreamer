"""CPU and simulator regression tests for the staged recipe diagnostic."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from envs.dmc import make_env
from scripts.check_online_checkpoint_smoke import fixture
from scripts.check_state_normalization import tiny_config
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.diagnose_goal_objective import cart_state, set_cart_state
from scripts.diagnose_planner_recipe import (
    arguments, main, reference_gate, reference_policy, reference_rankings,
)
from scripts.planner_recipe_support import (
    BlockReplay, NormalizedActionEncoder, action_statistics, block_actions, direct_gradient_plan,
    heldout_pairs, optimizer_arm, physical_cart_state, pose_distance, reference_configs, resize_images,
)
from scripts.train_planner_check import build_config
from training import load_model_family


def real_fixture(root):
    config, _ = fixture(root, "temporal_straightening")
    config.env.dataset_root = str(root)
    env = make_env(config.env, 42, include_physical_state=True)
    path = root / config.scenario.dataset / "data.hdf5"
    try:
        with h5py.File(path, "r+") as h5:
            for ep in range(3):
                env.reset()
                set_cart_state(env, [0., .06, .2, .1])
                def observation():
                    state = cart_state(env)
                    return [state[0], np.cos(state[1]), np.sin(state[1]), state[2], state[3]]
                states, images = [observation()], [env.render()]
                actions = np.random.default_rng(ep).uniform(-.2, .2, (128, 1)).astype(np.float32)
                for action in actions:
                    obs, _, _, _ = env.step(action)
                    states.append(observation())
                    images.append(obs["image"])
                h5["observations"][ep] = np.asarray(states, np.float32)
                h5["images"][ep] = np.stack(images)
                h5["actions"][ep] = actions
    finally:
        env.close()
    return config


class RecipeSupportTest(unittest.TestCase):
    def test_cli_validates_before_work(self):
        args = arguments(["--dataset-root", "/tmp/data"])
        self.assertEqual((args.stage, args.stride, args.reference_updates), ("all", 5, 3000))
        with patch("sys.stderr", new=io.StringIO()):
            for extra in (["--stride", "0"], ["--reference-blocks", "4"], ["--pairs", "3"],
                          ["--goal-tolerance", "nan", ".1"], ["--policy-steps", "498"],
                          ["--models", "leworldmodel", "leworldmodel"]):
                with self.assertRaises(SystemExit):
                    arguments(["--dataset-root", "/tmp/data", *extra])

    def test_action_blocks_preserve_every_action_and_frame_alignment(self):
        actions = torch.arange(30).reshape(1, 15, 2).float()
        images = torch.arange(16, dtype=torch.uint8).reshape(1, 16, 1, 1, 1).expand(-1, -1, 4, 4, 3)
        rewards = torch.ones(1, 15, 1)
        terminal = torch.zeros_like(rewards)
        terminal[:, 7] = 1
        labels = torch.arange(16).reshape(1, 16, 1).float()
        dataset = SimpleNamespace(state_mean=0, state_std=1, sample_episode_batch=lambda: (
            {"image": images, "physical_state": labels}, actions, rewards, terminal))
        obs, a, r, done = BlockReplay(dataset, 5, 4, "cpu").sample_episode_batch()
        torch.testing.assert_close(a, actions.reshape(1, 3, 10))
        self.assertEqual(obs["physical_state"].flatten().tolist(), [0, 5, 10, 15])
        self.assertEqual(obs["image"][0, :, 0, 0, 0].tolist(), [0, 5, 10, 15])
        self.assertEqual(r.flatten().tolist(), [5, 5, 5])
        self.assertEqual(done.flatten().tolist(), [0, 1, 0])
        with self.assertRaises(ValueError):
            block_actions(actions[:, :-1], 5)

    def test_statistics_exclude_heldout_and_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = fixture(root, "temporal_straightening")
            config.env.dataset_root = str(root)
            with h5py.File(root / config.scenario.dataset / "data.hdf5", "r+") as h5:
                h5["actions"][0, :2, 0] = [-.5, .5]
                h5["actions"][0, 2:, 0] = 99
                h5["actions"][1:, :, 0] = 1000
            with load_model_family(config.model_family).build_replay(config) as dataset:
                dataset.lengths[0] = 2
                mean, std, count = action_statistics(dataset)
            np.testing.assert_array_equal(mean, [0])
            np.testing.assert_array_equal(std, [.5])
            self.assertEqual(count, 2)

    def test_reference_shapes_do_not_mutate_production_or_raw_data(self):
        args = arguments(["--dataset-root", "/tmp", "--device", "cpu"])
        for name in ("leworldmodel", "temporal_straightening"):
            base = build_config(name, args)
            before = OmegaConf.to_container(base, resolve=True)
            raw, config = reference_configs(base, 5)
            self.assertEqual(before, OmegaConf.to_container(base, resolve=True))
            self.assertEqual(list(raw.model_io.observations.image), [64, 64, 3])
            self.assertEqual(list(raw.model_io.action.shape), [1])
            self.assertEqual(raw.replay.sequence_length, 16)
            self.assertEqual(list(config.model_io.observations.image), [224, 224, 3])
            self.assertEqual(list(config.model_io.action.shape), [5])
            self.assertEqual(config.jepa_model.predictor.layers, 6)
            self.assertEqual(config.jepa_model.encoder.embedding_dim, 192 if name == "leworldmodel" else 8)
            model = load_model_family(name).build_model(config)
            self.assertEqual(model.action_dim, 5)
            if name == "leworldmodel":
                self.assertEqual(len(model.encoder.blocks), 12)
            else:
                self.assertEqual(model.encoder.num_tokens, 196)

    def test_resize_and_normalization_preserve_dtype_layout_and_optimizer_parameters(self):
        image = torch.full((2, 4, 64, 64, 3), 127, dtype=torch.uint8)
        resized = resize_images(image, 224)
        self.assertEqual(resized.shape, (2, 4, 224, 224, 3))
        self.assertEqual(resized.dtype, torch.uint8)
        self.assertTrue((resized == 127).all())
        encoder = torch.nn.Linear(2, 3)
        wrapped = NormalizedActionEncoder(encoder, [.2, -.4], [.5, 2.], "cpu")
        x = torch.tensor([[.2, -.4], [.7, 1.6]])
        torch.testing.assert_close(wrapped(x), encoder(torch.tensor([[0., 0.], [1., 1.]])))
        self.assertEqual([id(p) for p in encoder.parameters()], [id(p) for p in wrapped.parameters()])

    def test_physical_pose_wraps_angles_and_gate_rejects_trivial_success(self):
        state = physical_cart_state([.1, -1., 0., .2, .3])
        np.testing.assert_allclose(state, [.1, np.pi, .2, .3])
        self.assertLess(pose_distance([0, np.pi - .001], [0, -np.pi + .001], [.01, .01]), 1)
        policies = {"expert": {"success_rate": 1.},
                    "zero": {"success_rate": .8, "minimum_pose_distance_mean": .5},
                    "planner": {"success_rate": .9, "minimum_pose_distance_mean": .4}}
        self.assertEqual(reference_gate(policies, 12, 8), "NO_CONTROL_EVIDENCE")
        policies["zero"]["success_rate"] = .25
        self.assertEqual(reference_gate(policies, 12, 8), "OFFLINE_CONTROL_OBSERVED")
        self.assertEqual(reference_gate(policies, 3, 8), "UNVALIDATED")


class GradientRecipeTest(unittest.TestCase):
    def test_direct_optimizer_has_correct_sign_scale_bounds_and_chunk_independence(self):
        def run(chunk):
            model = SimpleNamespace(device=torch.device("cpu"), action_dim=1, _gradient_actions=None,
                                    planner=SimpleNamespace(horizon=3, iterations=100, gradient_batch_size=chunk),
                                    encode=lambda history: history["image"], _first_mask=lambda first, batch: first)
            seen = []
            def cost(latent, past, candidates, goal):
                seen.append(candidates.detach().clone())
                return (candidates[:, :, -1] - goal[:, None]).square().sum(-1) + (candidates[:, :, 0] - goal[:, None]).square().sum(-1)
            model._goal_cost = cost
            goals = torch.tensor([[-.7], [.6], [2.]])
            action = direct_gradient_plan(model, {"image": torch.zeros(3, 1)}, torch.zeros(3, 2, 1), True,
                                          torch.ones(3, dtype=torch.bool), goals,
                                          mean=np.array([.2], np.float32), std=np.array([.5], np.float32), mode="direct")
            self.assertTrue(all((a.abs() <= 1.000001).all() for a in seen))
            self.assertLess(seen[0].abs().max().item(), 1e-7)
            torch.testing.assert_close(action, torch.tensor([[-.7], [.6], [1.]]), atol=.02, rtol=0)
            return action
        torch.testing.assert_close(run(1), run(3), rtol=0, atol=0)

    def test_arm_restores_caches_settings_method_and_frozen_parameters(self):
        config = tiny_config("temporal_straightening", "cartpole_balance_sparse")
        model = load_model_family(config.model_family).build_model(config)
        before = tensor_digest(model.state_dict())
        original = model.planner
        old = torch.ones(1)
        model._gradient_actions = old
        modes = [m.training for m in model.modules()]
        for name in ("tanh", "direct"):
            with optimizer_arm(model, name, [0.], [.5], iterations=2):
                with patch.object(model.state_head, "forward", side_effect=AssertionError("No physical-head planning")):
                    action = model.act({"image": torch.zeros(1, 3, 64, 64, 3, dtype=torch.uint8),
                                        "goal_image": torch.ones(1, 64, 64, 3, dtype=torch.uint8)},
                                       torch.zeros(1, 2, 1), deterministic=True, first=torch.ones(1, dtype=torch.bool))
                self.assertTrue(torch.isfinite(action).all())
            self.assertEqual(before, tensor_digest(model.state_dict()))
            self.assertEqual(modes, [m.training for m in model.modules()])
            self.assertIs(model.planner, original)
            self.assertIs(model._gradient_actions, old)
            self.assertNotIn("_gradient_plan", model.__dict__)
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with optimizer_arm(model, "direct", [0.], [1.]):
                raise RuntimeError("test failure")
        self.assertIs(model.planner, original)
        self.assertNotIn("_gradient_plan", model.__dict__)


class SimulatorRecipeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.config = real_fixture(cls.root)
        cls.args = arguments(["--dataset-root", str(cls.root), "--device", "cpu", "--pairs", "2",
                              "--minimum-pairs", "1", "--stride", "2", "--reference-blocks", "5",
                              "--goal-tolerance", ".001", ".001", "--reference-updates", "2"])
        raw = copy.deepcopy(cls.config)
        raw.replay.sequence_length = 7
        env = make_env(raw.env, 42)
        try:
            with load_model_family(raw.model_family).build_replay(raw) as dataset:
                cls.cases, _ = heldout_pairs(dataset, env, stride=2, horizon=5, count=2, seed=42, tolerance=[.001, .001])
        finally:
            env.close()
        if len(cls.cases) != 2:
            raise AssertionError("Real fixture did not produce two reproducible pairs")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_heldout_goals_are_nontrivial_disjoint_and_expert_replay_reaches(self):
        config = copy.deepcopy(self.config)
        OmegaConf.resolve(config)
        config.model_io.action.shape = [2]
        model = load_model_family(config.model_family).build_model(config)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
             patch.object(model, "act", side_effect=AssertionError("Expert baseline cannot call planner")):
            result = reference_policy(config, model, self.cases, self.args, Path(temp), "expert", [0., 0.], [1., 1.])
        self.assertEqual(result["success_rate"], 1.)
        self.assertEqual(result["initial_state_sha256"], result["final_state_sha256"])
        for case in self.cases:
            self.assertIn(case["episode"], [1, 2])
            self.assertGreater(case["initial_pose_distance"], 2)
            self.assertLess(case["restoration_max_error"], 1e-3)
            self.assertEqual(case["prefix"].shape, (3, 64, 64, 3))

    def test_control_and_rankings_do_not_fit_or_use_auxiliary_head(self):
        for name in ("leworldmodel", "temporal_straightening"):
            config = tiny_config(name, "cartpole_balance_sparse")
            OmegaConf.resolve(config)
            config.env.time_limit = 256
            config.model_io.action.shape = [2]
            model = load_model_family(name).build_model(config)
            before = tensor_digest(model.state_dict())
            with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
                 patch.object(model, "update", side_effect=AssertionError("No fitting")), \
                 patch.object(model.state_head, "forward", side_effect=AssertionError("No physical head")), \
                 patch("scripts.diagnose_planner_recipe.optimizer_arm", side_effect=lambda *a, **kw: optimizer_arm(*a, **kw, iterations=2)):
                control = reference_policy(config, model, self.cases[:1], self.args, Path(temp), "planner", [0., 0.], [1., 1.])
                rows = reference_rankings(config, model, self.cases[:1], self.args)
            self.assertEqual(before, tensor_digest(model.state_dict()))
            self.assertEqual(control["initial_state_sha256"], before)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]["actual_cost"]), 12)
            self.assertLess(rows[0]["physical_pose_distance"][0], 1)

    def test_end_to_end_reference_records_failures_without_claiming_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = copy.copy(self.args)
            args.output = Path(temporary) / "output"
            args.stage, args.models = "reference", ["leworldmodel"]
            def base(name, args):
                config = tiny_config(name, args.scenario)
                config.env.dataset_root = str(self.root)
                config.env.time_limit = 256
                OmegaConf.resolve(config)
                return config
            def configs(base, stride):
                raw, model = copy.deepcopy(base), copy.deepcopy(base)
                raw.replay.sequence_length = stride * 3 + 1
                model.model_io.action.shape = [stride]
                return raw, model
            with redirect_stdout(io.StringIO()), patch("scripts.diagnose_planner_recipe.arguments", return_value=args), \
                 patch("scripts.diagnose_planner_recipe.build_config", side_effect=base), \
                 patch("scripts.diagnose_planner_recipe.reference_configs", side_effect=configs), \
                 patch("scripts.diagnose_planner_recipe.heldout_pairs", return_value=(self.cases, 2)), \
                 patch("torch.set_num_interop_threads"), \
                 patch("torch.save", side_effect=AssertionError("No checkpoint writes")):
                status = main()
            report = json.loads((args.output / "report.json").read_text())
            self.assertEqual(status, 0, report["runs"])
            run = report["runs"][0]
            self.assertIn(run["status"], ["OFFLINE_CONTROL_OBSERVED", "NO_CONTROL_EVIDENCE", "UNVALIDATED"])
            self.assertEqual(run["offline"]["updates"], 2)
            self.assertEqual(run["policies"]["expert"]["success_rate"], 1.)
            self.assertEqual(report["online_updates"], 0)
            self.assertFalse(list(args.output.rglob("*.pt")))

    def test_end_to_end_optimizer_shares_one_fit_and_no_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = copy.copy(self.args)
            args.output = Path(temporary) / "output"
            args.stage, args.optimizer_updates, args.policy_steps = "optimizer", 2, 2
            def base(name, args):
                config = tiny_config(name, args.scenario)
                config.env.dataset_root = str(self.root)
                config.env.time_limit = 256
                OmegaConf.resolve(config)
                return config
            with redirect_stdout(io.StringIO()), patch("scripts.diagnose_planner_recipe.arguments", return_value=args), \
                 patch("scripts.diagnose_planner_recipe.build_config", side_effect=base), \
                 patch("scripts.diagnose_planner_recipe.PROFILES", {"balanced": ("balanced", [0., 0., 0., 0.])}), \
                 patch("scripts.diagnose_planner_recipe.optimizer_arm", side_effect=lambda *a, **kw: optimizer_arm(*a, **kw, iterations=2)), \
                 patch("torch.set_num_interop_threads"), \
                 patch("torch.save", side_effect=AssertionError("No checkpoint writes")):
                status = main()
            report = json.loads((args.output / "report.json").read_text())
            self.assertEqual(status, 0, report["runs"])
            run = report["runs"][0]
            self.assertEqual(run["offline"]["updates"], 2)
            self.assertEqual([a["name"] for a in run["arms"]], ["current", "tanh", "direct"])
            for arm in run["arms"]:
                self.assertEqual(arm["status"], "COMPLETE")
                self.assertEqual(arm["policy"]["initial_state_sha256"], run["offline"]["state_sha256"])
                self.assertEqual(arm["policy"]["final_state_sha256"], run["offline"]["state_sha256"])
            self.assertFalse(list(args.output.rglob("*.pt")))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
