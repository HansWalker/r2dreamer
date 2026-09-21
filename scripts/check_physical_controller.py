"""CPU/simulator contracts for isolated physical-controller comparisons."""

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
from omegaconf import OmegaConf

import tools
from scripts.check_fresh_readout import episodes
from scripts.check_online_checkpoint_smoke import ToyEnvironment
from scripts.check_planner_recipe import real_fixture
from scripts.check_state_normalization import tiny_config
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.diagnose_goal_objective import PROFILES, collect_objective_cases
from scripts.diagnose_physical_controller import (
    CartpoleCost, arguments, cache_windows, compare, decode_futures, main,
    physical_controller, physical_errors, physical_labels,
)
from scripts.smoke_models import synthetic_batch
from scripts.train_planner_check import build_config, new_model
from scripts.train_planner_learning_curve import validate_budget
from training import load_model_family
from training.evaluation import StateDataset
from training.trainer import online_update_target


class PhysicalControllerTests(unittest.TestCase):
    def test_defaults_keep_native_training_and_validate_arguments(self):
        args = arguments(["--dataset-root", "/tmp/data", "--device", "cpu"])
        self.assertEqual(args.eval_updates, [5000])
        self.assertEqual(args.head_updates, 2000)
        for name, objective in (("leworldmodel", "last"), ("temporal_straightening", "ts_mpc")):
            config = build_config(name, args)
            validate_budget(config, args)
            self.assertEqual(config.jepa_model.planner.objective, objective)
            self.assertEqual(config.jepa_model.history_size, 3)
            self.assertEqual(online_update_target(config, args.online_steps), 262)
        with patch("sys.stderr", new=io.StringIO()):
            for extra in (["--head-updates", "0"], ["--head-windows", "3"], ["--validation-windows", "5"],
                          ["--head-batch", "0"], ["--online-steps", "-1"], ["--expert-updates", "0"],
                          ["--models", "leworldmodel", "leworldmodel"]):
                with self.assertRaises(SystemExit):
                    arguments(["--dataset-root", "/tmp/data", *extra])

    def test_cost_uses_region_orientation_and_arrival_velocity_not_exact_center(self):
        cost = CartpoleCost()
        state = torch.zeros(2, 6, 4)
        state[1, :, 0] = .2
        self.assertEqual(cost(physical_labels(state)).tolist(), [0., 0.])
        state[0, :, 0] = .5
        self.assertGreater(cost(physical_labels(state))[0], 0)
        state.zero_()
        state[0, :3, 3] = 2
        state[1, -3:, 3] = 2
        values = cost(physical_labels(state))
        self.assertEqual(values[0], 0)
        self.assertGreater(values[1], 0)
        invalid = torch.zeros(1, 5, 5, requires_grad=True)
        loss = cost(invalid).sum()
        self.assertGreater(loss.item(), 1)
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, invalid)[0]).all())
        angles = torch.tensor([3.13, -3.13])
        state = torch.zeros(2, 1, 4)
        state[:, 0, 1] = angles
        torch.testing.assert_close(cost(physical_labels(state))[0], cost(physical_labels(state))[1])

    def test_decode_retains_causal_history_and_tokens(self):
        class ReadLastThree(torch.nn.Module):
            history = 3
            def forward(self, value):
                return value.flatten(2).unfold(1, 3, 1).sum(-1)
        for tokens in (False, True):
            history = torch.arange(6.).reshape(2, 3, 1)
            future = torch.arange(16.).reshape(2, 2, 4, 1).requires_grad_()
            if tokens:
                history, future = history.unsqueeze(-2), future.unsqueeze(-2)
            result = decode_futures(ReadLastThree(), history, future)
            self.assertEqual(result.shape, (2, 2, 4, 1))
            torch.testing.assert_close(result[0, 0, 0], torch.tensor([3.]))
            torch.testing.assert_close(result[0, 0, 1], torch.tensor([3.]))
            self.assertTrue(torch.isfinite(torch.autograd.grad(result.sum(), future)[0]).all())

    def test_forecast_cache_never_reads_real_future_features_and_actions_are_aligned(self):
        class Additive(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.device, self.history_size = torch.device("cpu"), 3
            def rollout(self, history, past, action):
                torch.testing.assert_close(history[:, 1:] - history[:, :-1], past)
                return history[:, None, -1:] + action.cumsum(2)
        model = Additive()
        data = episodes(targets=1, length=8)
        features = []
        for episode in data:
            episode["action"] = torch.arange(1, 8.)[:, None]
            value = torch.cat((torch.zeros(1, 1), episode["action"].cumsum(0)))
            episode["state"] = value.clone()
            features.append(value)
        with patch("scripts.diagnose_physical_controller.encode_pool", return_value=features):
            bank = cache_windows(model, data, 8, 5, 123)
        torch.testing.assert_close(bank["observed"], bank["forecast"], rtol=0, atol=0)
        changed = [value.clone() for value in features]
        for value in changed:
            value[3:] += 100
        with patch("scripts.diagnose_physical_controller.encode_pool", return_value=changed):
            other = cache_windows(model, data, 8, 5, 123)
        torch.testing.assert_close(bank["forecast"], other["forecast"], rtol=0, atol=0)
        self.assertFalse(torch.equal(bank["observed"], other["observed"]))
        self.assertFalse(bank["forecast"].requires_grad)

    def test_cost_override_preserves_action_gradients_and_restores_after_exception(self):
        for name in ("leworldmodel", "temporal_straightening"):
            config = tiny_config(name, "cartpole_balance_sparse")
            model = load_model_family(name).build_model(config)
            # LeWM's native zero-initialized modulation is action-blind before fitting.
            model.update(synthetic_batch(config, model)[0])
            model.eval()
            from scripts.diagnose_fresh_readout import fresh_head
            head = fresh_head(model.state_head, config.state_head, None, 20)
            history = model.encode({"image": torch.zeros(2, 3, 64, 64, 3, dtype=torch.uint8)}).detach()
            past = torch.zeros(2, 2, 1)
            actions = torch.full((2, 2, 3, 1), .1, requires_grad=True)
            method = model._goal_cost.__func__
            modes = [m.training for m in head.modules()]
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with physical_controller(model, head, CartpoleCost()):
                    with patch.object(model.state_head, "forward", side_effect=AssertionError("Evaluation head used")):
                        value = model._goal_cost(history, past, actions, None)
                        gradient = torch.autograd.grad(value.sum(), actions)[0]
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(gradient.abs().sum().item(), 0)
                    self.assertTrue(all(p.grad is None for p in head.parameters()))
                    self.assertTrue(all(not p.requires_grad for p in head.parameters()))
                    raise RuntimeError("injected")
            self.assertIs(model._goal_cost.__func__, method)
            self.assertNotIn("_goal_cost", model.__dict__)
            self.assertEqual(modes, [m.training for m in head.modules()])
            self.assertTrue(all(p.requires_grad for p in head.parameters()))

    def test_error_metrics_include_velocity_wrapped_angles_and_missing_failure_coverage(self):
        config = tiny_config("leworldmodel", "cartpole_balance_sparse")
        head = load_model_family("leworldmodel").build_model(config).state_head
        truth = physical_labels(torch.zeros(2, 5, 4))
        prediction = truth.clone()
        prediction[..., 4] += .5
        scores = physical_errors(head, prediction, truth, torch.tensor([.25, .1]))
        self.assertEqual(scores["rmse"]["1"]["physical_rmse"]["velocity[1]"], .5)
        self.assertIsNone(scores["false_goal_fraction"])
        self.assertEqual(scores["failure_states"], 0)
        self.assertEqual(scores["missed_goal_fraction"], 0)

    def test_full_comparison_preserves_model_optimizers_rng_and_evaluation_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = real_fixture(root)
            args = arguments(["--dataset-root", str(root), "--device", "cpu", "--policy-steps", "2",
                              "--horizon", "3", "--head-updates", "2", "--head-windows", "8",
                              "--validation-windows", "4", "--head-batch", "4"])
            data = episodes(length=9)
            with patch("scripts.diagnose_goal_objective.PROFILES", dict(list(PROFILES.items())[:2])), redirect_stdout(io.StringIO()):
                cases = collect_objective_cases(fixture, SimpleNamespace(
                    sim_seeds=[12000000, 12000001], horizons=[1, 2, 3], candidates=16), history_size=3)
            for name in args.models:
                config = tiny_config(name, args.scenario)
                config.env.dataset_root, config.env.time_limit = str(root), 256
                config.jepa_model.planner.horizon = args.horizon
                OmegaConf.resolve(config)
                with load_model_family(name).build_replay(config) as dataset:
                    model = new_model(config, dataset)
                    model.update(dataset.sample_episode_batch())
                    dataset_eval = StateDataset(dataset.h5, dataset.metadata, config.model_io,
                                                config.state_head.fields, model.state_head.targets)
                    windows = dataset_eval.sample_windows(2, 6, 20, 3, 0, 1)
                    batch = dataset_eval.read_batch(windows, 6)
                    model._cem_mean = torch.ones(1, 3, 1)
                    model._gradient_actions = torch.ones(1, 2, 3, 1)
                    caches = model._cem_mean, model._gradient_actions
                    before = tensor_digest(model.state_dict())
                    optimizers = copy.deepcopy(model.optimizer_state_dict())
                    grads = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
                    modes = [m.training for m in model.modules()]
                    rng = tools.get_rng_state()
                    output = root / name
                    output.mkdir()
                    with redirect_stdout(io.StringIO()):
                        result = compare(config, model, cases, batch, args, output, (data[:3], data[3:]))
                    self.assertEqual(before, tensor_digest(model.state_dict()))
                    torch.testing.assert_close(optimizers, model.optimizer_state_dict(), rtol=0, atol=0)
                    for p, old in zip(model.parameters(), grads):
                        torch.testing.assert_close(p.grad, old, rtol=0, atol=0)
                    self.assertEqual(modes, [m.training for m in model.modules()])
                    self.assertIs(caches[0], model._cem_mean)
                    self.assertIs(caches[1], model._gradient_actions)
                    after = tools.get_rng_state()
                    self.assertEqual(rng["python"], after["python"])
                    np.testing.assert_equal(rng["numpy"], after["numpy"])
                    torch.testing.assert_close(rng["torch"], after["torch"], rtol=0, atol=0)
                    candidate = result["physical_controller"]
                    self.assertEqual(candidate["fitting"]["updates"], 2)
                    self.assertNotEqual(candidate["fitting"]["initial_sha256"], candidate["fitting"]["final_sha256"])
                    self.assertEqual(set(candidate["validation"]["scores"]), {"expert", "zero", "random"})
                    self.assertEqual(candidate["policy"]["initial_state_sha256"], result["policy"]["initial_state_sha256"])
                    self.assertEqual(candidate["policy"]["agent_steps"], 2)
                    train_ids = {r["episode"] for r in candidate["fitting"]["windows"]}
                    val_ids = {r["episode"] for r in candidate["validation"]["windows"]}
                    self.assertFalse(train_ids & val_ids)
                    self.assertEqual(len(candidate["candidates"]["cases"]), 4)
                    json.dumps(result, allow_nan=False)

    def test_cli_offline_online_both_models_collects_pool_once_and_writes_no_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_fixture(root)
            args = arguments(["--dataset-root", str(root), "--output", str(root / "output"), "--device", "cpu",
                              "--expert-updates", "2", "--online-steps", "32", "--policy-steps", "2",
                              "--horizon", "3", "--head-updates", "2", "--head-windows", "8",
                              "--validation-windows", "4", "--head-batch", "4"])
            pool = episodes(length=9)
            created, envs = {}, []
            def config_for(name, args):
                config = tiny_config(name, args.scenario)
                config.env.dataset_root, config.env.time_limit = str(root), 256
                config.state_head.samples_per_update = 2
                config.training.online.steps, config.training.online.updates = 64, 8
                config.training.online.warmup_transitions = 8
                OmegaConf.resolve(config)
                return config
            def create(config, dataset):
                model = new_model(config, dataset)
                created[str(config.model_family)] = model
                return model
            def environment(config):
                env = ToyEnvironment()
                envs.append(env)
                return env
            with patch("scripts.diagnose_physical_controller.arguments", return_value=args), \
                 patch("scripts.diagnose_physical_controller.build_config", side_effect=config_for), \
                 patch("scripts.diagnose_physical_controller.collect_pool", return_value=(pool[:3], pool[3:])) as collect, \
                 patch("scripts.train_planner_learning_curve.new_model", side_effect=create), \
                 patch("scripts.train_planner_learning_curve.make_envs", side_effect=environment), \
                 patch("scripts.diagnose_goal_objective.PROFILES", dict(list(PROFILES.items())[:1])), \
                 patch("torch.set_num_interop_threads"), patch("torch.save", side_effect=AssertionError("Checkpoint write")), \
                 redirect_stdout(io.StringIO()):
                status = main()
            report = json.loads((args.output / "report.json").read_text())
            self.assertEqual(status, 0, [(r["model"], r.get("error")) for r in report["runs"]])
            self.assertEqual(collect.call_count, 1)
            self.assertTrue(all(env.closed for env in envs))
            for run in report["runs"]:
                self.assertEqual(run["offline_updates"], 2)
                self.assertEqual(run["online"]["updates"], 2)
                self.assertEqual(run["online"]["head_updates_before"], 2)
                self.assertEqual(run["online"]["head_updates_after"], 4)
                self.assertEqual(run["online"]["schedule_updates"], 8)
                self.assertEqual([s["phase"] for s in run["snapshots"]], ["offline", "online"])
                self.assertEqual(int(created[run["model"]].state_head.updates), 4)
                self.assertNotIn("_goal_cost", created[run["model"]].__dict__)
            self.assertIn("COMPLETE means execution", (args.output / "summary.txt").read_text())
            self.assertFalse(list(args.output.rglob("*.pt")))


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main()
