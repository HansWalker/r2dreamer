"""CPU regression tests; set TS_UPSTREAM_CACHE to test the pinned upstream files offline."""

import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from envs.dmc import make_env
from models.shared.physical_state import readout_mode
from scripts.action_conditioning_support import (
    action_gradients, action_variants, collect_branches, horizon_scores, probe_case, teacher_forced,
)
from scripts.check_planner_recipe import real_fixture
from scripts.check_state_normalization import tiny_config
from scripts.diagnose_action_conditioning import arguments, main
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.planner_recipe_support import NormalizedActionEncoder, heldout_pairs
from scripts.upstream_ts_probe import COMMIT, frozen_precision, parity, sources
from training import load_model_family


def model_config(name, action_dim=2):
    config = tiny_config(name, "cartpole_balance_sparse")
    OmegaConf.resolve(config)
    config.model_io.action.shape = [action_dim]
    return config


class ActionProbeTests(unittest.TestCase):
    def test_cli_rejects_invalid_budgets(self):
        args = arguments(["--dataset-root", "/tmp/data"])
        self.assertEqual(args.expert_updates, 3000)
        with patch("sys.stderr", new=io.StringIO()):
            for extra in (["--pairs", "1"], ["--expert-updates", "0"], ["--stride", "100"],
                          ["--goal-tolerance", "nan", ".01"], ["--models", "leworldmodel", "leworldmodel"]):
                with self.assertRaises(SystemExit):
                    arguments(["--dataset-root", "/tmp/data", *extra])

    def test_matched_actions_win_for_known_dynamics_and_tf_uses_true_history(self):
        history = torch.zeros(1, 3, 1)
        past = torch.zeros(1, 2, 1)
        actions = torch.tensor([-1., 0., .5, 1.])[:, None, None].expand(4, 5, 1)
        future = actions.cumsum(1)
        seen = []
        def predict(state, action):
            seen.append((state.clone(), action.clone()))
            return state + action
        model = SimpleNamespace(history_size=3, predict=predict)
        predictions = {k: teacher_forced(model, history, past, a, future) for k, a in action_variants(actions).items()}
        torch.testing.assert_close(predictions["matched"], future)
        self.assertTrue(torch.equal(seen[0][1][:, :2], past.expand(4, -1, -1)))
        torch.testing.assert_close(seen[2][0][:, -1], future[:, 1])
        # Even the wrong-action arms must receive the SAME observed histories.
        torch.testing.assert_close(seen[7][0], seen[2][0])
        scores = horizon_scores({"teacher_forced": predictions, "recursive": predictions}, future, history, torch.ones(1), "mean")
        for row in scores:
            self.assertEqual(row["teacher_forced"]["matched"]["mse"], 0)
            self.assertGreater(row["teacher_forced"]["shuffled"]["mse"], 0)
            self.assertAlmostEqual(row["teacher_forced"]["response_ratio"], 1.)
            self.assertAlmostEqual(row["teacher_forced"]["goal_cost_range_ratio"], 1.)

    def test_gradients_check_causality_and_finite_difference(self):
        model = SimpleNamespace(use_amp=False, goal_reduction="mean",
                                rollout=lambda history, past, controls: controls.cumsum(2))
        encoded = {"history": torch.zeros(1, 3, 2), "past": torch.zeros(1, 2, 2),
                   "actions": torch.ones(4, 5, 2) * .2, "goal": torch.ones(2)}
        gradients = action_gradients(model, encoded)
        for row in gradients:
            h = row["horizon"]
            self.assertTrue(all(x == 0 for x in row["latent_projection_gradient_by_block"][h:]))
            self.assertTrue(all(x > 0 for x in row["latent_projection_gradient_by_block"][:h]))
        derivative = gradients[0]["directional_derivative"]
        self.assertAlmostEqual(derivative["autograd"], derivative["finite_difference"], places=4)

    def test_precision_restores_on_failure_and_bad_upstream_hash_is_rejected(self):
        model = load_model_family("temporal_straightening").build_model(model_config("temporal_straightening"))
        model.use_amp = True
        flags = [p.requires_grad for p in model.parameters()]
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with frozen_precision(model, False):
                self.assertFalse(any(p.requires_grad for p in model.parameters()))
                raise RuntimeError("injected")
        self.assertTrue(model.use_amp)
        self.assertEqual(flags, [p.requires_grad for p in model.parameters()])
        with tempfile.TemporaryDirectory() as tmp, patch("scripts.upstream_ts_probe.urlopen", side_effect=AssertionError("No network")):
            (Path(tmp) / "vit.py").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                sources(tmp)


class UpstreamParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cache = Path(os.environ.get("TS_UPSTREAM_CACHE", str(Path("local/upstream_ts") / COMMIT)))
        if not (cache / "vit.py").exists():
            raise unittest.SkipTest("Set TS_UPSTREAM_CACHE to the pinned source cache for upstream tests.")
        cls.source = sources(cache)

    def test_actual_upstream_predictor_rollout_and_gradients_match(self):
        model = load_model_family("temporal_straightening").build_model(model_config("temporal_straightening"))
        model.action_encoder = NormalizedActionEncoder(model.action_encoder, [.1, -.1], [.4, .7], "cpu")
        dim = model.predictor.state_dim
        tokens = model.predictor._mask_patches
        history, past, actions = torch.randn(1, 3, tokens, dim), torch.randn(1, 2, 2), torch.randn(1, 1, 5, 2)
        before = tensor_digest(model.state_dict())
        with readout_mode(model):
            result = parity(model, self.source, history, past, actions)
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(before, tensor_digest(model.state_dict()))
        # Catch a real semantic error, not just two invocations of our own code.
        with readout_mode(model), patch.object(model, "rollout", side_effect=lambda *a: torch.zeros(1, 1, 5, tokens, dim) + a[2].sum()):
            result = parity(model, self.source, history, past, actions)
        self.assertEqual(result["status"], "MISMATCH")


class SimulatorActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.config = real_fixture(cls.root)
        env = make_env(cls.config.env, 42)
        try:
            with load_model_family(cls.config.model_family).build_replay(cls.config) as dataset:
                cls.cases, _ = heldout_pairs(dataset, env, stride=2, horizon=5, count=2, seed=42, tolerance=[.001, .001])
            cls.bank = collect_branches(env, cls.cases, 2, [.001, .001], 42)
        finally:
            env.close()
        if len(cls.bank) != 2:
            raise AssertionError("Fixture needs two heldout pairs")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_frozen_probes_do_not_fit_plan_or_decode(self):
        for name in ("temporal_straightening", "leworldmodel"):
            model = load_model_family(name).build_model(model_config(name))
            before = tensor_digest(model.state_dict())
            modes = [m.training for m in model.modules()]
            with patch.object(model, "act", side_effect=AssertionError("No planner")), \
                 patch.object(model, "update", side_effect=AssertionError("No fitting")), \
                 patch.object(model.state_head, "forward", side_effect=AssertionError("No head")):
                result, _ = probe_case(model, self.bank[0], 64, gradients=True)
            self.assertEqual(before, tensor_digest(model.state_dict()))
            self.assertEqual(modes, [m.training for m in model.modules()])
            self.assertFalse(any(m._forward_hooks for m in model.modules()))
            json.dumps(result, allow_nan=False)
            for row in result["precision"]["fp32"]["horizons"]:
                self.assertAlmostEqual(row["teacher_forced"]["zero"]["per_candidate_mse"][1],
                                       row["teacher_forced"]["matched"]["per_candidate_mse"][1], places=6)
            for case in self.bank:
                self.assertIn(case["episode"], (1, 2))
                self.assertEqual(case["images"].shape, (10, 5, 64, 64, 3))
                self.assertLess(case["pose_distance"][0][-1], 1)

    def test_end_to_end_fit_once_no_online_no_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(["--dataset-root", str(self.root), "--device", "cpu", "--models", "leworldmodel",
                              "--pairs", "2", "--minimum-pairs", "1", "--gradient-pairs", "1", "--stride", "2",
                              "--expert-updates", "2", "--fit-updates", "2", "--output", str(Path(temp) / "results")])
            cache = Path(os.environ.get("TS_UPSTREAM_CACHE", str(Path("local/upstream_ts") / COMMIT)))
            if (cache / "vit.py").exists():
                args.models = ["temporal_straightening", "leworldmodel"]
                args.upstream_cache = cache
            def base(name, args):
                config = tiny_config(name, args.scenario)
                config.env.dataset_root = str(self.root)
                config.env.time_limit = 256
                OmegaConf.resolve(config)
                return config
            def configs(base, stride):
                raw, model = copy.deepcopy(base), copy.deepcopy(base)
                raw.replay.sequence_length = 3 * stride + 1
                model.model_io.action.shape = [stride]
                return raw, model
            with redirect_stdout(io.StringIO()), patch("scripts.diagnose_action_conditioning.arguments", return_value=args), \
                 patch("scripts.diagnose_action_conditioning.build_config", side_effect=base), \
                 patch("scripts.diagnose_action_conditioning.reference_configs", side_effect=configs), \
                 patch("scripts.diagnose_action_conditioning.heldout_pairs", return_value=(self.cases, 2)), \
                 patch("scripts.diagnose_action_conditioning.collect_branches", return_value=self.bank), \
                 patch("torch.set_num_interop_threads"), patch("torch.save", side_effect=AssertionError("No checkpoints")):
                status = main()
            report = json.loads((args.output / "report.json").read_text())
            self.assertEqual(status, 0, report["runs"])
            self.assertEqual(len(report["runs"]), len(args.models))
            for run in report["runs"]:
                self.assertEqual(run["offline"]["updates"], 2)
                self.assertEqual(run["offline"]["state_sha256"], run["frozen_state_sha256"])
                if run["model"] == "temporal_straightening":
                    self.assertEqual(run["upstream"]["trained"]["status"], "PASS")
                self.assertEqual(len(run["cases"]), 2)
            self.assertEqual(report["online_updates"], 0)
            self.assertEqual(len(report["branches"]), 2)
            self.assertFalse(list(args.output.rglob("*.pt")))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
