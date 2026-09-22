"""Independent CPU/simulator contracts for frozen-checkpoint planning comparisons."""

import copy
import hashlib
import io
import json
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

import tools
from scripts.check_state_normalization import tiny_config
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.paper_faithful_followup_eval import _equal
from scripts.paper_faithful_followup_support import make_followup_manifest, policy_cases
from scripts.paper_faithful_support import BranchReplay, _digest, collect_branch_bank
from scripts.train_paper_faithful_check import save_checkpoint
from training import load_model_family
from training.protocol import implementation_sha256


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fixture(root):
    """Two tiny, actually updated sources and a real disjoint simulator bank."""
    configs = {}
    for arm, family in (("ts_agg_01_coverage", "temporal_straightening"), ("lewm_coverage", "leworldmodel")):
        config = tiny_config(family, "cartpole_balance_sparse")
        config.seed = 1
        config.env.time_limit = 1000
        config.training.expert.updates = 1
        config.jepa_model.planner.horizon = 5
        config.jepa_model.planner.samples = 4
        config.jepa_model.planner.iterations = 1
        config.jepa_model.planner.elites = 2
        with open_dict(config):
            config.jepa_model.curvature_mode = "agg" if family == "temporal_straightening" else "patch"
            config.jepa_model.aggregation = {"hidden_dim": 32, "output_dim": 8}
        OmegaConf.resolve(config)
        configs[arm] = config
    bank = collect_branch_bank(configs["ts_agg_01_coverage"], make_followup_manifest(6, 6, 6, seed=53100), candidates=6, horizon=15)
    torch.save(bank, root / "branches.pt")
    identity = {"task": "cartpole/balance_sparse", "dataset_id": "test-fixture", "collection_sha256": "0" * 64}
    report = {"format": "paper_faithful_followup_v1", "status": "COMPLETE", "online_updates": 0,
              "implementation_sha256": implementation_sha256(),
              "online_schedule_changed": False, "dataset_identity": identity,
              "settings": {"seeds": [1], "profile": "tiny", "forecast_horizons": [1, 5, 15], "planner_horizon": 5},
              "reference": {"status": "NOT_RUN", "reason": "Synthetic test fixture"},
              "fixture_training": "One native update using simulator branch replay only; not a research result.",
              "branches": {"file": "branches.pt", "metadata": bank["metadata"], "sha256": bank["sha256"]},
              "runs": []}
    for arm, config in configs.items():
        torch.manual_seed(1)
        model = load_model_family(config.model_family).build_model(config)
        if hasattr(model, "configure_pretraining"):
            model.configure_pretraining(20)
        replay = BranchReplay(bank, batch_size=4, episodes_per_batch=2, sequence_length=4, seed=3)
        model.train()
        model.update(replay.sample_training_batch())
        checkpoint = root / "seed_1" / arm / "native.pt"
        checkpoint.parent.mkdir(parents=True)
        digest = save_checkpoint(checkpoint, model, config, 1, identity)
        report["runs"].append({"arm": arm, "seed": 1, "family": str(config.model_family), "status": "COMPLETE", "updates": 1,
                               "config": OmegaConf.to_container(config, resolve=True),
                               "checkpoints": [{"updates": 1, "file": str(checkpoint.relative_to(root)), "sha256": digest}]})
    (root / "report.json").write_text(json.dumps(report, indent=2))
    return report, bank


def no_fitting():
    stack = ExitStack()
    for target in ("models.planning.LatentPlanner.update", "models.leworldmodel.model.LeWorldModel.update",
                   "training.planning.expert_update", "training.planning.build_replay", "training.planning.OnlineSession", "torch.save"):
        stack.enter_context(patch(target, side_effect=AssertionError(f"Frozen evaluation called {target}")))
    return stack


def assert_rng(test, before):
    after = tools.get_rng_state()
    test.assertEqual(before["python"], after["python"])
    test.assertEqual(before["numpy"][0], after["numpy"][0])
    np.testing.assert_array_equal(before["numpy"][1], after["numpy"][1])
    test.assertEqual(before["numpy"][2:], after["numpy"][2:])
    test.assertTrue(torch.equal(before["torch"], after["torch"]))


class FrozenPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / "source"
        cls.source.mkdir()
        cls.source_report, cls.bank = source_fixture(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def runner(self):
        from scripts import evaluate_paper_faithful_planning
        return evaluate_paper_faithful_planning

    def argv(self, output):
        return ["--source-run", str(self.source), "--output", str(output), "--device", "cpu", "--seeds", "1",
                "--policy-cases", "1", "--policy-steps", "2", "--encode-batch-size", "8"]

    def test_six_predeclared_conditions_and_default_cli(self):
        runner = self.runner()
        args = runner.arguments(["--source-run", str(self.source)])
        self.assertEqual(args.seeds, [1, 2, 3])
        self.assertEqual(args.horizons, [5, 15])
        self.assertEqual(args.policy_cases, 12)
        self.assertEqual(args.policy_steps, 200)
        rows = runner.conditions()
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({row["name"] for row in rows}), 6)
        expected = {(arm, h, weight) for h in (5, 15) for arm, weight in (
            ("ts_agg_01_coverage", 0.), ("ts_agg_01_coverage", .1), ("lewm_coverage", 0.))}
        self.assertEqual({(r["source_arm"], r["horizon"], r["aggregate_goal_weight"]) for r in rows}, expected)
        with redirect_stderr(io.StringIO()):
            for extra in (["--seeds", "1", "1"], ["--policy-steps", "0"], ["--policy-cases", "0"], ["--horizons", "0"]):
                with self.assertRaises(SystemExit):
                    runner.arguments([*self.argv(self.root / "invalid"), *extra])

    def test_frozen_source_provenance_and_conditions_preserve_every_source_state(self):
        runner = self.runner()
        source_hashes = {str(p.relative_to(self.source)): sha256(p) for p in self.source.rglob("*") if p.is_file()}
        with no_fitting():
            report, bank, records = runner.load_source(self.source, [1], horizons=(5, 15), policy_cases=1, policy_steps=2)
            self.assertEqual(len(records), 2)
            self.assertEqual(report["format"], "paper_faithful_followup_v1")
            by_arm = {record["source_arm"]: record for record in records}
            expected_ids = [c["id"] for c in policy_cases(bank["splits"]["validation"], 1)]
            args = runner.arguments(self.argv(self.root / "direct"))
            for arm, record in by_arm.items():
                self.assertEqual(record["updates"], 1)
                self.assertEqual(record["sha256"], sha256(record["path"]))
                self.assertEqual(record["dataset_identity"], report["dataset_identity"])
                config, model = runner.load_frozen(record, "cpu")
                payload = torch.load(record["path"], map_location="cpu", weights_only=False)
                self.assertEqual(tensor_digest(model.state_dict()), tensor_digest(payload["model_state_dict"]))
                model.train()
                model.encoder.eval()
                model._cem_mean = torch.randn(1, 5, 1)
                model._gradient_actions = torch.randn(1, 4, 5, 1)
                caches = model._cem_mean, model._gradient_actions
                cache_values = [x.clone() for x in caches]
                before_config = OmegaConf.to_container(config, resolve=True)
                before_weights = tensor_digest(model.state_dict())
                before_optimizers = copy.deepcopy(model.optimizer_state_dict())
                before_modes = [m.training for m in model.modules()]
                before_flags = [p.requires_grad for p in model.parameters()]
                results = []
                for condition in (row for row in runner.conditions() if row["source_arm"] == arm):
                    rng = tools.get_rng_state()
                    result = runner.evaluate_condition(config, model, bank["splits"]["validation"], condition, args)
                    results.append((condition, result))
                    assert_rng(self, rng)
                    self.assertEqual(before_config, OmegaConf.to_container(config, resolve=True))
                    self.assertEqual(before_weights, tensor_digest(model.state_dict()))
                    self.assertTrue(_equal(before_optimizers, model.optimizer_state_dict()))
                    self.assertEqual(before_modes, [m.training for m in model.modules()])
                    self.assertEqual(before_flags, [p.requires_grad for p in model.parameters()])
                    self.assertIs(model._cem_mean, caches[0])
                    self.assertIs(model._gradient_actions, caches[1])
                    for actual, original in zip((model._cem_mean, model._gradient_actions), cache_values):
                        self.assertTrue(torch.equal(actual, original))
                    self.assertEqual(result["policy"]["case_ids"], expected_ids)
                    self.assertEqual(result["policy"]["steps"], 2)
                    self.assertEqual(result["policy_planning_horizon"], condition["horizon"])
                    self.assertEqual(result["planner"]["aggregate_goal_weight"], condition["aggregate_goal_weight"])
                    for key, value in before_config["jepa_model"]["planner"].items():
                        if key not in {"horizon", "aggregate_goal_weight"}:
                            self.assertEqual(result["planner"][key], value)
                    self.assertEqual(len(result["policy"]["traces"]), 2)
                if arm == "ts_agg_01_coverage":
                    for horizon in (5, 15):
                        costs = {}
                        for condition, result in results:
                            if condition["horizon"] == horizon:
                                costs[condition["aggregate_goal_weight"]] = np.asarray(
                                    result["branches"]["cases"][0]["metrics"][str(horizon)]["predicted_cost"])
                        self.assertTrue(np.all(costs[.1] >= costs[0.] - 1e-7))
                        self.assertTrue(np.any(costs[.1] > costs[0.] + 1e-7))
                condition, first = results[0]
                repeated = runner.evaluate_condition(config, model, bank["splits"]["validation"], condition, args)
                self.assertEqual(first["policy"]["traces"], repeated["policy"]["traces"])
                planner = model.planner
                with patch.object(runner, "evaluate_snapshot", side_effect=ValueError("Injected evaluation error")):
                    with self.assertRaisesRegex(ValueError, "Injected"):
                        runner.evaluate_condition(config, model, bank["splits"]["validation"], condition, args)
                self.assertIs(model.planner, planner)
                self.assertIs(model._cem_mean, caches[0])
                self.assertIs(model._gradient_actions, caches[1])
                self.assertEqual(before_config, OmegaConf.to_container(config, resolve=True))
        self.assertEqual(source_hashes, {str(p.relative_to(self.source)): sha256(p) for p in self.source.rglob("*") if p.is_file()})

    def test_reject_incomplete_mismatched_and_corrupt_sources(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            root = Path(temporary)
            for name, mutate in (
                ("incomplete", lambda r: r.update(status="RUNNING")),
                ("checkpoint_hash", lambda r: r["runs"][0]["checkpoints"][-1].update(sha256="f" * 64)),
                ("config_mismatch", lambda r: r["runs"][0]["config"]["jepa_model"]["optim"].update(grad_clip=17.)),
                ("dataset_mismatch", lambda r: r["dataset_identity"].update(dataset_id="other")),
            ):
                destination = root / name
                shutil.copytree(self.source, destination)
                report = json.loads((destination / "report.json").read_text())
                mutate(report)
                (destination / "report.json").write_text(json.dumps(report))
                with self.subTest(case=name), self.assertRaises((ValueError, FileNotFoundError)):
                    runner.load_source(destination, [1], horizons=(5, 15), policy_cases=1, policy_steps=2)
            destination = root / "corrupt_bank"
            shutil.copytree(self.source, destination)
            bank = torch.load(destination / "branches.pt", weights_only=False)
            bank["splits"]["validation"][0]["states"][0, 0, 0] += 1
            torch.save(bank, destination / "branches.pt")
            with self.assertRaises(ValueError):
                runner.load_source(destination, [1], horizons=(5, 15), policy_cases=1, policy_steps=2)
            # A self-consistently rehashed source must still obey the policy reset
            # assumptions; integrity checks alone cannot establish compatibility.
            destination = root / "nonzero_prefix"
            shutil.copytree(self.source, destination)
            bank = torch.load(destination / "branches.pt", weights_only=False)
            case = bank["splits"]["validation"][0]
            case["past_action"][0, 0] = .5
            case["sha256"] = _digest({k: v for k, v in case.items() if k != "sha256"})
            bank["metadata"]["split_hashes"]["validation"] = _digest(bank["splits"]["validation"])
            bank["sha256"] = _digest(bank["metadata"])
            report = json.loads((destination / "report.json").read_text())
            report["branches"].update(metadata=bank["metadata"], sha256=bank["sha256"])
            (destination / "report.json").write_text(json.dumps(report))
            torch.save(bank, destination / "branches.pt")
            with self.assertRaisesRegex(ValueError, "zero-action"):
                runner.load_source(destination, [1], horizons=(5, 15), policy_cases=1, policy_steps=2)
        for seeds, horizons in (([1, 1], (5, 15)), ([2], (5, 15)), ([1], (5, 16))):
            with self.subTest(seeds=seeds, horizons=horizons), self.assertRaises(ValueError):
                runner.load_source(self.source, seeds, horizons=horizons, policy_cases=1, policy_steps=2)

    def test_main_dry_run_and_completed_report_use_validation_without_fitting(self):
        runner = self.runner()
        source_hashes = {str(p.relative_to(self.source)): sha256(p) for p in self.source.rglob("*") if p.is_file()}
        for dry in (True, False):
            output = self.root / ("dry" if dry else "complete")
            with no_fitting(), redirect_stdout(io.StringIO()):
                code = runner.main([*self.argv(output), *(["--dry-run"] if dry else [])])
            self.assertEqual(code, 0)
            if dry:
                self.assertFalse(output.exists())
                continue
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["training_updates"], 0)
            self.assertEqual(report["online_updates"], 0)
            self.assertFalse(any(output.rglob("*.pt")))
            self.assertEqual(report["status"], "COMPLETE")
            self.assertEqual(len(report["runs"]), 6)
            self.assertEqual(report["protocol"]["split"], "validation")
            self.assertFalse(report["protocol"]["test_evaluated"])
            self.assertEqual(report["source"]["report_sha256"], sha256(self.source / "report.json"))
            expected_ids = [c["id"] for c in policy_cases(self.bank["splits"]["validation"], 1)]
            self.assertEqual(report["protocol"]["case_ids"], expected_ids)
            for row in report["runs"]:
                self.assertEqual(row["status"], "COMPLETE")
                evaluation = row["evaluation"]["result"]
                policy = evaluation["policy"]
                self.assertEqual(policy["case_ids"], expected_ids)
                self.assertEqual(evaluation["initial_state_sha256"], evaluation["final_state_sha256"])
                self.assertEqual(evaluation["initial_state_sha256"], row["source_checkpoint"]["model_state_sha256"])
                self.assertEqual(policy["maximum_return"], 4)
                self.assertEqual(policy["return_mean"], sum(policy["returns"]) / len(policy["returns"]))
                folder = output / f"seed_{row['seed']}" / row["condition"]["name"]
                traces = [json.loads(line) for line in (folder / "validation_policy.jsonl").read_text().splitlines()]
                self.assertEqual(len(traces), 2)
                self.assertEqual(sum(t["rewards"][0] for t in traces), policy["returns"][0])
            sentinel = (output / "report.json").read_bytes()
            with no_fitting(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                try:
                    refused = runner.main(self.argv(output))
                except SystemExit as error:
                    refused = error.code
                except ValueError:
                    refused = 1
            self.assertNotEqual(refused, 0)
            self.assertEqual((output / "report.json").read_bytes(), sentinel)
        self.assertEqual(source_hashes, {str(p.relative_to(self.source)): sha256(p) for p in self.source.rglob("*") if p.is_file()})


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
