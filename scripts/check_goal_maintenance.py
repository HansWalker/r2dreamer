"""CPU/simulator checks for frozen goal-maintenance planning on all three tasks."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from models.shared.latent_goal import latent_goal_cost
from scripts.check_state_normalization import tiny_config
from scripts.diagnose_fresh_readout import tensor_digest
from scripts.evaluate_goal_maintenance import (
    arguments, branch_maintenance, conditions, evaluate_condition, load_frozen, load_source, main,
)
from scripts.paper_faithful_duration_support import TaskBranchReplay, collect_bank
from scripts.train_paper_faithful_duration import (
    FORMAT, MODELS, TASKS, file_hash, runtime_versions, save_checkpoint, source_hashes,
)
from training import load_model_family
from training.protocol import implementation_sha256


def source_fixture(root):
    """Small real simulator banks and actually updated checkpoints; no expert files."""
    report = {"format": FORMAT, "status": "COMPLETE", "implementation_sha256": implementation_sha256(),
              "source_hashes": source_hashes(), "versions": runtime_versions("cpu"),
              "banks": {}, "datasets": {}, "runs": []}
    for task in TASKS:
        bank = None
        for family in MODELS:
            config = tiny_config(family, task)
            config.seed = 1
            config.env.time_limit = 1000
            config.training.expert.updates = 1
            config.jepa_model.planner.horizon = 3
            config.jepa_model.planner.samples = 4
            config.jepa_model.planner.iterations = 1
            config.jepa_model.planner.elites = 2
            config.state_head.samples_per_update = 4
            OmegaConf.resolve(config)
            if bank is None:
                bank = collect_bank(config, counts={"train": 2, "validation": 2, "test": 1},
                                    candidates=6, horizon=5, seed=71_000_000 + TASKS.index(task) * 100)
                bank_path = root / f"{task}_bank.pt"
                torch.save(bank, bank_path)
                report["banks"][task] = {"file": bank_path.name, "file_sha256": file_hash(bank_path),
                                         "sha256": bank["sha256"], "metadata": bank["metadata"]}
            identity = {"task": str(config.scenario.collection_task), "dataset_id": "simulator-test-fixture"}
            report["datasets"][task] = {"identity": identity}
            torch.manual_seed(1)
            model = load_model_family(family).build_model(config)
            if hasattr(model, "configure_pretraining"):
                model.configure_pretraining(2)
            replay = TaskBranchReplay(bank, batch_size=4, episodes_per_batch=2, sequence_length=4, seed=5)
            model.train()
            model.update(replay.sample_training_batch())
            row = {"key": f"{task}/{family}", "task": task, "model": family, "seed": 1,
                   "status": "COMPLETE", "updates": 1, "config": OmegaConf.to_container(config, resolve=True)}
            path = root / row["key"] / "latest.pt"
            path.parent.mkdir(parents=True)
            save_checkpoint(path, model, replay, replay, row, 1, identity, bank["sha256"])
            report["runs"].append(row)
    (root / "report.json").write_text(json.dumps(report, indent=2))
    return report


def no_training():
    stack = ExitStack()
    for target in ("models.planning.LatentPlanner.update", "models.leworldmodel.model.LeWorldModel.update",
                   "training.planning.build_replay", "training.planning.expert_update", "training.planning.OnlineSession"):
        stack.enter_context(patch(target, side_effect=AssertionError(f"Frozen evaluator called {target}")))
    return stack


class GoalPreferenceTests(unittest.TestCase):
    def test_staying_near_goal_beats_passing_through_and_allows_approach(self):
        # Both end exactly at the goal. Only the first stays there over the tail.
        for shape, reduction in (((4,), "sum"), ((3, 4), "mean")):
            values = torch.tensor([[9., 7., 0., 0., 0.], [0., 0., 3., 2., 0.]])
            path = values.reshape(1, 2, 5, *([1] * len(shape))).expand(1, 2, 5, *shape).clone().requires_grad_()
            goal = torch.zeros(1, *shape, requires_grad=True)
            last = latent_goal_cost(path, goal, reduction=reduction, mode="last")
            self.assertEqual(last[0, 0], last[0, 1])
            cost = latent_goal_cost(path, goal, reduction=reduction, mode="tail", tail_steps=3)
            self.assertLess(cost[0, 0], cost[0, 1])
            cost.sum().backward()
            self.assertEqual(path.grad[:, :, :2].abs().sum(), 0)
            self.assertGreater(path.grad[:, :, 2:4].abs().sum(), 0)
            self.assertIsNone(goal.grad)

    def test_seconds_resolve_without_shortening_native_horizon(self):
        planner = OmegaConf.create({"horizon": 5, "objective": "last", "tail_steps": 3})
        rows = conditions(planner, .02, .5, .3)
        self.assertEqual([(row["horizon"], row["tail_steps"]) for row in rows], [(5, 3), (25, 3), (25, 15)])
        self.assertEqual(planner.horizon, 5)
        planner.horizon = 25
        rows = conditions(planner, .04, .5, .3)
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[-1]["horizon"], rows[-1]["tail_steps"]), (25, 8))
        self.assertAlmostEqual(rows[-1]["hold_seconds"], .32)
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                conditions(planner, .02, value, .3)
        with self.assertRaises(ValueError):
            conditions(planner, .02, .1, .3)

    def test_branch_maintenance_handles_cost_ties_without_optimistic_selection(self):
        cases = [{"id": "a", "success": torch.tensor([[False, True, True], [True, False, False]])}]
        branches = {"cases": [{"id": "a", "metrics": {"3": {"actual_cost": [0., 0.], "predicted_cost": [1., 0.]}}}]}
        result = branch_maintenance(branches, cases, 3, 2)
        self.assertEqual(result["all"]["actual_selected_occupancy"], .5)
        self.assertEqual(result["all"]["actual_selected_maintenance"], .5)
        self.assertEqual(result["all"]["predicted_selected_maintenance"], 0.)
        self.assertEqual(result["informative"]["cases"], 1)


class FrozenComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.source = cls.root / "source"
        cls.source.mkdir()
        cls.report = source_fixture(cls.source)
        cls.original_hashes = {str(path.relative_to(cls.source)): file_hash(path) for path in cls.source.rglob("*") if path.is_file()}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def command(self, output, *extra):
        return ["--source-run", str(self.source), "--output", str(self.root / output), "--device", "cpu",
                "--lookahead-seconds", ".1", "--hold-seconds", ".06", "--policy-cases", "2", "--policy-steps", "3", *extra]

    def test_all_three_tasks_both_models_full_execution_preserves_source(self):
        log = io.StringIO()
        with no_training(), redirect_stdout(log):
            status = main(self.command("complete", "--tasks", *TASKS))
        self.assertEqual(status, 0, log.getvalue())
        report = json.loads((self.root / "complete/report.json").read_text())
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["training_updates"], 0)
        self.assertEqual(report["online_updates"], 0)
        self.assertEqual(set(report["controls"]), set(TASKS))
        seen = set()
        for row in report["runs"]:
            seen.add((row["task"], row["model"]))
            result = row["evaluation"]["result"]
            self.assertEqual(result["initial_state_sha256"], result["final_state_sha256"])
            p = result["policy"]
            self.assertEqual(p["case_ids"], report["controls"][row["task"]]["case_ids"])
            self.assertTrue(all("/validation/" in case for case in p["case_ids"]))
            folder = self.root / "complete" / row["task"] / row["model"] / row["condition"]["name"]
            trace = [json.loads(line) for line in (folder / p["traces_file"]).read_text().splitlines()]
            self.assertEqual([sum(x["rewards"][i] for x in trace) for i in range(2)], p["returns"])
            self.assertTrue((folder / p["plans_file"]).is_file())
            source = next(x for x in self.report["runs"] if x["task"] == row["task"] and x["model"] == row["model"])
            for key, value in source["config"]["jepa_model"]["planner"].items():
                if key not in ("horizon", "objective", "tail_steps"):
                    self.assertEqual(result["planner"][key], value)
        self.assertEqual(seen, {(task, model) for task in TASKS for model in MODELS})
        self.assertIn("Run | complete | status=COMPLETE", log.getvalue())
        self.assertEqual(self.original_hashes, {str(path.relative_to(self.source)): file_hash(path)
                                              for path in self.source.rglob("*") if path.is_file()})

    def test_dry_run_and_score_only_do_not_act(self):
        with no_training(), patch("models.planning.LatentPlanner.act", side_effect=AssertionError("unexpected action")), redirect_stdout(io.StringIO()):
            self.assertEqual(main(self.command("dry", "--dry-run")), 0)
            self.assertFalse((self.root / "dry").exists())
            self.assertEqual(main(self.command("scores", "--score-only")), 0)
        report = json.loads((self.root / "scores/report.json").read_text())
        self.assertEqual(report["controls"], {})
        self.assertTrue(all(row["evaluation"]["result"]["policy"] is None for row in report["runs"]))
        self.assertFalse(list((self.root / "scores").rglob("*.pt")))

    def test_reject_unavailable_horizon_and_modified_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "exceeds saved"):
            load_source(arguments(self.command("unused", "--lookahead-seconds", ".5")))
        original = torch.load
        with patch("scripts.evaluate_goal_maintenance.torch.load", wraps=original) as loader:
            def alter(path, **kwargs):
                payload = original(path, **kwargs)
                if path.name == "latest.pt":
                    payload["updates"] += 1
                return payload
            loader.side_effect = alter
            with self.assertRaisesRegex(ValueError, "Final checkpoint"):
                load_source(arguments(self.command("unused")))

    def test_exception_restores_planner_and_prints_run_name(self):
        args = arguments(self.command("unused"))
        _, banks, records = load_source(args)
        config, model = load_frozen(records[0], "cpu")
        before = tensor_digest(model.state_dict())
        planner, cache = model.planner, torch.zeros(1)
        model._cem_mean = cache
        with patch("scripts.evaluate_goal_maintenance.evaluate", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                evaluate_condition(config, model, banks[records[0]["task"]]["splits"]["validation"],
                                   records[0]["conditions"][-1], args)
        self.assertIs(model.planner, planner)
        self.assertIs(model._cem_mean, cache)
        self.assertEqual(tensor_digest(model.state_dict()), before)
        for error, expected, code in ((RuntimeError("injected"), "FAIL", 1), (KeyboardInterrupt(), "INTERRUPTED", 130)):
            log = io.StringIO()
            with patch("scripts.evaluate_goal_maintenance.load_source", side_effect=error), redirect_stdout(log), redirect_stderr(io.StringIO()):
                self.assertEqual(main(self.command("failed")), code)
            self.assertIn(f"Run | failed | status={expected}", log.getvalue())


if __name__ == "__main__":
    unittest.main()
