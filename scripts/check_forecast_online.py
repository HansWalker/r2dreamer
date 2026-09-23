"""Real CPU/simulator checks for the two-model online forecast experiment."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

import tools
from dmc_expert.storage import dataset_identity
from scripts.check_goal_maintenance import source_fixture
from scripts.check_paper_faithful_duration import write_fixture
from scripts.check_state_normalization import tiny_config
from scripts.evaluate_goal_maintenance import load_source
from scripts.forecast_online_support import forecast_errors, probe_cases
from scripts.paper_faithful_duration_support import TaskBranchReplay
from scripts.paper_faithful_followup_eval import _equal
from scripts.train_forecast_online import arguments, configure, load_training, main, snapshot
from scripts.train_paper_faithful_duration import file_hash


def online_fixture(family, task):
    config = tiny_config(family, task)
    config.env.env_num = 2
    config.replay.batch_size = config.training.expert.batch_size = 4
    config.replay.episodes_per_batch = 2
    config.replay.max_size = 256
    config.training.online.steps = 96
    config.training.online.updates = 12
    config.training.online.warmup_transitions = 8
    # Real native updates/readout updates; no unavailable expert HDF5 in this fixture.
    config.state_head.online.expert_fraction = 0.
    return config


class ForecastOnlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.source = cls.root / "source"
        cls.source.mkdir()
        with patch("scripts.check_goal_maintenance.TASKS", ("cartpole_balance_sparse",)), \
                patch("scripts.check_goal_maintenance.tiny_config", side_effect=online_fixture):
            cls.source_report = source_fixture(cls.source)
        # Exercise the production 50% detached-readout retention path with a real
        # tiny HDF5 dataset. Save a stale source path to test --dataset-root too.
        cls.dataset = cls.root / "relocated_expert"
        identity = None
        for row in cls.source_report["runs"]:
            config = OmegaConf.create(row["config"])
            config.expert_data.train_episodes = 2
            config.expert_data.heldout_episodes = 1
            config.expert_data.policy_mode = "mpc"
            config.state_head.online.expert_fraction = .5
            config.training.expert.data_path = str(cls.dataset / str(config.scenario.dataset))
            if identity is None:
                write_fixture(config)
                metadata = json.loads((Path(config.training.expert.data_path) / "metadata.json").read_text())
                identity = dataset_identity(metadata)
            config.env.dataset_root = "/unavailable/source/expert"
            config.training.expert.data_path = "/unavailable/source/expert/cartpole_balance_sparse"
            row["config"] = OmegaConf.to_container(config, resolve=True)
            path = cls.source / row["key"] / "latest.pt"
            saved = torch.load(path, weights_only=False)
            saved.update(training_config=row["config"], row=copy.deepcopy(row), dataset_identity=identity)
            torch.save(saved, path)
            cls.source_report["datasets"][row["task"]]["identity"] = identity
        (cls.source / "report.json").write_text(json.dumps(cls.source_report))
        cls.hashes = {str(p.relative_to(cls.source)): file_hash(p) for p in cls.source.rglob("*") if p.is_file()}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def command(self, name, *extra):
        return ["--source-run", str(self.source), "--output", str(self.root / name), "--device", "cpu",
                "--dataset-root", str(self.dataset),
                "--online-steps", "48", "--save-every", "16", "--lookahead-seconds", ".1",
                "--hold-seconds", ".06", "--policy-cases", "2", "--policy-steps", "3", *extra]

    def prepared(self, name="direct"):
        args = arguments(self.command(name))
        _, banks, records = load_source(args)
        return args, banks, records

    def test_full_two_model_online_run_updates_weights_and_preserves_fixed_cases(self):
        output = self.root / "complete"
        log = io.StringIO()
        with redirect_stdout(log):
            status = main(self.command("complete"))
        self.assertEqual(status, 0, log.getvalue())
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(len(report["runs"]), 2)
        for row in report["runs"]:
            self.assertEqual(row["status"], "COMPLETE")
            self.assertEqual((row["raw_steps"], row["updates"]), (48, 4))
            self.assertEqual(row["budget"]["schedule_updates"], 12)
            self.assertEqual(row["budget"]["planner"]["horizon"], 5)
            self.assertNotEqual(row["budget"]["planner"]["objective"], "tail")
            snapshots = row["snapshots"]
            self.assertEqual([s["updates"] for s in snapshots], [0, 1, 4])
            self.assertNotEqual(snapshots[0]["model_state_sha256"], snapshots[-1]["model_state_sha256"])
            ids, returns, fixed = None, None, None
            self.assertEqual(len({s["evaluation"]["result"]["fixed_initial_plans"]["data_sha256"] for s in snapshots}), 1)
            folder = output / row["task"] / row["model"]
            for saved in snapshots:
                result = saved["evaluation"]["result"]
                self.assertTrue(result["training_state_preserved"])
                p = result["policy"]
                self.assertEqual(p["case_ids"], report["control"]["case_ids"])
                current = result["fixed_initial_plans"]["branches"]["cases"]
                now_ids = [case["id"] for case in current]
                now_returns = [case["metrics"]["5"]["returns"] for case in current]
                if ids is None:
                    ids, returns = now_ids, now_returns
                self.assertEqual(now_ids, ids)
                self.assertEqual(now_returns, returns)
                for item in result["forecast_errors"]["cases"]:
                    self.assertAlmostEqual(item["recursive_mse_by_step"][0], item["observed_history_mse_by_step"][0], places=5)
                probes = torch.load(folder / p["plans_file"], weights_only=False)
                self.assertEqual(len(result["current_plan_diagnostics"]["forecast_errors"]["cases"]), len(probes))
                if fixed is None:
                    fixed = probe_cases(probes)
                traces = [json.loads(line) for line in (folder / p["traces_file"]).read_text().splitlines()]
                for probe in probes:
                    index = p["case_ids"].index(probe["case_id"])
                    torch.testing.assert_close(probe["actions"][0, 0], torch.tensor(traces[probe["step"]]["actions"][index]), rtol=0, atol=0)
            latest = torch.load(folder / "latest.pt", weights_only=False)
            self.assertFalse(latest["resume_supported"])
            self.assertEqual(latest["updates"], 4)
            self.assertTrue(latest["optimizer_state_dict"])
            metrics = [json.loads(line) for line in (folder / "online_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(metrics), 12)
            self.assertTrue(all(m["metrics"].get("native/expert_sequences", 0) == 0 for m in metrics))
            self.assertTrue(all(m["metrics"]["state/expert_examples"] == 2 for m in metrics if m["metrics"]))
            self.assertEqual(row["online"]["head_updates_after"] - row["online"]["head_updates_before"], 4)
        self.assertIn("Run | complete | status=COMPLETE", log.getvalue())
        self.assertEqual(self.hashes, {str(p.relative_to(self.source)): file_hash(p) for p in self.source.rglob("*") if p.is_file()})

    def test_source_load_retains_optimizers_and_schedule_is_a_prefix(self):
        args, banks, records = self.prepared()
        for record in records:
            config, _, budget = configure(record, banks[record["task"]], args)
            _, model = load_training(record, config)
            saved = torch.load(record["path"], weights_only=False)
            for name, optimizer in model.optimizers.items():
                self.assertTrue(_equal(optimizer.state_dict(), saved["optimizer_state_dict"][name]))
            self.assertTrue(_equal(model.state_head.optimizer.state_dict(), saved["optimizer_state_dict"]["state_head"]))
            self.assertEqual((budget["midpoint_updates"], budget["updates"], budget["schedule_updates"]), (1, 4, 12))
            if hasattr(model, "configure_online"):
                model.configure_online(config.training.online.updates, resumed=False)
                self.assertEqual(model.scheduler.last_epoch, 0)
            override = copy.copy(args)
            override.dataset_root = self.root / "relocated"
            changed, _, _ = configure(record, banks[record["task"]], override)
            self.assertEqual(changed.training.expert.data_path, str(override.dataset_root / str(config.scenario.dataset)))

    def test_full_snapshot_leaves_the_next_training_update_unchanged(self):
        args, banks, records = self.prepared("isolation")
        for record in records:
            bank = banks[record["task"]]
            config, condition, _ = configure(record, bank, args)
            _, model = load_training(record, config)
            if hasattr(model, "configure_online"):
                model.configure_online(config.training.online.updates, resumed=False)
            model.train()
            replay = TaskBranchReplay(bank, batch_size=4, episodes_per_batch=2, sequence_length=4, seed=19)
            batch = replay.sample_training_batch()
            model.update(batch)
            baseline = copy.deepcopy(model)
            rng = tools.get_rng_state()
            folder = self.root / "isolation" / record["model"]
            folder.mkdir(parents=True)
            row = {"model": record["model"], "raw_steps": 24, "updates": 1, "snapshots": []}
            with redirect_stdout(io.StringIO()):
                snapshot(config, model, bank, condition, args, folder, row, "before", None)
            model.update(batch)
            tools.set_rng_state(rng)
            baseline.update(batch)
            self.assertTrue(_equal(model.state_dict(), baseline.state_dict()))
            self.assertTrue(_equal(model.optimizer_state_dict(), baseline.optimizer_state_dict()))

    def test_real_history_alignment_with_known_dynamics(self):
        class Exact(torch.nn.Module):
            history_size, device = 3, torch.device("cpu")
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
            def encode(self, obs):
                return obs["image"].float()
            def predict(self, state, action):
                return state + action
            def rollout(self, history, past, actions):
                return history[:, None, -1:] + actions.cumsum(2)
        model = Exact()
        actions = torch.tensor([[[2.], [-1.], [3.]], [[-2.], [4.], [1.]]])
        case = {"id": "known", "prefix": torch.tensor([[1.], [3.], [6.]]),
                "past_action": torch.tensor([[2.], [3.]]), "action": actions, "image": 6 + actions.cumsum(1)}
        result = forecast_errors(model, [case])["cases"][0]
        self.assertEqual(result["recursive_mse_by_step"], [0., 0., 0.])
        self.assertEqual(result["observed_history_mse_by_step"], [0., 0., 0.])
        self.assertEqual(result["persistence_mse_by_step"], [4., 2.5, 12.5])
        self.assertEqual(result["observed_history_persistence_mse_by_step"], [4., 8.5, 5.])

    def test_dry_run_and_invalid_budgets_write_no_outputs(self):
        for name, extra, expected in (("dry", ["--dry-run"], 0),
                                      ("fractional", ["--online-steps", "47"], 1),
                                      ("overlap", ["--online-seed", "71000000"], 1),
                                      ("warmup", ["--online-steps", "16"], 1),
                                      ("missing_expert", ["--dataset-root", str(self.root / "missing")], 1)):
            with redirect_stdout(io.StringIO()):
                code = main(self.command(name, *extra))
            self.assertEqual(code, expected)
            self.assertFalse((self.root / name).exists())

    def test_failure_and_interrupt_keep_partial_results_and_print_run_name(self):
        for name, error, code, status in (("failed", RuntimeError("injected collection error"), 1, "FAIL"),
                                          ("interrupted", KeyboardInterrupt(), 130, "INTERRUPTED")):
            log = io.StringIO()
            with patch("training.planning.OnlineSession.collect", side_effect=error), redirect_stdout(log):
                result = main(self.command(name))
            self.assertEqual(result, code)
            report = json.loads((self.root / name / "report.json").read_text())
            self.assertEqual(report["status"], status)
            self.assertEqual(len(report["runs"][0]["snapshots"]), 1)
            self.assertTrue((self.root / name / "cartpole_balance_sparse/temporal_straightening/latest.pt").is_file())
            self.assertIn(f"Run | {name} | status={status}", log.getvalue())


if __name__ == "__main__":
    unittest.main()
