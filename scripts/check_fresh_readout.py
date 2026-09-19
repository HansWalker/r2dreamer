"""CPU checks for fresh-head fitting, frozen features, data isolation, and time alignment."""

import copy
import io
import itertools
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from dmc_expert.storage import dataset_identity
from models.shared.physical_state import PhysicalStateHead
from scripts.check_online_checkpoint_smoke import fixture
from scripts.check_state_normalization import settings, tiny_config
from scripts.diagnose_fresh_readout import (
    FeatureBank,
    cache_forecasts,
    check_partition,
    expert_partition,
    fit_diagnostic,
    fit_head,
    fresh_head,
    load_checkpoint,
    main,
    physical_metrics,
    tensor_digest,
    write_reports,
)
from training import load_model_family
from training.protocol import checkpoint_compatibility


def options(**overrides):
    return SimpleNamespace(**{
        "context_length": 3, "horizons": [1, 5, 10, 100], "windows_per_episode": 1,
        "eval_batch_size": 2, "encode_batch_size": 32, "data_seed": 8_000_000, "fit_seed": 9_000_000,
        "updates": 2, "fit_updates": 2, "fit_batch_size": 4, "fit_tolerance": .05,
        "expert_train": 1, "expert_validation": 1, "device": "cpu", "seed": 0, **overrides,
    })


def episodes(targets=5, length=104, action_dim=1):
    result = []
    for index, source in enumerate(("expert", "zero", "random", "expert", "zero", "random")):
        item = {"id": f"test:{index}", "policy": source,
                "image": torch.full((length, 64, 64, 3), index * 20, dtype=torch.uint8),
                "state": torch.randn(length, targets) * .1,
                "action": torch.rand(length - 1, action_dim) * 2 - 1}
        item["sha256"] = tensor_digest({key: item[key] for key in ("image", "state", "action")})
        result.append(item)
    return result


class FreshReadoutTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_sampling_is_source_balanced_causal_and_episode_uniform(self):
        data = episodes(length=12)
        feature = [torch.stack((torch.full((12,), i), torch.arange(12)), -1).float() for i in range(6)]
        bank = FeatureBank(data, feature, 3)
        x, y, sampled = bank.sample(256, torch.Generator().manual_seed(14))
        ids, starts = sampled["episodes"], sampled["starts"]
        self.assertEqual(sum(data[i]["policy"] == "expert" for i in ids), 128)
        self.assertEqual(sum(data[i]["policy"] == "zero" for i in ids), 64)
        self.assertEqual(sum(data[i]["policy"] == "random" for i in ids), 64)
        for row, (episode, start) in enumerate(zip(ids, starts)):
            torch.testing.assert_close(x[row, :, 0], torch.full((3,), float(episode)))
            torch.testing.assert_close(x[row, :, 1], torch.arange(start, start + 3).float())
            torch.testing.assert_close(y[row], data[episode]["state"][start:start + 3])
        for count in (0, 2, 5):
            with self.assertRaisesRegex(ValueError, "multiples of four"):
                bank.sample(count, torch.Generator())

    def test_partition_checks_identity_and_exact_duplicates_but_not_shared_states(self):
        data = episodes(length=4)
        check_partition(data[:3], data[3:])
        for key in ("id", "sha256"):
            copied = copy.deepcopy(data)
            copied[3][key] = copied[0][key]
            with self.assertRaisesRegex(ValueError, "disjoint"):
                check_partition(copied[:3], copied[3:])
        with self.assertRaises(ValueError):
            check_partition([data[0], data[0]], data[3:])

    def test_fresh_head_keeps_architecture_expert_scales_and_rng_not_old_weights(self):
        for tokens in (1, 16):
            config = settings()
            old = PhysicalStateHead(8, config, history=3, tokens=tokens)
            old.set_stats([0, 1, 0, 0, 0], [.03, .001, .01, .15, .2])
            state = copy.deepcopy(old.state_dict())
            rng = torch.get_rng_state().clone()
            fresh = fresh_head(old, config, torch.ones(30, 5), 900)
            torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
            self.assertEqual(sum(p.numel() for p in fresh.parameters()), sum(p.numel() for p in old.parameters()))
            torch.testing.assert_close(fresh.std, old.std, atol=0, rtol=0)
            torch.testing.assert_close(fresh.output_scale, torch.tensor([.1, 1, 1, 1, 1]))
            self.assertFalse(torch.equal(fresh.project[0].weight, old.project[0].weight))
            self.assertFalse(fresh._online)
            self.assertEqual(fresh.optimizer.param_groups[0]["lr"], config.lr)
            for key, value in state.items():
                torch.testing.assert_close(old.state_dict()[key], value, atol=0, rtol=0)

    def test_goal_metrics_report_both_errors_and_missing_coverage(self):
        head = PhysicalStateHead(8, settings())
        tolerance = torch.tensor([.25, .10004])
        truth = torch.tensor([[0., 1, 0, 0, 0], [0, -1, 0, 0, 0]])
        swapped = truth.flip(0)
        values = physical_metrics(swapped, truth, head, tolerance, "box")
        self.assertEqual(values["false_success_rate"], 1)
        self.assertEqual(values["missed_success_rate"], 1)
        no_failures = physical_metrics(truth[:1], truth[:1], head, tolerance, "box")
        self.assertIsNone(no_failures["false_success_rate"])
        self.assertEqual(no_failures["missed_success_rate"], 0)
        # A crossing of +/-pi is a small angular error, not a full revolution.
        angles = torch.tensor([3.13, -3.13])
        states = torch.stack((torch.zeros(2), angles.cos(), angles.sin(), torch.zeros(2), torch.zeros(2)), -1)
        wrapped = physical_metrics(states[:1], states[1:], head, tolerance, "box")
        self.assertLess(wrapped["relation_rmse"][1], .03)

    def test_cached_forecast_uses_only_prefix_and_correct_action_target_indices(self):
        class AdditiveModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.device = torch.device("cpu")
                self.history_size = 3
                self.state_head = SimpleNamespace(history=3)
                self.calls = 0

            def rollout(self, history, past, actions):
                self.calls += 1
                return history[:, None, -1:] + actions.cumsum(2)

        model = AdditiveModel()
        actions = torch.arange(1, 110).float()[:, None]
        feature = torch.cat((torch.zeros(1, 1), actions.cumsum(0)))
        data = [{"image": torch.zeros(110, 1), "state": feature.clone(), "action": actions, "policy": "random"}]
        args = options(context_length=6, windows_per_episode=3)
        windows, cached = cache_forecasts(model, data, [feature], args)
        torch.testing.assert_close(cached["observed"], cached["forecast"], atol=0, rtol=0)
        for row, window in enumerate(windows):
            expected = feature[[window.start + 6 + h - 1 for h in args.horizons], 0]
            torch.testing.assert_close(cached["truth"][row, :, 0], expected)
            torch.testing.assert_close(cached["observed"][row, :, -1, 0], expected)
        # Corrupting real future features changes observed decoding, but never the forecast.
        args.windows_per_episode = 1
        windows, cached = cache_forecasts(model, data, [feature], args)
        changed = feature.clone()
        changed[windows[0].start + args.context_length:] += 1000
        _, other = cache_forecasts(model, data, [changed], args)
        torch.testing.assert_close(cached["forecast"], other["forecast"], atol=0, rtol=0)
        self.assertFalse(torch.equal(cached["observed"], other["observed"]))

    def test_small_batch_fitting_learns_without_encoder_gradients(self):
        config = settings()
        config.projection_dim, config.hidden_dim, config.lr = 8, 32, .003
        head = PhysicalStateHead(8, config, history=3)
        head.samples_per_update = 4
        x = torch.randn(4, 3, 8, requires_grad=True)
        y = torch.randn(4, 3, 5) * .1
        initial = (head(x)[:, 0] - y[:, -1]).square().mean().item()
        fit_head(head, None, 200, 7, io.StringIO(), fixed=(x, y, {}))
        final = (head(x)[:, 0] - y[:, -1]).square().mean().item()
        self.assertLess(final, initial * .01)
        self.assertIsNone(x.grad)
        self.assertEqual(int(head.examples), 800)

    def test_expert_partition_reads_train_only_and_reports_real_episode_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = fixture(root, "leworldmodel")
            path = root / str(config.scenario.dataset)
            metadata = json.loads((path / "metadata.json").read_text())
            metadata["episode_splits"] = {"train": [0, 2], "heldout": [2, 3]}
            train, validation = expert_partition(path, metadata, config, PhysicalStateHead(8, config.state_head).targets, options())
            self.assertEqual({e["episode_index"] for e in train + validation}, {0, 1})
            check_partition(train, validation)
            self.assertEqual(len(train[0]["image"]), len(train[0]["action"]) + 1)
            with self.assertRaisesRegex(ValueError, "TRAIN episodes"):
                expert_partition(path, metadata, config, PhysicalStateHead(8, config.state_head).targets,
                                 options(expert_train=2))

    def test_real_checkpoint_loading_is_read_only_and_rejects_online(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, job = fixture(root, "leworldmodel")
            path = Path(job["checkpoint"])
            saved = torch.load(path, weights_only=False)
            saved["training_config"]["state_head"]["samples_per_update"] = 4
            saved["compatibility"] = checkpoint_compatibility(OmegaConf.create(saved["training_config"]))
            torch.save(saved, path)
            before = path.read_bytes()
            with patch("torch.save", side_effect=AssertionError("No checkpoint writes")):
                _, model, _ = load_checkpoint(path, "cartpole_balance_sparse", "leworldmodel", options())
            self.assertEqual(int(model.state_head.updates), 1)
            self.assertEqual(path.read_bytes(), before)
            saved["phase"] = "online"
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "completed expert"):
                load_checkpoint(path, "cartpole_balance_sparse", "leworldmodel", options())

    def test_full_diagnostic_freezes_both_native_models_and_keeps_reporting_honest(self):
        for name, scenario in itertools.product(("leworldmodel", "temporal_straightening"),
                                                ("cartpole_balance_sparse", "reacher", "ball_in_cup")):
            with self.subTest(model=name, scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                config = tiny_config(name, scenario)
                config.state_head.samples_per_update = 4
                model = load_model_family(name).build_model(config)
                dimensions = len(model.state_head.coordinates)
                model.state_head.set_stats(torch.zeros(dimensions), torch.full((dimensions,), .1))
                original = tensor_digest(model.state_dict())
                data = episodes(targets=dimensions, action_dim=model.action_dim)
                path = Path(temporary)
                with patch.object(model, "act", side_effect=AssertionError("No action optimization")), \
                     patch.object(model, "update", side_effect=AssertionError("No native updates")), \
                     patch("torch.save", side_effect=AssertionError("No checkpoint writes")):
                    result = fit_diagnostic(model, config, data[:3], data[3:], options(), path)
                self.assertEqual(tensor_digest(model.state_dict()), original)
                self.assertTrue(result["native_unchanged"])
                self.assertEqual(result["head_examples"], 8)
                self.assertEqual(result["small_batch"]["examples"], 8)
                self.assertEqual(result["head_lr"], float(config.state_head.lr))
                self.assertEqual(len((path / "metrics.jsonl").read_text().splitlines()), 4)
                self.assertIn("5", result["after"]["validation"]["random"]["forecast"])
                self.assertEqual(result["before"]["validation"]["random"]["true_persistence"],
                                 result["after"]["validation"]["random"]["true_persistence"])
                result.update(status="COMPLETE", scenario=scenario, model=name)
                summary = write_reports(path, [result], {}, options())
                report = json.loads((path / "report.json").read_text())
                self.assertFalse(report["checkpoint_writes"])
                self.assertFalse(report["benchmark_heldout_used"])
                self.assertFalse(report["validation_fitting"])
                self.assertIn("not that the model is repaired", summary)
                self.assertTrue((path / "physical_errors.csv").is_file())

    def test_launcher_loads_checkpoints_and_collects_one_common_pool_for_both_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ("leworldmodel", "temporal_straightening")
            sources = []
            for name in names:
                source = root / name
                source.mkdir()
                config, job = fixture(source, name)
                config.expert_data.train_episodes, config.expert_data.heldout_episodes = 2, 1
                config.state_head.samples_per_update = 4
                metadata_path = source / str(config.scenario.dataset) / "metadata.json"
                metadata = json.loads(metadata_path.read_text())
                metadata["episode_splits"] = {"train": [0, 2], "heldout": [2, 3]}
                metadata_path.write_text(json.dumps(metadata))
                checkpoint = torch.load(job["checkpoint"], weights_only=False)
                checkpoint["training_config"] = OmegaConf.to_container(config, resolve=True)
                checkpoint["compatibility"] = checkpoint_compatibility(config)
                checkpoint["dataset_identity"] = dataset_identity(metadata)
                target = root / "runs" / "cartpole_balance_sparse" / name / "default" / "seed_0" / "pretrained.pt"
                target.parent.mkdir(parents=True)
                torch.save(checkpoint, target)
                sources.append((target, target.read_bytes()))
            args = options(models=list(names), scenarios=["cartpole_balance_sparse"],
                           run_root=root / "runs", dataset_root=root / names[0], output=root / "reports",
                           sim_train=1, sim_validation=1)

            def simulator_episode(config, model, seed, mode):
                generator = torch.Generator().manual_seed(seed)
                return {"seed": seed, "policy": mode, "agent_steps": 128, "return": 10.,
                        "image": torch.randint(0, 256, (129, 64, 64, 3), dtype=torch.uint8, generator=generator),
                        "state": torch.randn(129, 5, generator=generator),
                        "action": torch.rand(128, 1, generator=generator) * 2 - 1}

            with patch("scripts.diagnose_fresh_readout.arguments", return_value=args), \
                 patch("scripts.diagnose_fresh_readout.collect_episode", side_effect=simulator_episode) as collector, \
                 patch("torch.save", side_effect=AssertionError("No checkpoint writes")), \
                 redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit_status:
                main()
            self.assertEqual(exit_status.exception.code, 0)
            self.assertEqual(collector.call_count, 4)
            report = json.loads((args.output / "report.json").read_text())
            self.assertEqual(len(report["results"]), 2)
            self.assertTrue(all(result["status"] == "COMPLETE" for result in report["results"]))
            self.assertEqual(report["results"][0]["windows"], report["results"][1]["windows"])
            for path, original in sources:
                self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
