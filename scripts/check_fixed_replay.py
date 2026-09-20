"""CPU regression checks for fixed-unit heads and controlled native adaptation."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from dmc_expert.storage import dataset_identity
from scripts.check_online_checkpoint_smoke import fixture
from scripts.check_state_normalization import tiny_config
from scripts.diagnose_fixed_replay import (
    MODES,
    FixedBatches,
    batchnorm_state,
    main,
    replay_partition,
    representation_state,
    run_ablation,
    tensor_digest,
    write_report,
)
from scripts.diagnose_fresh_readout import fresh_head
from scripts.smoke_models import synthetic_batch
from training import load_model_family
from training.protocol import checkpoint_compatibility


def options(**overrides):
    return SimpleNamespace(**{
        "context_length": 3, "horizons": [1, 5], "windows_per_episode": 1,
        "encode_batch_size": 8, "eval_batch_size": 4, "data_seed": 600, "fit_seed": 700,
        "updates": 2, "head_updates": 2, "eval_every": 1, "expert_train": 2,
        "expert_validation": 1, "replay_validation": 1, "seed": 0, "device": "cpu", **overrides,
    })


def pool(model, count=6, length=12):
    result = []
    for index in range(count):
        item = {
            "id": str(index), "policy": "expert" if index < count // 2 else "replay",
            "image": torch.randint(0, 256, (length, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(length, len(model.state_head.coordinates)),
            "action": torch.rand(length - 1, model.action_dim) * 2 - 1,
        }
        item["sha256"] = tensor_digest({key: item[key] for key in ("image", "state", "action")})
        result.append(item)
    return result


class FixedReplayTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)

    def test_freezes_survive_train_act_and_leave_predictor_trainable(self):
        for name in ("leworldmodel", "temporal_straightening"):
            config = tiny_config(name, "cartpole_balance_sparse")
            model = load_model_family(name).build_model(config)
            batch, _, _ = synthetic_batch(config, model, batch_size=4)
            for mode in MODES:
                with self.subTest(model=name, mode=mode):
                    model.set_adaptation_mode(mode)
                    model.eval().train()
                    bns = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
                    self.assertTrue(bns)
                    self.assertTrue(all(module.training == (mode == "native") for module in bns))
                    self.assertTrue(all(p.requires_grad for p in model.predictor.parameters()))
                    self.assertTrue(all(p.requires_grad for p in model.pred_projector.parameters()))
                    before_bn = tensor_digest(batchnorm_state(model))
                    before_encoder = tensor_digest(representation_state(model))
                    model.update(batch)
                    self.assertEqual(before_bn == tensor_digest(batchnorm_state(model)), mode != "native")
                    self.assertEqual(before_encoder == tensor_digest(representation_state(model)), mode == "frozen_encoder")
                    planner = "_gradient_plan" if str(model.planner.type) == "gradient" else "_cem"
                    with patch.object(model, planner, return_value=torch.zeros(4, model.action_dim)):
                        model.act({"image": batch[0]["image"][:, :3],
                                   "goal_image": batch[0]["image"][:, -1]}, batch[1][:, :2])
                    self.assertTrue(all(module.training == (mode == "native") for module in bns))
                    for module in (model.encoder, model.projector):
                        self.assertEqual(module.training, mode != "frozen_encoder")
                        self.assertTrue(all(p.requires_grad == (mode != "frozen_encoder") for p in module.parameters()))
            model.set_adaptation_mode("native")
            self.assertTrue(all(p.requires_grad for p in model.encoder.parameters()))
            with self.assertRaises(ValueError):
                model.set_adaptation_mode("typo")

    def test_clipping_reports_preclip_norm_and_frequency_without_changing_losses(self):
        config = tiny_config("temporal_straightening", "cartpole_balance_sparse")
        self.assertEqual(config.jepa_model.optim.grad_clip, 1.0)
        model = load_model_family("temporal_straightening").build_model(config)
        batch, _, _ = synthetic_batch(config, model, batch_size=4)
        model.grad_clip = 1e-5
        metrics = model.update(batch)
        self.assertGreater(metrics["grad_norm"], model.grad_clip)
        self.assertEqual(metrics["grad_clipped"], 1)
        self.assertEqual(metrics["grad_clip_fraction_since_load"], 1)
        grads = [p.grad for name, p in model.named_parameters() if not name.startswith("state_head.") and p.grad is not None]
        self.assertLessEqual(torch.stack([grad.norm() for grad in grads]).norm().item(), 1.01e-5)
        model.grad_clip = 1e10
        metrics = model.update(batch)
        self.assertEqual(metrics["grad_clipped"], 0)
        self.assertEqual(metrics["grad_clip_fraction_since_load"], .5)

    def test_fresh_default_uses_fixed_units_not_variance_or_old_conditioning(self):
        config = tiny_config("leworldmodel", "cartpole_balance_sparse")
        old = load_model_family("leworldmodel").build_model(config).state_head
        old.set_stats([0, 1, 0, 0, 0], [.03, .001, .01, .15, .2])
        old.output_scale.copy_(old.std)
        head = fresh_head(old, config.state_head, None, 123)
        torch.testing.assert_close(head.output_scale, torch.ones(5), rtol=0, atol=0)
        torch.testing.assert_close(head.loss_scale, torch.ones(5), rtol=0, atol=0)
        torch.testing.assert_close(head.std, old.std, rtol=0, atol=0)

    def test_fixed_sampling_alignment_repetition_and_episode_split(self):
        config = tiny_config("leworldmodel", "cartpole_balance_sparse")
        config.replay.batch_size, config.replay.episodes_per_batch = 4, 2
        model = load_model_family("leworldmodel").build_model(config)
        episodes = pool(model)
        left, right = [FixedBatches(episodes[:2], config, 12) for _ in range(2)]
        plan = left.draw()
        self.assertEqual(plan, right.draw())
        batch = left.batch(plan)
        encoded = [episode["state"] for episode in episodes[:2]]
        features, labels = left.features(plan, encoded)
        torch.testing.assert_close(features, labels)
        torch.testing.assert_close(batch[0]["physical_state"], labels)
        for row, (index, start) in enumerate((i, s) for i, starts in plan for s in starts):
            torch.testing.assert_close(batch[1][row], episodes[index]["action"][start:start + 3])
        rows = list(left.sampler.completed) + [TensorDict({"image": episodes[2]["image"],
            "physical_state": episodes[2]["state"], "action": torch.zeros(12, 1)}, batch_size=[12])]
        payload = {"replay_state": {"replay": {"completed": rows}}}
        train, valid = replay_partition(payload, 1, 3, 5, 123)
        self.assertFalse({e["id"] for e in train} & {e["id"] for e in valid})
        self.assertEqual(len(train), 2)
        self.assertEqual(len(train[0]["image"]), len(train[0]["action"]) + 1)
        payload["replay_state"]["replay"]["completed"] = [rows[0], rows[0], rows[1]]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            replay_partition(payload, 1, 3, 5, 123)

    def test_real_ablations_use_identical_batches_and_never_plan_collect_or_save(self):
        for name in ("leworldmodel", "temporal_straightening"):
            with self.subTest(model=name), tempfile.TemporaryDirectory() as temporary:
                config = tiny_config(name, "cartpole_balance_sparse")
                config.replay.batch_size, config.replay.episodes_per_batch = 4, 2
                config.state_head.samples_per_update = 4
                config.training.online.updates = 8
                family = load_model_family(name)
                model = family.build_model(config)
                data = pool(model)
                checkpoint = copy.deepcopy(family.checkpoint(model))
                hashes = []
                original_update = model.update
                def capture(batch, hashes=hashes, original_update=original_update):
                    hashes.append(tensor_digest({**batch[0], "action": batch[1]}))
                    return original_update(batch)
                with patch.object(model, "update", side_effect=capture), \
                     patch.object(model, "act", side_effect=AssertionError("No planner")), \
                     patch("torch.save", side_effect=AssertionError("No checkpoint writes")), redirect_stdout(io.StringIO()):
                    result = run_ablation(config, model, checkpoint, (data[:2], data[3:5]),
                                          [data[2], data[5]], options(), Path(temporary))
                self.assertEqual(hashes[:2], hashes[2:4])
                self.assertEqual(hashes[:2], hashes[4:])
                for trial in result["trials"]:
                    self.assertEqual(trial["status"], "COMPLETE", trial.get("error"))
                    self.assertEqual(trial["updates"], 2)
                    self.assertEqual(len(trial["snapshots"]), 2)
                    if trial["mode"] == "frozen_encoder":
                        self.assertTrue(trial["encoder_and_projector_unchanged"])
                        self.assertEqual(trial["snapshots"][-1]["latent_drift_rms"], 0)
                result.update(scenario="cartpole_balance_sparse", model=name)
                write_report(Path(temporary), [result], options())
                report = json.loads((Path(temporary) / "report.json").read_text())
                self.assertFalse(report["checkpoint_writes"])
                self.assertFalse(report["validation_fitting"])
                self.assertTrue((Path(temporary) / "physical_errors.csv").exists())

    def test_launcher_loads_saved_replay_and_preserves_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, job = fixture(root, "leworldmodel")
            config.expert_data.train_episodes, config.expert_data.heldout_episodes = 2, 1
            config.state_head.samples_per_update = 4
            config.replay.batch_size, config.replay.episodes_per_batch = 4, 1
            metadata_path = root / str(config.scenario.dataset) / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["episode_splits"] = {"train": [0, 2], "heldout": [2, 3]}
            metadata_path.write_text(json.dumps(metadata))
            checkpoint = torch.load(job["checkpoint"], weights_only=False)
            checkpoint["training_config"] = OmegaConf.to_container(config, resolve=True)
            checkpoint["compatibility"] = {**checkpoint_compatibility(config), "recipe_version": 6}
            checkpoint["dataset_identity"] = dataset_identity(metadata)
            run = root / "runs" / "cartpole_balance_sparse" / "leworldmodel" / "default" / "seed_0"
            run.mkdir(parents=True)
            torch.save(checkpoint, run / "pretrained.pt")
            model = load_model_family("leworldmodel").build_model(config)
            bank = FixedBatches(pool(model), config, 33)
            bank.sampler._obs_keys = ("image", "physical_state")
            saved = {**checkpoint, "phase": "online", "replay_state": bank.sampler.state_dict()}
            torch.save(saved, run / "latest.pt")
            before = {p: p.read_bytes() for p in run.glob("*.pt")}
            args = options(run_root=root / "runs", dataset_root=root, output=root / "reports",
                           scenarios=["cartpole_balance_sparse"], models=["leworldmodel"], expert_train=1)
            with patch("scripts.diagnose_fixed_replay.arguments", return_value=args), \
                 patch("torch.save", side_effect=AssertionError("No checkpoint writes")), \
                 redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as status:
                main()
            report = json.loads((args.output / "report.json").read_text())
            self.assertEqual(status.exception.code, 0, report)
            self.assertEqual(len(report["results"][0]["trials"]), 3)
            for path, data in before.items():
                self.assertEqual(path.read_bytes(), data)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
