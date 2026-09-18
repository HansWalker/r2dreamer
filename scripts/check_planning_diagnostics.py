"""CPU regression tests for held-out readout and representation diagnostics.

Run with: python -m scripts.check_planning_diagnostics
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from models.planning import LatentPlanner
from scripts.diagnose_planning_models import (
    analyze_batch,
    analyze_checkpoint,
    latent_statistics,
    summarize_batches,
    write_reports,
)
from scripts.smoke_models import synthetic_batch
from training import load_model_family
from training.evaluation import StateDataset, Window


class ToyReadout(nn.Module):
    history = 2
    coordinates = ("position[0]",)

    def __init__(self, bias=0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias))
        self.register_buffer("std", torch.tensor([2.0]))

    def forward(self, features):
        return features[:, self.history - 1:] + self.bias


class ToyPlanner(LatentPlanner):
    """Exactly solvable integrator; exercises the real latent_rollout dispatcher."""

    def __init__(self, bias=0.0, ignores_actions=False):
        nn.Module.__init__(self)
        self.history_size = 2
        self.state_head = ToyReadout(bias)
        self.ignores_actions = ignores_actions

    def encode(self, obs):
        return obs["image"].float().flatten(2)[..., :1]

    def rollout(self, history, past_action, candidates):
        increments = torch.zeros_like(candidates) if self.ignores_actions else candidates
        return history[:, None, -1:] + increments.cumsum(2)


def toy_batch(model, count=2):
    states = torch.arange(6.0)[None, :, None].expand(count, -1, -1)
    return analyze_batch(model, {"image": states}, torch.ones(count, 5, 1), states,
                         3, [1, 3], -torch.ones(count, 3, 1))


class PlanningDiagnosticsTest(unittest.TestCase):
    def test_rank_constant_rank_one_and_full_rank(self):
        constant = latent_statistics(torch.ones(8, 3))
        self.assertEqual(constant["effective_rank"], 0)
        self.assertEqual(constant["rms_std"], 0)
        self.assertEqual(constant["zero_variance_fraction"], 1)
        rank_one = torch.arange(8.0)[:, None] * torch.tensor([[1.0, -1.0, 0.0]])
        self.assertAlmostEqual(latent_statistics(rank_one)["effective_rank"], 1)
        isotropic = torch.tensor([[1.0, 0], [-1.0, 0], [0, 1.0], [0, -1.0]])
        stats = latent_statistics(isotropic)
        self.assertAlmostEqual(stats["effective_rank"], 2)
        self.assertAlmostEqual(stats["participation_ratio"], 2)
        shifted = latent_statistics(3 * isotropic + 100)
        self.assertAlmostEqual(shifted["effective_rank"], stats["effective_rank"])
        self.assertAlmostEqual(shifted["rms_std"], 3 * stats["rms_std"])

    def test_ordered_patches_not_spatially_averaged(self):
        values = torch.arange(8.0)[:, None, None]
        patches = torch.cat((values, -values), dim=1)
        self.assertEqual(patches.mean(1).abs().max(), 0)
        self.assertGreater(latent_statistics(patches)["rms_std"], 0)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            latent_statistics(torch.full((2, 3), float("nan")))

    def test_perfect_dynamics_and_persistence_alignment(self):
        batch = toy_batch(ToyPlanner())
        for name in ("observed", "forecast"):
            torch.testing.assert_close(batch["errors"][name], torch.zeros(2, 2, 1))
        for name in ("decoded_persistence", "true_persistence"):
            torch.testing.assert_close(batch["errors"][name], torch.tensor([[[1.0], [9.0]]]).expand(2, -1, -1))
        torch.testing.assert_close(batch["latent_error"], torch.zeros(2, 2))
        torch.testing.assert_close(batch["temporal"], torch.ones(2))

    def test_bad_readout_is_separate_from_good_dynamics(self):
        batch = toy_batch(ToyPlanner(bias=2.0))
        torch.testing.assert_close(batch["errors"]["observed"], torch.full((2, 2, 1), 4.0))
        torch.testing.assert_close(batch["errors"]["forecast"], batch["errors"]["observed"])
        torch.testing.assert_close(batch["latent_error"], torch.zeros(2, 2))

    def test_action_response_detects_action_blind_predictor(self):
        responsive = toy_batch(ToyPlanner())
        for name, expected in (("zero", [1.0, 9.0]), ("random", [4.0, 36.0])):
            torch.testing.assert_close(responsive["responses"][name]["latent_squared_delta"],
                                       torch.tensor([expected]).expand(2, -1))
        blind = toy_batch(ToyPlanner(ignores_actions=True))
        for response in blind["responses"].values():
            self.assertGreater(response["action_squared_delta"].min(), 0)
            self.assertEqual(response["latent_squared_delta"].abs().max(), 0)

    def test_cohorts_normalization_and_rms_aggregation(self):
        model = ToyPlanner(bias=2.0)
        batch = toy_batch(model)
        batch["errors"]["observed"][1] = 16
        result = summarize_batches([batch], [Window(1, 0), Window(2, 0, "motion")], model.state_head, [1, 3])
        all_metrics = result["all"]["physical"]["observed"]["1"]
        self.assertAlmostEqual(all_metrics["rmse"]["position[0]"], 10 ** 0.5, places=6)
        self.assertEqual(all_metrics["mean_normalized_mse"], 2.5)
        self.assertEqual(all_metrics["normalized_loss_fraction"]["position[0]"], 1)
        self.assertEqual(result["uniform"]["physical"]["observed"]["1"]["mean_normalized_mse"], 1)
        self.assertEqual(result["motion"]["physical"]["observed"]["1"]["mean_normalized_mse"], 4)

    def test_heldout_reader_batch_invariance_and_reports(self):
        with tempfile.TemporaryDirectory() as temporary, h5py.File(Path(temporary) / "data.hdf5", "w") as h5:
            states = np.broadcast_to(np.arange(8)[None, :, None], (3, 8, 1)).astype(np.float32)
            h5["observations"] = states
            h5["images"] = states[..., None, None].astype(np.uint8)
            h5["actions"] = np.ones((3, 7, 1), np.float32)
            h5["complete"] = [1, 1, 1]
            h5["lengths"] = [7, 7, 7]
            metadata = {"observation_keys": ["position"], "observation_shapes": {"position": [1]},
                        "episode_splits": {"train": [0, 1], "heldout": [1, 3]}}
            model_io = OmegaConf.create({"observations": {"image": [1, 1, 1]},
                                        "action": {"shape": [1], "kind": "continuous"}})
            dataset = StateDataset(h5, metadata, model_io, {"position": [0]},
                                   SimpleNamespace(encode=lambda x: x, positions=[0]))
            windows = dataset.sample_windows(2, 6, 7, 3, 0.5, 8)
            self.assertEqual({window.episode for window in windows}, {1, 2})
            with self.assertRaisesRegex(ValueError, "never training"):
                dataset.read_batch([Window(0, 0)], 6)
            args = SimpleNamespace(context_length=3, horizons=[1, 3], window_seed=7, batch_size=1)
            one = analyze_checkpoint(ToyPlanner(), dataset, windows, args)
            args.batch_size = 2
            two = analyze_checkpoint(ToyPlanner(), dataset, windows, args)
            self.assertEqual(one, two)
            record = {"scenario": "toy", "model": "toy", "checkpoint_name": "final.pt",
                      "readout_std": {"position[0]": 2}, "cohorts": one}
            summary = write_reports(Path(temporary), [record], args)
            self.assertIn("toy/toy/final.pt", summary)
            report = json.loads((Path(temporary) / "report.json").read_text())
            self.assertFalse(report["evaluation_fitting"])
            with (Path(temporary) / "physical_errors.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3 * 4 * 2)

    def test_real_models_no_mutation_fitting_or_planning(self):
        for family in ("leworldmodel", "temporal_straightening"):
            with self.subTest(family=family):
                with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
                    config = compose(config_name=f"{family}_dmc_vision", overrides=["device=cpu", "scenario=ball_in_cup"])
                OmegaConf.resolve(config)
                torch.manual_seed(7)
                model = load_model_family(family).build_model(config).eval()
                batch, obs, action = synthetic_batch(config, model, batch_size=2, length=5)
                labels = batch[0]["physical_state"]
                before = {name: value.clone() for name, value in model.state_dict().items()}
                with patch.object(model.state_head, "fit", side_effect=AssertionError("Fitting called")), \
                     patch.object(model, "act", side_effect=AssertionError("Planning called")):
                    result = analyze_batch(model, obs, action[:, :-1], labels, 3, [1, 2], torch.zeros(2, 2, 2))
                    altered = {"image": obs["image"].clone()}
                    altered["image"][:, 3:] = 255 - altered["image"][:, 3:]
                    other = analyze_batch(model, altered, action[:, :-1], labels, 3, [1, 2], torch.zeros(2, 2, 2))
                torch.testing.assert_close(result["errors"]["forecast"], other["errors"]["forecast"], rtol=0, atol=0)
                self.assertTrue(torch.isfinite(result["errors"]["forecast"]).all())
                self.assertFalse(result["errors"]["forecast"].requires_grad)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
