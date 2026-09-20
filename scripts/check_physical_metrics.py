"""Physical-unit evaluation contracts, shared by every model family.

Run with: python -m scripts.check_physical_metrics
No datasets, simulator, training runs, or checkpoint writes are needed.
"""

import math
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

from models.shared.physical_state import PhysicalStateTargets, format_physical_rmse
from scripts.check_state_normalization import settings
from training import load_model_family
from training.evaluation import Window, evaluate_state_prediction


def targets_for(scenario):
    config = settings(scenario)
    return PhysicalStateTargets(config.task, config.fields)


class PhysicalMetricsTest(unittest.TestCase):
    def test_wrap_boundary_and_units_for_cartpole_and_reacher(self):
        for scenario in ("cartpole_balance_sparse", "reacher"):
            targets = targets_for(scenario)
            prediction = torch.zeros(2, len(targets.coordinates))
            truth = prediction.clone()
            for cosine, sine in targets.angle_pairs.values():
                prediction[:, cosine] = 2 * math.cos(math.pi - .02)
                prediction[:, sine] = 2 * math.sin(math.pi - .02)
                truth[:, cosine] = math.cos(-math.pi + .03)
                truth[:, sine] = math.sin(-math.pi + .03)
            error = targets.metric_error(prediction, truth)
            result = targets.metric_summary(error.square())
            for name in targets.angle_pairs:
                self.assertAlmostEqual(result["physical_rmse"][name], .05, places=5)
                self.assertEqual(result["physical_units"][name], "rad")
            expected = "m/s" if scenario == "cartpole_balance_sparse" else "rad/s"
            self.assertEqual(result["physical_units"]["velocity[0]"], expected)
            self.assertNotIn("position[1]", result["physical_rmse"])

    def test_undefined_orientation_penalized_and_nonfinite_not_hidden(self):
        targets = targets_for("cartpole_balance_sparse")
        truth = torch.tensor([[0., 1., 0., 0., 0.]])
        prediction = truth.clone()
        prediction[:, 1:3] = 0
        self.assertAlmostEqual(targets.metric_error(prediction, truth)[0, -1].item(), math.pi, places=6)
        prediction[:, 1] = float("nan")
        self.assertTrue(torch.isnan(targets.metric_error(prediction, truth)[0, -1]))

    def test_circular_mean_uses_vectors_not_mean_of_angles(self):
        targets = targets_for("cartpole_balance_sparse")
        predictions = torch.tensor([[0, -math.cos(.01), math.sin(.01), 0, 0],
                                    [0, -math.cos(.01), -math.sin(.01), 0, 0]])
        truth = torch.tensor([[0., -1., 0., 0., 0.]])
        error = targets.metric_error(predictions.mean(0, keepdim=True), truth)
        torch.testing.assert_close(error, torch.zeros_like(error), atol=1e-6, rtol=0)

    def test_translation_units_rms_aggregation_and_format(self):
        targets = targets_for("ball_in_cup")
        errors = torch.tensor([[1.] * 8, [3.] * 8])
        result = targets.metric_summary(errors.square())
        for value in result["physical_rmse"].values():
            self.assertAlmostEqual(value, math.sqrt(5), places=6)
        self.assertEqual(list(result["physical_units"].values()), ["m"] * 4 + ["m/s"] * 4)
        formatted = format_physical_rmse(result, before=result, baseline=result)
        self.assertIn("position[0][m]=2.24->2.24 (hold=2.24)", formatted)

    def test_reordered_fields_do_not_change_angle_identity(self):
        for scenario in ("cartpole_balance_sparse", "reacher"):
            config = settings(scenario)
            original = targets_for(scenario)
            reordered = PhysicalStateTargets(config.task, dict(reversed(list(config.fields.items()))))
            for angle, pair in original.angle_pairs.items():
                self.assertEqual([original.coordinates[i] for i in pair],
                                 [reordered.coordinates[i] for i in reordered.angle_pairs[angle]])

    def test_full_evaluation_across_cpu_model_variants(self):
        root = str(Path(__file__).resolve().parents[1] / "configs")
        with initialize_config_dir(config_dir=root, version_base=None):
            smoke = compose(config_name="dmc_smoke")
            OmegaConf.resolve(smoke)
        for family, variants in smoke.models.items():
            for variant, entry in variants.items():
                with self.subTest(family=family, variant=variant):
                    if variant == "mamba3":
                        self.skipTest("Mamba3 requires CUDA kernels; metric aggregation is shared")
                    with initialize_config_dir(config_dir=root, version_base=None):
                        config = compose(config_name=entry.config, overrides=[
                            *smoke.training.overrides, *entry.overrides, "device=cpu",
                            "scenario=cartpole_balance_sparse", "evaluation.final.horizons=[1,3]",
                            "evaluation.final.state_windows=2", "evaluation.final.state_batch_size=4",
                            "evaluation.final.state_samples=2",
                        ])
                    model = load_model_family(family).build_model(config)
                    model.state_head.updates.fill_(1)
                    raw = torch.zeros(2, 6, 5)
                    raw[..., 0] = torch.arange(6) * .1
                    raw[..., 1] = 1
                    observation = {"image": torch.randint(0, 256, (2, 6, 64, 64, 3), dtype=torch.uint8)}
                    actions = torch.zeros(2, 5, 1)
                    windows = [Window(1, 0), Window(2, 0, "motion")]
                    dataset = SimpleNamespace(
                        episodes=np.array([1, 2]), sample_windows=lambda *args: windows,
                        batches=lambda *args: iter([(observation, actions, raw)]),
                    )
                    before = {name: value.clone() for name, value in model.state_dict().items()}
                    rng = torch.random.get_rng_state().clone()
                    with tempfile.TemporaryDirectory() as directory:
                        with h5py.File(Path(directory) / "data.hdf5", "w"):
                            pass
                        with patch("training.evaluation.StateDataset", return_value=dataset):
                            result = evaluate_state_prediction(model, config, directory, {})
                    self.assertTrue(model.training)
                    self.assertFalse(result["evaluation_fitting"])
                    self.assertEqual(result["physical_units"]["pole_angle"], "rad")
                    for horizon in (1, 3):
                        baseline = result["physical_true_persistence_rmse"][str(horizon)]
                        self.assertAlmostEqual(baseline["position[0]"], .1 * horizon, places=6)
                        self.assertEqual(baseline["pole_angle"], 0)
                        self.assertEqual(result["cohorts"]["motion"]["physical_true_persistence_rmse"][str(horizon)], baseline)
                        self.assertTrue(all(math.isfinite(v) for v in result["physical_rmse"][str(horizon)].values()))
                    self.assertIn("rmse", result)  # Existing raw-coordinate consumers keep working.
                    self.assertIn("derived_rmse", result)
                    self.assertIn("physical_persistence_rmse", result)
                    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
                    self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
