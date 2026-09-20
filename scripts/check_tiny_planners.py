"""Fast guards for the diagnostic settings and planner/readout separation."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir

from scripts.smoke_tiny_planners import FAMILIES, native_control, tiny_config
from training import load_model_family


class TinyPlannerTest(unittest.TestCase):
    def test_small_models_keep_native_objectives_and_optimizers(self):
        for name in FAMILIES:
            for scenario in ("cartpole_balance_sparse", "reacher", "ball_in_cup"):
                with self.subTest(model=name, scenario=scenario):
                    args = SimpleNamespace(scenario=scenario, device="cpu", seed=0, episode_steps=64,
                                           offline_updates=128, online_updates=128)
                    config = tiny_config(name, args)
                    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
                                               version_base=None):
                        production = compose(config_name=f"{name}_dmc_vision", overrides=[f"scenario={scenario}"])
                    for key in ("optim", "sigreg", "curvature_weight", "goal", "history_size"):
                        self.assertEqual(config.jepa_model[key], production.jepa_model[key])
                    self.assertEqual(config.state_head.online, production.state_head.online)
                    self.assertEqual(config.env.model_io, production.env.model_io)
                    model = load_model_family(name).build_model(config)
                    self.assertLess(sum(p.numel() for p in model.parameters()), 25000)
                    self.assertEqual(model.state_head.samples_per_update, 16)
                    self.assertEqual(model.sequence_length, 4)

    def test_control_tripwire_restores_physical_head_after_error(self):
        head = torch.nn.Linear(2, 2)
        model = SimpleNamespace(state_head=head)
        features = torch.ones(1, 2)
        before = head(features)
        with self.assertRaisesRegex(RuntimeError, "Planner used"):
            native_control(model, lambda: head(features))
        torch.testing.assert_close(head(features), before)
        self.assertEqual(native_control(model, lambda: 1), 1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
