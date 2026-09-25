"""CPU checks of common endpoints, frozen inference and future-image isolation."""
import unittest
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.analyze_world_model_capabilities import MODELS
from scripts.evaluate_history_context import crop_history, evaluate_batch, tensor_digest, summarize, main
from training import load_model_family


class HistoryContextTest(unittest.TestCase):
    def test_failed_run_keeps_name_and_status(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'named_test'
            log = io.StringIO()
            with patch('scripts.evaluate_history_context.preflight', side_effect=FileNotFoundError('missing final.pt')):
                with redirect_stdout(log), self.assertRaises(FileNotFoundError):
                    main(['--output', str(output)])
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['status'], 'FAILED')
            self.assertEqual(report['models'], [])
            self.assertIn('Run: named_test | Status: FAILED', log.getvalue())

    def test_rmse_uses_squared_errors_and_separate_cohorts(self):
        errors = {key: torch.tensor([[[1.]], [[9.]]])
                  for key in ('current', 'forecast', 'observed', 'decoded_hold', 'true_hold')}
        windows = [SimpleNamespace(cohort='uniform'), SimpleNamespace(cohort='motion')]
        result = summarize(errors, windows, [1], ['x'])
        self.assertAlmostEqual(result['all']['forecast']['1']['x'], 5 ** .5, places=6)
        self.assertEqual(result['uniform']['current']['0']['x'], 1.)
        self.assertEqual(result['motion']['current']['0']['x'], 3.)

    def test_common_endpoint(self):
        x = torch.arange(89).reshape(1, 89, 1)
        for context in (4, 16, 64):
            obs, actions, truth = crop_history({'image': x}, x[:, :-1], x, 64, context)
            self.assertEqual(obs['image'][0, context - 1].item(), 63)
            self.assertEqual(actions[0, context - 1].item(), 63)
            torch.testing.assert_close(truth[:, context:], x[:, 64:])

    def test_cpu_backends_frozen_and_no_future_image_leak(self):
        torch.set_num_threads(1)
        for name in MODELS:
            if 'mamba3' in name:
                continue  # Native Mamba3 kernels require the Lambda GPU environment.
            with self.subTest(model=name):
                family_name, variant = name.split('/')
                with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'configs'), version_base=None):
                    smoke = compose(config_name='dmc_smoke')
                    OmegaConf.resolve(smoke)
                    entry = smoke.models[family_name][variant]
                    config = compose(config_name=entry.config, overrides=[
                        *smoke.training.overrides, *entry.overrides, 'device=cpu', 'scenario=cartpole_balance_sparse'])
                model = load_model_family(family_name).build_model(config).eval().requires_grad_(False)
                before = tensor_digest(model.state_dict())
                obs = {str(k): torch.randint(256, (1, 89, *map(int, shape)), dtype=torch.uint8)
                       for k, shape in config.model_io.observations.items()}
                actions = torch.rand(1, 88, 1) * 2 - 1
                truth = torch.randn(1, 89, len(model.state_head.coordinates))
                conditions = {}
                for context in (4, 16, 64):
                    cropped = crop_history(obs, actions, truth, 64, context)
                    rng = torch.get_rng_state().clone()
                    values = evaluate_batch(model, config, *cropped, context=context, horizons=[1, 5, 25], samples=2, seed=71)
                    torch.testing.assert_close(rng, torch.get_rng_state())
                    altered = {k: v.clone() for k, v in cropped[0].items()}
                    for value in altered.values():
                        value[:, context:] = 0
                    changed = evaluate_batch(model, config, altered, cropped[1], cropped[2] + 3,
                                             context=context, horizons=[1, 5, 25], samples=2, seed=71)
                    for key in ('forecast_values', 'current_values'):
                        torch.testing.assert_close(values[key], changed[key], atol=0, rtol=0)
                    self.assertEqual(values['forecast'].shape, (1, 3, 4))
                    conditions[context] = values
                if family_name in ('tdmpc2', 'leworldmodel', 'temporal_straightening'):
                    for context in (4, 16):
                        torch.testing.assert_close(conditions[context]['forecast_values'], conditions[64]['forecast_values'], atol=2e-4, rtol=2e-4)
                self.assertEqual(before, tensor_digest(model.state_dict()))
                print(f'PASS: {name}', flush=True)


if __name__ == '__main__':
    unittest.main()
