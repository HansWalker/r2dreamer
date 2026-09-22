"""CPU checks; PAPER_REFERENCE_CACHE selects an already downloaded pinned cache."""

import copy
import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf, open_dict

from scripts.check_state_normalization import tiny_config
from scripts.paper_faithful_reference import (
    LEWM_COMMIT, ReferenceUnavailable, lewm_parity, pinned_sources, run_reference_checks,
)
from training import load_model_family


def configurations():
    result = {}
    for label, family, mode in (("lewm", "leworldmodel", None), ("ts_patch", "temporal_straightening", "patch"), ("ts_agg", "temporal_straightening", "agg")):
        config = tiny_config(family, "cartpole_balance_sparse")
        OmegaConf.resolve(config)
        if mode:
            with open_dict(config):
                config.jepa_model.curvature_mode = mode
                config.jepa_model.aggregation = {"hidden_dim": 16, "output_dim": 8}
        result[label] = config
    return result


class SourceChecks(unittest.TestCase):
    def test_original_launcher_preserves_existing_results(self):
        from scripts.run_upstream_reference import main
        with tempfile.TemporaryDirectory() as root, patch("scripts.run_upstream_reference.subprocess.run") as run, redirect_stderr(io.StringIO()):
            sentinel = Path(root) / "manifest.json"
            sentinel.write_text("previous results")
            with self.assertRaises(SystemExit):
                main(["--checkpoint-dir", root + "/weights", "--dataset-root", root + "/data", "--output", root])
            self.assertEqual(sentinel.read_text(), "previous results")
            run.assert_not_called()

    def test_original_task_launcher_does_not_claim_success_or_evaluate_missing_assets(self):
        from scripts.run_upstream_reference import main
        probe = SimpleNamespace(returncode=0, stdout=json.dumps({"cuda_available": False, "versions": {"torch": "test"}, "api": {}}), stderr="")
        with tempfile.TemporaryDirectory() as root, patch("scripts.run_upstream_reference.subprocess.run", return_value=probe) as run, redirect_stdout(io.StringIO()):
            code = main(["--source-root", root + "/source", "--checkpoint-dir", root + "/weights",
                         "--dataset-root", root + "/data", "--output", root + "/out", "--run"])
            report = json.loads((Path(root) / "out/manifest.json").read_text())
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "NOT_RUN")
        self.assertEqual(run.call_count, 1)  # Only runtime probing; no conversion/evaluation.
        self.assertTrue(report["prerequisites_missing"])

    def test_missing_offline_cache_is_explicit_and_never_downloads(self):
        with tempfile.TemporaryDirectory() as cache, patch("scripts.paper_faithful_reference.urlopen", side_effect=AssertionError("network")):
            with self.assertRaises(ReferenceUnavailable):
                pinned_sources("leworldmodel", cache, False)
            report = run_reference_checks({"lewm": configurations()["lewm"]}, cache=cache, allow_download=False)
        self.assertEqual(report["status"], "UNAVAILABLE")
        self.assertEqual(report["original_task"]["status"], "NOT_RUN")

    def test_corrupt_cached_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as cache:
            path = Path(cache) / "leworldmodel" / LEWM_COMMIT / "module.py"
            path.parent.mkdir(parents=True)
            path.write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                pinned_sources("leworldmodel", cache, False)


class PinnedComponentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = Path(os.environ.get("PAPER_REFERENCE_CACHE", "local/reference_sources"))
        try:
            cls.lewm, _ = pinned_sources("leworldmodel", cls.cache, False)
            pinned_sources("temporal_straightening", cls.cache, False)
        except ReferenceUnavailable as error:
            raise unittest.SkipTest(str(error)) from error

    def test_actual_upstream_components_and_gradients_all_modes(self):
        configs = configurations()
        configs["ts_agg_coverage"] = copy.deepcopy(configs["ts_agg"])
        before = {k: OmegaConf.to_container(v, resolve=True) for k, v in configs.items()}
        state = torch.random.get_rng_state().clone()
        with patch("scripts.paper_faithful_reference.urlopen", side_effect=AssertionError("network")):
            report = run_reference_checks(configs, SimpleNamespace(device="cpu", seed=17), cache=self.cache, allow_download=False)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["original_task"]["status"], "NOT_RUN")
        self.assertEqual(report["models"]["ts_agg_coverage"]["reused_component_check"], "ts_agg")
        self.assertEqual(before, {k: OmegaConf.to_container(v, resolve=True) for k, v in configs.items()})
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for family in ("ts_patch", "ts_agg", "lewm"):
            self.assertTrue(report["models"][family]["provenance"]["files"])
        checks = report["models"]["ts_agg"]["components"]["loss_and_step"]["checks"]
        self.assertIn("aggregation_output", checks)
        self.assertIn("optimizer_step_group_2", checks)

    def test_sigreg_mismatch_is_detected(self):
        model = load_model_family("leworldmodel").build_model(configurations()["lewm"])
        original = type(model.sigreg).forward
        with patch.object(type(model.sigreg), "forward", lambda self, value: 2 * original(self, value)):
            result = lewm_parity(model, self.lewm)
        self.assertEqual(result["status"], "MISMATCH")
        self.assertFalse(result["checks"]["sigreg_loss"]["pass"])


if __name__ == "__main__":
    unittest.main()
