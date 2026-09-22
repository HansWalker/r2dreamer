"""CPU/simulator contracts for broader held-out starts and privileged controls."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from envs.dmc import make_env
from scripts.check_state_normalization import tiny_config
from scripts.paper_faithful_followup_support import (
    BOUNDS, COHORTS, JITTER, STATE_NAMES, followup_manifest_metadata,
    make_followup_manifest, policy_cases, simulator_controls, validate_followup_manifest,
)
from scripts.paper_faithful_support import _digest, collect_anchor, validate_manifest


class FollowupSupportTests(unittest.TestCase):
    def test_manifest_is_reproducible_disjoint_and_materially_diverse(self):
        manifest = make_followup_manifest()
        self.assertEqual(manifest, make_followup_manifest())
        self.assertNotEqual(manifest, make_followup_manifest(seed=30_000_001))
        validate_manifest(manifest)
        metadata = followup_manifest_metadata(manifest)
        self.assertEqual(metadata["distinct_sampled_states"], 56)
        self.assertEqual(metadata["distinct_realized_initial_states"], 56)
        self.assertEqual(len({c["seed"] for rows in manifest.values() for c in rows}), 56)
        all_states = []
        for split, cases in manifest.items():
            for cohort in COHORTS:
                values = np.asarray([c["sampled_state"] for c in cases if c["cohort"] == cohort])
                self.assertTrue((values.min(0) < 0).all())
                self.assertTrue((values.max(0) > 0).all())
                for axis, name in enumerate(STATE_NAMES):
                    absolute = name not in BOUNDS[cohort]
                    key = f"{name}_abs" if absolute else name
                    low, high = BOUNDS[cohort][key]
                    column = np.abs(values[:, axis]) if absolute else values[:, axis]
                    self.assertGreater(np.ptp(column), .4 * (high - low))
                self.assertEqual(metadata["material_diversity"][split][cohort]["anchors"], len(values))
            for case in cases:
                sign = -1 if case["seed"] % 2 else 1
                jitter = np.random.default_rng(case["seed"]).uniform(-1, 1, 4) * JITTER
                realized = sign * (np.asarray(case["nominal_state"]) + jitter)
                np.testing.assert_allclose(realized, np.asarray(case["sampled_state"]) + sign * jitter, atol=0, rtol=0)
                all_states.append(tuple(realized))
        self.assertEqual(len(set(all_states)), 56)

    def test_manifest_rejects_nominal_mismatch_duplicate_states_and_missing_signs(self):
        with self.assertRaisesRegex(ValueError, "at least six"):
            make_followup_manifest(validation_anchors=3)
        valid = make_followup_manifest(6, 6, 6)
        broken = copy.deepcopy(valid)
        broken["test"][0]["nominal_state"][0] += .1
        with self.assertRaisesRegex(ValueError, "mirror"):
            validate_followup_manifest(broken)
        broken = copy.deepcopy(valid)
        first, second = broken["test"][:2]
        second["sampled_state"] = first["sampled_state"].copy()
        second["nominal_state"] = (np.asarray(second["sampled_state"]) * (-1 if second["seed"] % 2 else 1)).tolist()
        with self.assertRaisesRegex(ValueError, "Repeated"):
            validate_followup_manifest(broken)
        broken = copy.deepcopy(valid)
        for case in broken["test"]:
            case["sampled_state"][2] = abs(case["sampled_state"][2])
            case["nominal_state"] = (np.asarray(case["sampled_state"]) * (-1 if case["seed"] % 2 else 1)).tolist()
        with self.assertRaisesRegex(ValueError, "both signs"):
            validate_followup_manifest(broken)

    def test_policy_subset_balances_cohorts_and_signs_without_mutation(self):
        cases = make_followup_manifest()["test"]
        before = _digest(cases)
        chosen = policy_cases(cases, 6)
        self.assertEqual([c["id"] for c in chosen], [c["id"] for c in policy_cases(cases[::-1], 6)])
        self.assertEqual({c["cohort"] for c in policy_cases(cases, 3)}, set(COHORTS))
        for cohort in COHORTS:
            angle = [c["sampled_state"][1] for c in chosen if c["cohort"] == cohort]
            self.assertEqual(len(angle), 2)
            self.assertLess(min(angle), 0)
            self.assertGreater(max(angle), 0)
        self.assertEqual(before, _digest(cases))
        with self.assertRaises(ValueError):
            policy_cases(cases, len(cases) + 1)

    def test_real_controls_match_stored_branches_and_freeze_sources(self):
        config = tiny_config("temporal_straightening", "cartpole_balance_sparse")
        config.env.time_limit = 1000
        spec = make_followup_manifest(6, 6, 6, seed=30_000_050)["test"][0]
        env = make_env(config.env, spec["seed"], include_physical_state=False)
        try:
            case = collect_anchor(env, spec, candidates=6, horizon=3)
        finally:
            env.close()
        cases = [case]
        source_before = _digest(cases)
        rng_before = torch.get_rng_state().clone()
        numpy_before = copy.deepcopy(np.random.get_state())
        with patch("training.load_model_family", side_effect=AssertionError("Native model access")):
            controls = simulator_controls(config, cases, 3)
        self.assertEqual(source_before, _digest(cases))
        torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        for mode, branch in (("zero", 0), ("privileged_state_feedback", 5)):
            result = controls[mode]
            self.assertEqual(result["steps"], 3)
            self.assertFalse(result["native_model_accessed"])
            self.assertEqual(result["uses_privileged_state"], mode != "zero")
            self.assertEqual(result["maximum_return"], 3 * int(config.env.action_repeat))
            self.assertEqual(result["returns"], [float(case["rewards"][branch].sum())])
            previous = case["anchor_state"]
            for step, trace in enumerate(result["traces"]):
                self.assertEqual(trace["step"], step + 1)
                np.testing.assert_allclose(trace["actions"][0], case["action"][branch, step], atol=0, rtol=0)
                np.testing.assert_allclose(trace["physical_state_before"][0], previous, atol=0, rtol=0)
                np.testing.assert_allclose(trace["physical_state_after"][0], case["states"][branch, step], atol=0, rtol=0)
                self.assertEqual(trace["rewards"], [float(case["rewards"][branch, step])])
                self.assertEqual(trace["success"], [bool(case["successes"][branch, step])])
                previous = trace["physical_state_after"][0]
            self.assertEqual(result["maintenance_occupancy"], [float(case["successes"][branch, -1])])
        invalid = copy.deepcopy(cases)
        invalid[0]["past_action"][0, 0] = 1
        with self.assertRaises(AssertionError):
            simulator_controls(config, invalid, 3)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
