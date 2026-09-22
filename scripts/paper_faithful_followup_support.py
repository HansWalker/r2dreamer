"""Broader controlled starts and simulator controls for the offline follow-up.

This changes diagnostic data support only. It does not modify native model losses,
optimizers, online schedules, or the original narrow-run artifacts. Bounds describe
the sampled state before the existing collector adds its small independent jitter;
the anchor is reached after the stored observed prefix, so its bounds can differ.
"""

import copy
import math
from contextlib import ExitStack

import numpy as np
import torch

import tools
from envs.dmc import make_env
from scripts.diagnose_goal_objective import cart_state, feedback_gain, set_cart_state
from scripts.diagnose_planner_oracle import simulator_branch
from scripts.paper_faithful_support import _digest, validate_manifest


COHORTS = ("balanced", "boundary", "recoverable")
STATE_NAMES = ("cart_position", "pole_angle", "cart_velocity", "pole_velocity")
STATE_UNITS = ("metres", "radians", "metres_per_second", "radians_per_second")
JITTER = np.asarray([.005, .002, .005, .01])
BOUNDS = {
    "balanced": {"cart_position": [-.20, .20], "pole_angle": [-.06, .06],
                 "cart_velocity": [-.25, .25], "pole_velocity": [-.50, .50]},
    "boundary": {"cart_position": [-.27, .27], "pole_angle_abs": [.075, .115],
                 "cart_velocity": [-.35, .35], "pole_velocity": [-.70, .70]},
    "recoverable": {"cart_position": [-.35, .35], "pole_angle_abs": [.12, .25],
                    "cart_velocity": [-.50, .50], "pole_velocity": [-1., 1.]},
}


def _stratified(generator, count, bounds):
    # Independently permuted one-dimensional strata avoid tying velocity to angle.
    quantiles = (np.arange(count) + generator.uniform(.2, .8, count)) / count
    generator.shuffle(quantiles)
    low, high = bounds
    return low + (high - low) * quantiles


def make_followup_manifest(train_anchors=32, validation_anchors=12, test_anchors=12, seed=30_000_000):
    """Disjoint seed/anchor splits with >=2 samples per cohort and both state signs."""
    counts = tuple(map(int, (train_anchors, validation_anchors, test_anchors)))
    if min(counts) < 6 or int(seed) < 0:
        raise ValueError("Use at least six anchors per split for both signs in all three cohorts, and a nonnegative seed.")
    manifest, offset = {}, 0
    for split_index, (split, count) in enumerate(zip(("train", "validation", "test"), counts, strict=True)):
        states = {}
        for cohort_index, cohort in enumerate(COHORTS):
            indices = list(range(cohort_index, count, len(COHORTS)))
            generator = np.random.default_rng(np.random.SeedSequence([int(seed), split_index, cohort_index, 937]))
            bounds = BOUNDS[cohort]
            columns = []
            for name in STATE_NAMES:
                key = name if name in bounds else f"{name}_abs"
                column = _stratified(generator, len(indices), bounds[key])
                if key.endswith("_abs"):
                    signs = np.where(np.arange(len(indices)) % 2, -1., 1.)
                    generator.shuffle(signs)
                    column *= signs
                columns.append(column)
            for index, state in zip(indices, np.stack(columns, axis=1), strict=True):
                states[index] = state
        rows = []
        for index in range(count):
            current_seed = int(seed) + offset + index
            cohort = COHORTS[index % len(COHORTS)]
            sampled = states[index]
            # collect_anchor mirrors odd seeds after adding jitter. Cancel its
            # mirror for the designed state, preserving the collector unchanged.
            mirror = -1. if current_seed % 2 else 1.
            rows.append({"id": f"followup/{current_seed}/{cohort}", "seed": current_seed,
                         "split": split, "profile": cohort, "cohort": cohort,
                         "sampled_state": sampled.tolist(), "nominal_state": (mirror * sampled).tolist(),
                         "state_sampling_version": "stratified_continuous_v1"})
        manifest[split] = rows
        offset += count
    validate_followup_manifest(manifest)
    return manifest


def validate_followup_manifest(manifest):
    """Reject leakage, reused states, missing signs, or nearly identical starts."""
    validate_manifest(manifest)
    states_seen = set()
    for cases in manifest.values():
        for case in cases:
            if case["cohort"] not in COHORTS or case["profile"] != case["cohort"]:
                raise ValueError("Follow-up profiles must use the three documented cohorts.")
            state = np.asarray(case["sampled_state"], dtype=np.float64)
            if state.shape != (4,) or not np.isfinite(state).all():
                raise ValueError("Follow-up sampled states must be finite four-dimensional physical states.")
            expected = state * (-1. if int(case["seed"]) % 2 else 1.)
            if not np.array_equal(expected, case["nominal_state"]):
                raise ValueError("Nominal state must compensate for the existing seed-parity mirror.")
            key = tuple(state)
            if key in states_seen:
                raise ValueError("Repeated sampled states are not independent held-out starts.")
            states_seen.add(key)
        for cohort in COHORTS:
            values = np.asarray([c["sampled_state"] for c in cases if c["cohort"] == cohort])
            if len(values) < 2:
                raise ValueError("Each split needs at least two independent anchors per cohort.")
            for axis, name in enumerate(STATE_NAMES):
                column = values[:, axis]
                if not (column.min() < 0 < column.max()):
                    raise ValueError(f"Cohort {cohort} must cover both signs of {name}.")
                bounds = BOUNDS[cohort]
                absolute = name not in bounds
                low, high = bounds[f"{name}_abs" if absolute else name]
                checked = np.abs(column) if absolute else column
                if checked.min() < low or checked.max() > high:
                    raise ValueError(f"Cohort {cohort} exceeds the documented {name} bounds.")
                if np.ptp(checked) < .15 * (high - low):
                    raise ValueError(f"Cohort {cohort} has insufficient material {name} diversity.")


def followup_manifest_metadata(manifest):
    validate_followup_manifest(manifest)
    realized = []
    spread = {}
    for split, cases in manifest.items():
        spread[split] = {}
        for case in cases:
            jitter = np.random.default_rng(int(case["seed"])).uniform(-1, 1, 4) * JITTER
            state = np.asarray(case["nominal_state"]) + jitter
            realized.append(tuple(state * (-1. if int(case["seed"]) % 2 else 1.)))
        for cohort in COHORTS:
            values = np.asarray([c["sampled_state"] for c in cases if c["cohort"] == cohort])
            spread[split][cohort] = {"anchors": len(values), "minimum": values.min(0).tolist(),
                                     "maximum": values.max(0).tolist(), "range": np.ptp(values, axis=0).tolist()}
    if len(set(realized)) != len(realized):
        raise ValueError("Collector jitter produced duplicate initial states.")
    return {"version": "stratified_continuous_v1", "state_order": list(STATE_NAMES),
            "state_units": list(STATE_UNITS), "sampled_state_bounds": copy.deepcopy(BOUNDS),
            "collector_jitter_max_abs": JITTER.tolist(), "seed_parity_mirror_compensated": True,
            "sampling": "Independent permutations of continuous within-cohort marginal strata for each split; angle signs balanced for unsigned magnitude bands.",
            "split_unit": "Simulator seed and sampled anchor assigned before generating any sibling branches",
            "same_support_across_splits": True, "distinct_sampled_states": len(realized),
            "distinct_realized_initial_states": len(set(realized)), "material_diversity": spread,
            "manifest_sha256": _digest(manifest),
            "scope": "Held-out continuous starts within these prescribed bounds, not arbitrary-state, own-policy, or ordinary-reset generalization.",
            "cohort_note": "Cohorts describe sampled initial states before prefix drift. Recoverable is a near-failure label, not a recovery guarantee; compare the privileged simulator feedback control."}


def policy_cases(cases, count):
    """Stable round-robin cohort subset; two per cohort span both angle signs."""
    count = int(count)
    if count < 1 or count > len(cases) or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Policy selection needs a positive available count and distinct anchor IDs.")
    queues = {}
    for cohort_index, cohort in enumerate(COHORTS):
        rows = sorted((c for c in cases if c["cohort"] == cohort), key=lambda c: (int(c["seed"]), c["id"]))
        preferred_sign = -1 if cohort_index % 2 else 1
        order = []
        for sign in (preferred_sign, -preferred_sign):
            chosen = next((c for c in rows if sign * (c["initial_state"] if "initial_state" in c
                                                      else c["sampled_state"])[1] > 0), None)
            if chosen is not None:
                order.append(chosen)
        chosen_ids = {c["id"] for c in order}
        queues[cohort] = order + [c for c in rows if c["id"] not in chosen_ids]
    selected = []
    while len(selected) < count:
        before = len(selected)
        for cohort in COHORTS:
            if queues[cohort] and len(selected) < count:
                selected.append(queues[cohort].pop(0))
        if len(selected) == before:
            raise ValueError("Unknown cohorts cannot be used in the standardized policy subset.")
    return selected


def _restore_case(env, case):
    env.reset()
    set_cart_state(env, case["initial_state"])
    frames = [env.render().copy()]
    for action in case["past_action"].cpu().numpy():
        observation, _, done, _ = env.step(action)
        if done:
            raise ValueError("Observed control prefix crossed an episode boundary.")
        frames.append(observation["image"].copy())
    torch.testing.assert_close(torch.from_numpy(np.stack(frames)), case["prefix"], rtol=0, atol=0)
    np.testing.assert_allclose(cart_state(env), case["anchor_state"], rtol=0, atol=1e-12)


@tools.preserve_rng_state
def simulator_controls(config, cases, steps):
    """Zero and clipped simulator-state LQR controls on exact native-policy starts.

    No native model is accepted or called. Feedback is privileged and is not a
    learned-policy baseline or an optimal-return upper bound. Future image renders
    are unnecessary; full action/reward/success/state traces remain available.
    """
    steps = int(steps)
    if not cases or steps < 1 or str(config.env.task) != "dmc_cartpole_balance_sparse":
        raise ValueError("Require Cartpole cases and a positive control duration.")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Control anchors must be distinct.")
    before = _digest(cases)
    envs, gains = [], []
    try:
        for case in cases:
            env = make_env(config.env, int(case["seed"]), include_physical_state=False)
            envs.append(env)
            _restore_case(env, case)
            if env._episode_step + steps > env._max_steps:
                raise ValueError("Control duration would cross an episode boundary.")
            gain = np.asarray(case["feedback_gain"] if "feedback_gain" in case else feedback_gain(env))
            if gain.shape != (4,) or not np.isfinite(gain).all():
                raise ValueError("Require a finite four-dimensional feedback gain.")
            gains.append(gain)
        results = {}
        for mode in ("zero", "privileged_state_feedback"):
            returns, successes, traces = np.zeros(len(cases)), [], []
            with ExitStack() as stack:
                branches = [stack.enter_context(simulator_branch(env)) for env in envs]
                for branch in branches:
                    branch._observation = lambda _time_step: {}
                for step in range(steps):
                    actions, rewards, good, pre, post = [], [], [], [], []
                    for branch, gain in zip(branches, gains, strict=True):
                        state = cart_state(branch)
                        action = np.asarray([0. if mode == "zero" else np.clip(-(gain @ state), -1., 1.)], dtype=np.float32)
                        _, reward, done, _ = branch.step(action)
                        if done and step != steps - 1:
                            raise ValueError("Simulator control reached an early episode boundary.")
                        actions.append(action.tolist())
                        rewards.append(float(reward))
                        good.append(float(branch._env.task.get_reward(branch._env.physics)) >= 1 - 1e-6)
                        pre.append(state.tolist())
                        post.append(cart_state(branch).tolist())
                    returns += rewards
                    successes.append(good)
                    traces.append({"step": step + 1, "actions": actions, "rewards": rewards, "success": good,
                                   "physical_state_before": pre, "physical_state_after": post})
            tail = max(1, math.ceil(.2 * steps))
            occupancy = np.asarray(successes[-tail:]).mean(0)
            results[mode] = {"steps": steps, "case_ids": [c["id"] for c in cases], "returns": returns.tolist(),
                             "return_mean": float(returns.mean()), "maximum_return": steps * int(config.env.action_repeat),
                             "tail_steps": tail, "maintenance_occupancy": occupancy.tolist(),
                             "maintenance_rate": float((occupancy >= .9).mean()), "traces": traces,
                             "uses_privileged_state": mode != "zero", "native_model_accessed": False,
                             "scope": "Controlled starts matching the stored image/action prefix; feedback is a privileged feasibility diagnostic, not an optimal oracle."}
            for env, case in zip(envs, cases, strict=True):
                np.testing.assert_allclose(cart_state(env), case["anchor_state"], rtol=0, atol=1e-12)
                if env._episode_step != len(case["past_action"]):
                    raise RuntimeError("A simulator control changed its parent's episode counter.")
        if _digest(cases) != before:
            raise RuntimeError("Simulator controls mutated the source cases.")
        return results
    finally:
        for env in envs:
            env.close()
