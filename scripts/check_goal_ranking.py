"""CPU checks of goal supervision, all three tasks, and real offline/online wiring."""

import copy
import io
import json
import math
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from envs.dmc import make_env
from models.shared.goal_ranking import goal_ranking_loss
from models.shared.latent_goal import latent_goal_cost
from models.shared.physical_state import PhysicalStateTargets
from scripts import check_forecast_online as fixtures
from scripts.check_state_normalization import tiny_config
from scripts.goal_ranking_support import GoalPairBank, evaluate_goal_pairs, goal_error
from scripts.paper_faithful_duration_support import collect_bank
from scripts.smoke_models import synthetic_batch
from scripts.train_forecast_online import main
from scripts.train_paper_faithful_duration import file_hash
from training import load_model_family


FAMILIES = ("temporal_straightening", "leworldmodel")
TASKS = ("cartpole_balance_sparse", "reacher", "ball_in_cup")


def real_pair(config, seed=17, split="train"):
    """Render a successful and clearly failed pose with exact simulator labels."""
    env = make_env(config.env, seed, include_physical_state=True)
    try:
        observation = env.reset()
        physics = env._goal_renderer.physics
        origin = physics.data.qpos.copy()
        frames, states = [], []
        targets = PhysicalStateTargets(config.state_head.task, config.state_head.fields)
        for failed in (False, True):
            with physics.reset_context():
                physics.data.qpos[:] = origin
                if config.scenario.name == "cartpole_balance_sparse":
                    physics.named.data.qpos["hinge_1"] += .4 if failed else .03
                elif config.scenario.name == "reacher":
                    if failed:
                        target = physics.named.model.geom_pos["target", :2] - physics.named.model.body_pos["arm", :2]
                        physics.named.data.qpos["shoulder"] = np.arctan2(target[1], target[0]) + np.pi
                        physics.named.data.qpos["wrist"] = 0.
                    else:
                        physics.named.data.qpos["shoulder"] += .03
                else:
                    physics.named.data.qpos["ball_x"] += .15 if failed else .01
            truth = env._env.task.get_observation(physics)
            raw = np.concatenate([np.asarray(truth[key]).reshape(-1)[list(indices)]
                                  for key, indices in config.state_head.fields.items()])
            states.append(torch.from_numpy(targets.encode(raw)))
            frames.append(torch.from_numpy(physics.render(*env._size, camera_id=env._camera).copy()))
            assert bool(env._env.task.get_reward(physics) >= 1 - 1e-6) != failed
        return {"id": f"{config.scenario.name}/{split}/{seed}", "split": split,
                "image": torch.stack(frames)[None], "states": torch.stack(states)[None],
                "goal_image": torch.from_numpy(observation["goal_image"].copy())}
    finally:
        env.close()


def configured(family, task="cartpole_balance_sparse", weight=.1):
    config = tiny_config(family, task)
    config.state_head.samples_per_update = 4
    config.jepa_model.goal_ranking.weight = weight
    return config, load_model_family(family).build_model(config)


class GoalRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_wrong_order_receives_correct_gradient_and_matching_native_cost(self):
        # Deliberately wrong learned geometry; goal is row 2, detached by native cost.
        for reduction in ("mean", "sum"):
            encoder = nn.Embedding(3, 2)
            with torch.no_grad():
                encoder.weight.copy_(torch.tensor([[2., 2.], [1., 1.], [0., 0.]]))
            model = SimpleNamespace(encoder=encoder, projector=nn.Identity(), device="cpu",
                                    goal_reduction=reduction, _aggregate_goal_weight=lambda: 0.)
            model.encode = lambda obs: encoder(obs["image"][..., 0, 0, 0].long())
            images = torch.arange(3).reshape(1, 3, 1, 1, 1)
            loss, metrics = goal_ranking_loss(model, images, .1)
            self.assertEqual(float(metrics["goal_ranking/pair_accuracy"]), 0.)
            optimizer = torch.optim.SGD(encoder.parameters(), lr=.1)
            optimizer.zero_grad()
            loss.backward()
            self.assertTrue(torch.all(encoder.weight.grad[0] > 0))  # Pull good toward goal.
            self.assertTrue(torch.all(encoder.weight.grad[1] < 0))  # Separate bad from goal.
            self.assertEqual(encoder.weight.grad[2].abs().sum(), 0)
            optimizer.step()
            later, _ = goal_ranking_loss(model, images, .1)
            self.assertLess(float(later.detach()), float(loss.detach()))
            for _ in range(12):
                optimizer.zero_grad()
                later, _ = goal_ranking_loss(model, images, .1)
                later.backward()
                optimizer.step()
            latent = model.encode({"image": images})
            native = latent_goal_cost(latent[:, :2, None], latent[:, 2], mode="last", reduction=reduction)
            self.assertLess(float(native[0, 0].detach()), float(native[0, 1].detach()))

    def test_real_simulator_geometry_and_same_episode_goals_for_all_tasks(self):
        for task in TASKS:
            with self.subTest(task=task):
                config, _ = configured("leworldmodel", task)
                cases = [real_pair(config, seed) for seed in (17, 18)]
                targets = PhysicalStateTargets(config.state_head.task, config.state_head.fields)
                for case in cases:
                    errors = goal_error(case["states"], targets, config.jepa_model.goal.geometry,
                                        list(config.jepa_model.goal.tolerance))
                    self.assertLessEqual(float(errors[0, 0]), 1.)
                    self.assertGreaterEqual(float(errors[0, 1]), 2.)
                source = GoalPairBank(cases, config, pairs=8)
                rng = torch.get_rng_state().clone()
                images = source.sample()
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                for row in images:
                    self.assertTrue(any(torch.equal(row, torch.cat((c["image"][0], c["goal_image"][None])))
                                        for c in cases))
                with self.assertRaisesRegex(ValueError, "split"):
                    GoalPairBank([{**cases[0], "split": "validation"}], config)
                empty = {**cases[0], "states": cases[0]["states"][:, :1], "image": cases[0]["image"][:, :1]}
                with self.assertRaisesRegex(ValueError, "No eligible"):
                    GoalPairBank([empty], config)

    def test_gradients_reach_only_representation_without_changing_buffers_rng_or_modes(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                config, model = configured(family)
                model.train()
                images = GoalPairBank([real_pair(config)], config, pairs=2).sample()
                before = {k: v.clone() for k, v in model.named_buffers()}
                modes = [m.training for m in model.modules()]
                rng = torch.get_rng_state().clone()
                # Ensure an active hinge regardless of random initial ordering.
                loss, _ = goal_ranking_loss(model, images, 100.)
                loss.backward()
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.encoder.parameters()
                                       if p.grad is not None), 0.)
                if family == "leworldmodel":
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.projector.parameters()
                                           if p.grad is not None), 0.)
                for component in (model.predictor, model.pred_projector, model.action_encoder, model.state_head):
                    self.assertTrue(all(p.grad is None for p in component.parameters()))
                self.assertEqual(modes, [m.training for m in model.modules()])
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                for key, value in model.named_buffers():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_all_models_tasks_native_updates_and_disabled_path(self):
        for family in FAMILIES:
            for task in TASKS:
                with self.subTest(family=family, task=task):
                    config, model = configured(family, task)
                    pairs = GoalPairBank([real_pair(config)], config, pairs=2)
                    batch, _, _ = synthetic_batch(config, model, batch_size=4)
                    model._goal_ranking_source = pairs
                    model.train()
                    metrics = model.update(batch)
                    self.assertTrue(all(math.isfinite(float(x)) for x in metrics.values()))
                    self.assertEqual(metrics["goal_ranking/pairs"], 2.)
                    self.assertAlmostEqual(metrics["loss"], metrics["native_loss"] +
                                           metrics["goal_ranking/weighted_loss"], places=5)
                    self.assertEqual(pairs.draws, 2)
                    # Disabled source must not be sampled, and explicit goal images
                    # must have zero effect on native updates, buffers, and RNG.
                    model.goal_ranking_weight = 0.
                    other = copy.deepcopy(model)
                    torch.manual_seed(8)
                    left = model.update(batch)
                    rng = torch.get_rng_state().clone()
                    torch.manual_seed(8)
                    right = other.update(batch, goal_batch=pairs.sample(seed=9))
                    self.assertEqual(left["loss"], right["loss"])
                    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)
                    self.assertEqual(pairs.draws, 2)

    def test_validation_measurement_is_repeatable_and_cannot_be_training_source(self):
        config, model = configured("leworldmodel")
        source = GoalPairBank([real_pair(config, split="validation")], config, split="validation", pairs=2)
        original = source.state_dict()
        a = evaluate_goal_pairs(model, source, pairs=8)
        b = evaluate_goal_pairs(model, source, pairs=8)
        self.assertEqual(a, b)
        self.assertEqual(source.draws, 0)
        torch.testing.assert_close(source.generator.get_state(), original["generator_state"], rtol=0, atol=0)
        model._goal_ranking_source = source
        batch, _, _ = synthetic_batch(config, model, batch_size=4)
        with self.assertRaisesRegex(ValueError, "Only TRAIN"):
            model.update(batch)


class GoalRankingExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def longer_bank(*args, **kwargs):
            return collect_bank(*args, **{**kwargs, "horizon": 25})
        with patch("scripts.check_goal_maintenance.collect_bank", side_effect=longer_bank):
            fixtures.ForecastOnlineTests.setUpClass.__func__(cls)
        for row in cls.source_report["runs"]:
            row["coverage_fraction"] = .5
        (cls.source / "report.json").write_text(json.dumps(cls.source_report))
        cls.hashes = {str(p.relative_to(cls.source)): file_hash(p) for p in cls.source.rglob("*") if p.is_file()}

    @classmethod
    def tearDownClass(cls):
        fixtures.ForecastOnlineTests.tearDownClass.__func__(cls)

    def test_real_offline_online_two_model_run_records_supervision_and_validation(self):
        command = fixtures.ForecastOnlineTests.command(
            self, "ranked", "--online-steps", "64", "--retain-offline", "--training-horizon", "5",
            "--offline-updates", "4", "--goal-ranking-weight", ".1", "--goal-ranking-pairs", "4")
        log = io.StringIO()
        with redirect_stdout(log):
            status = main(command)
        self.assertEqual(status, 0, log.getvalue())
        report = json.loads((self.root / "ranked/report.json").read_text())
        self.assertEqual(len(report["runs"]), 2)
        self.assertTrue(report["training_adaptation"]["paper_objective_changed"])
        for row in report["runs"]:
            self.assertEqual((row["offline_updates"], row["updates"]), (4, 7))
            self.assertEqual(row["budget"]["goal_ranking"]["pairs_per_update"], 4)
            for snapshot in row["snapshots"]:
                ranking = snapshot["evaluation"]["result"]["goal_ranking"]
                self.assertEqual(ranking["pairs"], 512)
                self.assertEqual(ranking["source"]["split"], "validation")
                self.assertTrue(0 <= ranking["goal_ranking/pair_accuracy"] <= 1)
            self.assertEqual(len({s["evaluation"]["result"]["goal_ranking"]["images_sha256"]
                                  for s in row["snapshots"]}), 1)
            folder = self.root / "ranked" / row["task"] / row["model"]
            offline = torch.load(folder / "offline_latest.pt", weights_only=False)
            final = torch.load(folder / "latest.pt", weights_only=False)
            self.assertEqual(offline["goal_ranking_sampler"]["draws"], 4 * 4)
            self.assertEqual(final["goal_ranking_sampler"]["draws"], 11 * 4)
            native = [json.loads(line) for line in (folder / "online_metrics.jsonl").read_text().splitlines()]
            active = [entry["metrics"] for entry in native if entry["metrics"]]
            self.assertEqual(active[-1]["native/offline_sequences"], 0)
            self.assertTrue(all(m["goal_ranking/pairs"] == 4 for m in active))
        self.assertIn("Run | ranked | status=COMPLETE", log.getvalue())
        self.assertEqual(self.hashes, {str(p.relative_to(self.source)): file_hash(p)
                                      for p in self.source.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
