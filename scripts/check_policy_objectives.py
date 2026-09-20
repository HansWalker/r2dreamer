"""CPU regression checks for saturated actor samples, return targets, and checkpoint reuse.

Run with: python -m scripts.check_policy_objectives
"""

import copy
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from models.shared.distributions import TanhNormal as DreamerNormal
from models.storm.objectives import TanhNormal as StormNormal
from models.storm.objectives import lambda_return
from scripts.smoke_models import synthetic_batch
from training import load_model_family
from training.protocol import checkpoint_compatibility, validate_checkpoint


def config_for(name):
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        config = compose(config_name=name, overrides=["device=cpu", "scenario=ball_in_cup"])
    OmegaConf.resolve(config)
    if config.model_family == "dreamer":
        config.model.compile = False
        config.model.imag_batch_size = 4
        config.model.imag_horizon = 3
    elif config.model_family == "storm":
        config.storm_train.imagine_horizon = 3
    return config


class PolicyObjectivesTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_saturated_sample_scores_and_gradients(self):
        for distribution in (DreamerNormal, StormNormal):
            with self.subTest(distribution=distribution.__module__):
                mean = torch.tensor([[20.0, -20.0]], requires_grad=True)
                log_std = torch.full_like(mean, -5.0, requires_grad=True)
                raw = (mean.detach() + log_std.detach().exp() * torch.tensor([[0.5, -0.5]])).detach()
                action = raw.tanh()
                self.assertTrue((action.abs() == 1).all())
                dist = distribution(mean, log_std.exp())
                actual = (
                    dist.log_prob_from_raw(raw) if distribution is DreamerNormal
                    else dist.log_prob(action, raw=raw)
                )
                correction = 2 * (math.log(2) - raw - torch.nn.functional.softplus(-2 * raw))
                expected = (torch.distributions.Normal(mean, log_std.exp()).log_prob(raw) - correction).sum(-1)
                torch.testing.assert_close(actual, expected)
                gradients = torch.autograd.grad(actual.sum(), (mean, log_std))
                torch.testing.assert_close(gradients[0], (raw - mean.detach()) / log_std.detach().exp().square())
                torch.testing.assert_close(gradients[1], ((raw - mean.detach()) / log_std.detach().exp()).square() - 1)
                self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
                self.assertLess(dist.log_prob(action).item(), -1e6)

    def test_sampling_preserves_actions_and_rng(self):
        mean, std = torch.tensor([[20.0, -20.0, 0.0]]), torch.ones(1, 3)
        for distribution in (DreamerNormal, StormNormal):
            with self.subTest(distribution=distribution.__module__):
                dist = distribution(mean, std)
                torch.manual_seed(19)
                action, raw = dist.rsample_with_raw() if distribution is DreamerNormal else dist.sample_with_raw()
                torch.manual_seed(19)
                previous = dist.rsample() if distribution is DreamerNormal else dist.sample()[0]
                torch.testing.assert_close(action, previous, rtol=0, atol=0)
                torch.testing.assert_close(action, raw.tanh(), rtol=0, atol=0)

    def test_unsaturated_scores_match_and_expert_boundaries_remain_finite(self):
        raw = torch.tensor([[0.2, -0.7]])
        for distribution in (DreamerNormal, StormNormal):
            with self.subTest(distribution=distribution.__module__):
                dist = distribution(torch.zeros_like(raw), torch.ones_like(raw))
                actual = dist.log_prob_from_raw(raw) if distribution is DreamerNormal else dist.log_prob(raw.tanh(), raw)
                torch.testing.assert_close(actual, dist.log_prob(raw.tanh()))
                endpoints = torch.tensor([[1.0, -1.0]])
                self.assertTrue(torch.isfinite(dist.log_prob(endpoints)).all())
                torch.testing.assert_close(dist.log_prob(endpoints), dist.log_prob(endpoints.clamp(-1 + 1e-6, 1 - 1e-6)))

    def test_one_transition_always_bootstraps_next_state(self):
        for lambd in (0.0, 0.95, 1.0):
            for trailing_dimension in (True, False):
                with self.subTest(lambd=lambd, trailing_dimension=trailing_dimension):
                    inputs = (torch.tensor([[[1.0]]]), torch.tensor([[[10.0], [20.0]]]), torch.zeros(1, 1, 1))
                    if not trailing_dimension:
                        inputs = tuple(x.squeeze(-1) for x in inputs)
                    actual = lambda_return(*inputs, 0.99, lambd)
                    torch.testing.assert_close(actual, torch.tensor([[[20.8]]]))

    def test_multistep_returns_and_terminal_mask(self):
        reward = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
        value = torch.tensor([[[10.0], [20.0], [30.0], [40.0]], [[7.0], [8.0], [9.0], [10.0]]])
        terminal = torch.tensor([[[0.0], [1.0], [0.0]], [[0.0], [0.0], [0.0]]])
        for lambd in (0.0, 0.95, 1.0):
            for trailing_dimension in (True, False):
                with self.subTest(lambd=lambd, trailing_dimension=trailing_dimension):
                    expected = torch.empty_like(reward)
                    for batch in range(2):
                        future = value[batch, -1, 0].item()
                        for index in reversed(range(3)):
                            future = reward[batch, index, 0].item() + 0.9 * (1 - terminal[batch, index, 0].item()) * (
                                (1 - lambd) * value[batch, index + 1, 0].item() + lambd * future
                            )
                            expected[batch, index, 0] = future
                    inputs = (reward, value, terminal) if trailing_dimension else tuple(x.squeeze(-1) for x in (reward, value, terminal))
                    torch.testing.assert_close(lambda_return(*inputs, 0.9, lambd), expected)

    def test_dreamer_expert_and_saturated_online_update(self):
        config = config_for("offline_dmc_expert_gru_vision")
        family = load_model_family("dreamer")
        model = family.build_model(config)
        batch, _, _ = synthetic_batch(config, model)
        with patch.object(DreamerNormal, "log_prob_from_raw", side_effect=AssertionError("Expert scoring changed")):
            family.expert_update(model, batch.clone())
        with torch.no_grad():
            model.actor.output.weight.zero_()
            model.actor.output.bias[:model.act_dim].fill_(20)
        contexts = [(batch[index, :0], [0]) for index in range(batch.shape[0])]
        replay = SimpleNamespace(sample=lambda: (contexts, batch.clone()))
        with patch.object(DreamerNormal, "log_prob", side_effect=AssertionError("Online actor inverted tanh")):
            metrics = model.update(replay)
        self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values()))
        self.assertEqual(model.state_head.updates.item(), 2)

    def test_storm_expert_and_saturated_online_update(self):
        from training.storm import OnlineSession

        config = config_for("storm_dmc_transformer_vision")
        family = load_model_family("storm")
        model = family.build_model(config)
        batch, _, _ = synthetic_batch(config, model)
        family.expert_update(model, batch)
        with torch.no_grad():
            model.actor_critic.actor[-1].weight.zero_()
            model.actor_critic.actor[-1].bias[:model.actor_critic.action_dim].fill_(20)
            model.actor_critic.actor[-1].bias[model.actor_critic.action_dim:].fill_(-5)
        original = StormNormal.log_prob

        def score(dist, action, raw=None):
            self.assertIsNotNone(raw)
            self.assertFalse(raw.requires_grad)
            self.assertTrue((action == 1).all())
            torch.testing.assert_close(action, raw.tanh(), rtol=0, atol=0)
            return original(dist, action, raw)

        session = OnlineSession(config, model, None)
        session.replay = SimpleNamespace(sample=lambda **kwargs: batch)
        with patch.object(StormNormal, "log_prob", score):
            metrics = session.update(1)
        self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values()))
        self.assertEqual(model.state_head.updates.item(), 2)
        with self.assertRaisesRegex(ValueError, "pre-squash"):
            model.actor_critic.update(None, None, None, None)

    def test_checkpoint_reuse_is_phase_and_family_specific(self):
        for name in ("offline_dmc_expert_gru_vision", "storm_dmc_transformer_vision", "tdmpc2_dmc_vision",
                     "leworldmodel_dmc_vision", "temporal_straightening_dmc_vision"):
            config = config_for(name)
            family = str(config.model_family)
            compatibility = checkpoint_compatibility(config)
            self.assertEqual(compatibility["recipe_version"], 9)
            identity = {
                "protocol": "test", "model_family": family, "model_variant": "test",
                "scenario": "test", "task": "test", "seed": 0,
            }
            checkpoint = {
                "experiment_protocol": "test", "checkpoint_id": "test", "run_identity": identity,
                "compatibility": {**compatibility, "recipe_version": 2},
                "training_config": OmegaConf.to_container(config, resolve=True),
            }
            with patch("training.protocol.run_identity", return_value=identity):
                for phase in ("expert", "online"):
                    checkpoint["phase"] = phase
                    with self.subTest(family=family, phase=phase):
                        with self.assertRaisesRegex(ValueError, "recipe_version"):
                            validate_checkpoint(checkpoint, config)
                        validate_checkpoint(checkpoint, config, training=False)
                        corrected = {**checkpoint, "compatibility": compatibility}
                        validate_checkpoint(corrected, config)
                checkpoint["phase"] = "expert"
                if family in {"dreamer", "storm"}:
                    recipe3 = {**checkpoint, "compatibility": {**compatibility, "recipe_version": 3}}
                    with self.assertRaisesRegex(ValueError, "recipe_version"):
                        validate_checkpoint(recipe3, config)
                    recipe3["phase"] = "online"
                    with self.assertRaisesRegex(ValueError, "recipe_version"):
                        validate_checkpoint(recipe3, config)
                bad = copy.deepcopy(checkpoint)
                bad["compatibility"]["training_sha256"] = "different"
                with self.assertRaisesRegex(ValueError, "training_sha256"):
                    validate_checkpoint(bad, config)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
