"""Check gradual replay handover, sequence alignment and growing episode boundaries."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from buffer import SequenceBuffer
from scripts.offline_online_replay import OfflineOnlineSession, ScheduledReplay


def fixture():
    # Episode 1 is held out and must never be read, despite being much longer.
    class Expert:
        episodes = np.array([0, 2])
        lengths = np.array([5, 100, 7])
        actions = np.stack([np.arange(100) + 20 * i for i in range(3)])[..., None]

        def _read_observations(self, episode, start, length):
            if episode == 1:
                raise AssertionError("Read held-out expert episode")
            values = self.actions[episode, start:start + length]
            return {"image": values.astype(np.uint8), "physical_state": values.astype(np.float32)}

    prefix = torch.arange(60, 63, dtype=torch.uint8)[:, None]
    future = torch.tensor([[[63], [64], [65]], [[73], [74], [75]]], dtype=torch.uint8)
    case = {"id": "train", "split": "train", "prefix": prefix, "image": future,
            "prefix_state": prefix.float(), "states": future.float(),
            "past_action": prefix[:2].float(),
            "action": torch.tensor([[[62.], [63.], [64.]], [[62.], [73.], [74.]]])}
    bank = {"splits": {"train": [case], "validation": [{**case, "id": "heldout", "split": "validation"}]}}
    replay = SequenceBuffer(OmegaConf.create({"max_size": 100, "storage_device": "cpu", "device": "cpu",
                                             "batch_size": 4, "sequence_length": 4, "episodes_per_batch": 1, "seed": 9}))
    replay.start(1)
    for value in range(100, 111):
        replay.append({"image": torch.tensor([[value]], dtype=torch.uint8), "physical_state": torch.tensor([[float(value)]])},
                      torch.tensor([[float(value)]]), torch.zeros(1, 1), torch.zeros(1), torch.tensor([value == 104]))
    return Expert(), bank, replay


class ScheduledReplayTests(unittest.TestCase):
    def test_every_valid_window_is_reachable_once_and_actions_align(self):
        expert, bank, online = fixture()
        sampler = ScheduledReplay(expert, bank, 19, 4, 4, updates=3, start_fraction=14/19)
        self.assertEqual(sampler.inventory(online), {"expert": 8, "branch": 6, "online": 5})
        with patch("scripts.offline_online_replay.torch.randint", side_effect=[torch.arange(14), torch.arange(5)]), \
                patch("scripts.offline_online_replay.torch.randperm", return_value=torch.arange(19)):
            (obs, action), metrics = sampler.sample(online, 0)
        torch.testing.assert_close(action, obs["image"][:, :-1].float())
        torch.testing.assert_close(obs["physical_state"], obs["image"].float())
        self.assertEqual(obs["image"].shape, (19, 4, 1))
        self.assertEqual(len(set(sampler.last_draws)), 19)
        # Last expert target is the stored terminal observation; online episodes
        # include only recorded current observations and cannot cross the reset.
        self.assertEqual(obs["image"][7, :, 0].tolist(), [44, 45, 46, 47])
        self.assertEqual(obs["image"][-1, :, 0].tolist(), [107, 108, 109, 110])
        self.assertEqual(sampler.total_samples, {"expert": 8, "branch": 6, "online": 5})
        self.assertEqual(metrics["native/offline_sequences"], 14)

    def test_windows_are_uniform_within_pools_and_online_data_accumulates(self):
        expert, bank, online = fixture()
        sampler = ScheduledReplay(expert, bank, 6000, 4, 22, updates=3)
        _, metrics = sampler.sample(online, 0)
        self.assertEqual(metrics["native/offline_sequences"], 3000)
        self.assertEqual(metrics["native/online_sequences"], 3000)
        for source, count in (("expert", 8), ("branch", 6)):
            probability = .5 * count / 14
            self.assertAlmostEqual(metrics[f"replay/{source}_fraction"], probability)
            self.assertLess(abs(metrics[f"native/{source}_sequences"] - 6000 * probability),
                            6 * (6000 * probability * (1 - probability)) ** .5)
        online.append({"image": torch.tensor([[111]], dtype=torch.uint8), "physical_state": torch.tensor([[111.]])},
                      torch.tensor([[111.]]), torch.zeros(1, 1), torch.zeros(1), torch.tensor([False]))
        self.assertEqual(sampler.inventory(online), {"expert": 8, "branch": 6, "online": 6})
        sampler.batch_size = 4
        with patch("scripts.offline_online_replay.torch.randint",
                   side_effect=[torch.empty(0, dtype=torch.long), torch.full((4,), 5)]), \
                patch.object(expert, "_read_observations", side_effect=AssertionError("Read original data at endpoint")):
            (obs, _), metrics = sampler.sample(online, 2)
        self.assertEqual(obs["image"][0, :, 0].tolist(), [108, 109, 110, 111])
        self.assertEqual(metrics["replay/online_fraction"], 1.)
        self.assertEqual(metrics["replay/online_pool_fraction"], .3)
        self.assertEqual(metrics["native/offline_sequences"], 0)
        self.assertEqual(sum(sampler.total_samples.values()), 6004)

    def test_empty_online_pool_and_sampler_rng_are_independent(self):
        expert, bank, online = fixture()
        empty = SequenceBuffer(SimpleNamespace(max_size=100, storage_device="cpu", device="cpu", batch_size=4,
                                              sequence_length=4, episodes_per_batch=1, seed=9))
        empty.start(1)
        left = ScheduledReplay(expert, bank, 64, 4, 42, updates=3)
        right = ScheduledReplay(expert, bank, 64, 4, 42, updates=3)
        with self.assertRaisesRegex(RuntimeError, "Collect usable online sequences"):
            left.sample(empty, 0)
        self.assertEqual(sum(left.total_samples.values()), 0)
        rng = torch.get_rng_state().clone()
        (_, a), metrics = left.sample(online, 0)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        torch.rand(100)
        (_, b), _ = right.sample(online, 0)
        torch.testing.assert_close(a, b)
        self.assertEqual(left.last_draws, right.last_draws)
        self.assertEqual(metrics["replay/online_fraction"], .5)
        self.assertEqual(metrics["native/online_sequences"], 32)
        bank["splits"]["train"][0]["split"] = "validation"
        with self.assertRaises(ValueError):
            ScheduledReplay(expert, bank, 4, 4, 0, updates=3)

    def test_handover_has_exact_start_midpoint_endpoint_and_never_restarts(self):
        expert, bank, online = fixture()
        sampler = ScheduledReplay(expert, bank, 128, 4, 22, updates=1839)
        quotas = [sampler.quota(i)[1] for i in range(1839)]
        self.assertEqual((quotas[0], quotas[919], quotas[-1]), (64, 32, 0))
        self.assertEqual(quotas, sorted(quotas, reverse=True))
        self.assertEqual(sampler.quota(1839), (0., 0))
        self.assertEqual(sampler.quota(10000), (0., 0))
        for index, expected in ((0, 64), (919, 32), (1838, 0)):
            _, metrics = sampler.sample(online, index)
            self.assertEqual(metrics["native/offline_sequences"], expected)
            self.assertEqual(metrics["native/online_sequences"], 128 - expected)
            self.assertAlmostEqual(sum(metrics[f"replay/{source}_fraction"] for source in sampler.sources), 1.)
        with self.assertRaises(ValueError):
            sampler.quota(-1)
        with self.assertRaises(ValueError):
            ScheduledReplay(expert, bank, 128, 4, 22, updates=1)

    def test_schedule_advances_inside_update_bursts_and_readout_budget_stays_full(self):
        expert, bank, online = fixture()
        sampler = ScheduledReplay(expert, bank, 8, 4, 22, updates=5)
        config = SimpleNamespace(replay=SimpleNamespace(max_size=100, storage_device="cpu", device="cpu",
                                 batch_size=4, sequence_length=4, episodes_per_batch=1, seed=9),
                                 env=SimpleNamespace(action_repeat=2))
        quotas = []
        def update(batch, *, readout_batch):
            self.assertEqual(batch[0]["image"].shape[0], 8)
            self.assertEqual(readout_batch[0]["image"].shape[0], 4)
            quotas.append(sum(source != "online" for source, _, _ in sampler.last_draws))
            return {}
        session = OfflineOnlineSession(config, SimpleNamespace(sequence_length=4, update=update), None, sampler)
        session.replay = online
        session.update(2)
        metrics = session.update(3)
        self.assertEqual(quotas, [4, 3, 2, 1, 0])
        self.assertEqual(session.updates, 5)
        self.assertEqual(metrics["replay/sampling_update"], 5)
        self.assertEqual(metrics["replay/offline_target_fraction"], 0.)
        self.assertEqual(sum(sampler.total_samples.values()), 40)


if __name__ == "__main__":
    unittest.main()
