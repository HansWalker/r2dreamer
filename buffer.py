import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler

from constants import replay_cache_keys


class Buffer:
    def __init__(self, config):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.warmup_length = int(getattr(config, "warmup_length", 0))
        self.sample_length = self.warmup_length + self.batch_length + 1
        self.num_eps = 0
        self._buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=config.max_size, device=self.storage_device, ndim=2),
            sampler=SliceSampler(
                num_slices=self.batch_size, end_key=None, traj_key="episode", truncated_key=None, strict_length=True
            ),
            prefetch=0,
            batch_size=self.batch_size * self.sample_length,
        )

    def add_transition(self, data):
        # This is batched data and lifted for storage.
        # (B, ...) -> (B, 1, ...)
        self._buffer.extend(data.unsqueeze(1))

    def sample(self):
        sample_td, info = self._buffer.sample(return_info=True)
        # The sampler returns a flattened batch of length B*(T+1).
        # (B*(T+1), ...) -> (B, T+1, ...)
        sample_td = sample_td.view(-1, self.sample_length)
        src_dev = sample_td.device
        if src_dev.type == "cpu" and self.device.type == "cuda":
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=True)
        # Row 0 seeds the recurrent state. Optional warmup rows rebuild context
        # with current model weights before losses are computed on the suffix.
        initial = [sample_td["stoch"][:, 0], sample_td["deter"][:, 0]]
        for key in replay_cache_keys(sample_td.keys()):
            initial.append(sample_td[key][:, 0])
        initial = tuple(initial)
        sequence = sample_td[:, 1:]

        # Dynamics receive the previous action for each observation.
        sequence.set_("action", sample_td["action"][:, :-1])
        if self.warmup_length:
            warmup_data = sequence[:, : self.warmup_length]
            data = sequence[:, self.warmup_length :]
        else:
            warmup_data = None
            data = sequence
        index = [ind.view(-1, self.sample_length)[:, 1 + self.warmup_length :] for ind in info["index"]]
        return warmup_data, data, index, initial

    def update(self, index, stoch, deter, *cache, cache_keys=None):
        # Flatten the data
        index = [ind.reshape(-1) for ind in index]
        # (B, T, S, K) -> (B*T, S, K)
        stoch = stoch.reshape(-1, *stoch.shape[2:]).float()
        # (B, T, D) -> (B*T, D)
        deter = deter.reshape(-1, *deter.shape[2:]).float()
        values = {"stoch": stoch, "deter": deter}
        for key, value in zip(cache_keys or (), cache):
            values[key] = value.reshape(-1, *value.shape[2:]).float()
        # In storage, the length is the first dimension, and the batch (number of environments) is the second dimension.
        n = index[0].shape[0]
        self._buffer[index[1], index[0]] = TensorDict(values, batch_size=(n,))

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()
