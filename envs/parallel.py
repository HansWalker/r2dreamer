import atexit
import contextlib
import sys
import traceback
from functools import partial

import numpy as np
import torch
from tensordict import TensorDict


def _stack(rows):
    tensors = {key: torch.as_tensor(np.stack([row[key] for row in rows])) for key in rows[0]}
    batch = TensorDict(tensors, batch_size=(len(rows),), device="cpu")
    for key in batch.keys():
        if batch[key].ndim == 1:
            batch[key] = batch[key].unsqueeze(-1)
    return batch


class ParallelEnv:
    def __init__(self, constructor, env_num, pin_memory=False):
        self.workers = [ProcessEnv(constructor(index)) for index in range(env_num)]
        self.pin_memory = bool(pin_memory)

    @property
    def env_num(self):
        return len(self.workers)

    def close(self):
        for worker in self.workers:
            worker.close()

    def reset(self):
        promises = [worker.reset() for worker in self.workers]
        obs = _stack([promise() for promise in promises])
        return obs.pin_memory() if self.pin_memory else obs

    def reset_done(self, obs, done):
        indices = done.nonzero(as_tuple=False).flatten()
        if not len(indices):
            return obs
        promises = [self.workers[index].reset() for index in indices.cpu().tolist()]
        obs = obs.clone()
        reset_obs = _stack([promise() for promise in promises])
        obs[indices] = reset_obs.pin_memory() if self.pin_memory else reset_obs
        return obs

    def step(self, action, reset_mask=None):
        """Step active environments and optionally reset selected workers."""
        values = action.detach().cpu().numpy()
        reset = [False] * self.env_num if reset_mask is None else reset_mask.detach().cpu().reshape(-1).tolist()
        promises = [
            worker.reset() if flag else worker.step(value)
            for worker, value, flag in zip(self.workers, values, reset, strict=True)
        ]

        rows, rewards, dones = [], [], []
        for promise, was_reset in zip(promises, reset, strict=True):
            if was_reset:
                rows.append(promise())
                rewards.append(0.0)
                dones.append(False)
            else:
                obs, reward, finished, _ = promise()
                rows.append(obs)
                rewards.append(reward)
                dones.append(finished)

        obs = _stack(rows)
        reward = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1, 1)
        done = torch.as_tensor(dones, dtype=torch.bool)
        if self.pin_memory:
            obs, reward, done = obs.pin_memory(), reward.pin_memory(), done.pin_memory()
        return obs, reward, done


class ProcessEnv:
    def __init__(self, constructor):
        import multiprocessing

        import cloudpickle

        context = multiprocessing.get_context("spawn")
        self._pipe, pipe = context.Pipe()
        constructor = cloudpickle.dumps(constructor)
        self._process = context.Process(target=self._loop, args=(pipe, constructor))
        self._process.start()
        self._next_id = 0
        self._results = {}
        atexit.register(self.close)

    def reset(self):
        return self._submit("reset")

    def step(self, action):
        return self._submit("step", action)

    def close(self):
        with contextlib.suppress(AttributeError, OSError):
            self._pipe.send((None, None, None))
            self._pipe.close()
        with contextlib.suppress(AttributeError, AssertionError):
            self._process.join(0.1)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(0.1)

    def _submit(self, method, *args):
        call_id = self._next_id
        self._next_id += 1
        self._pipe.send((call_id, method, args))
        return partial(self._receive, call_id)

    def _receive(self, call_id):
        while call_id not in self._results:
            try:
                received_id, result, error = self._pipe.recv()
            except (OSError, EOFError) as error:
                raise RuntimeError("Lost connection to worker.") from error
            if error:
                raise RuntimeError(error)
            self._results[received_id] = result
        return self._results.pop(call_id)

    @staticmethod
    def _loop(pipe, constructor):
        call_id = None
        env = None
        try:
            import cloudpickle

            env = cloudpickle.loads(constructor)()
            while True:
                call_id, method, args = pipe.recv()
                if method is None:
                    return
                pipe.send((call_id, getattr(env, method)(*args), None))
        except (EOFError, KeyboardInterrupt):
            pass
        except Exception:
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print(f"Error inside process worker: {stacktrace}.", flush=True)
            pipe.send((call_id, None, stacktrace))
        finally:
            with contextlib.suppress(Exception):
                env.close()
            with contextlib.suppress(Exception):
                pipe.close()
