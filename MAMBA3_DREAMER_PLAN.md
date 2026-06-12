# Mamba3 Dreamer Change List

## Decided

- Replace the current rolling token-window Mamba sketch with the official Mamba3 recurrent/cache path.
- Use one Mamba3 layer for the first implementation.
- Compute `deter` from the Mamba3 transition output.
- Use the same transition inputs as the GRU baseline for comparison quality:
  - previous `deter`
  - previous stochastic latent state
  - previous action
- Use a single projection layer to convert the combined transition inputs into the Mamba3 input token.
- Carry Mamba3 context/cache alongside `stoch` and `deter`.
- Store each timestep's Mamba3 context/cache as flat replay/model-state fields:

  ```text
  mamba_angle_state
  mamba_ssm_state
  mamba_k_state
  mamba_v_state
  ```

- Use the default dtype produced by the Mamba3/cache allocation path for cache tensors.
- Do not use the raw Mamba3 cache directly as the Dreamer feature vector.
- Use official Mamba3 step mode for training, acting, and imagination.
- Use the same small experiment model dimensions for the GRU and Mamba3 comparison:

  ```yaml
  deter: 512
  hidden: 128
  discrete: 16
  depth: 16
  units: 128
  ```

- Use `warmup: 0` for Mamba3 because the cache is stored in replay.
- Use SISO Mamba3 for the first run:

  ```yaml
  is_mimo: false
  mimo_rank: 1
  ```

- Keep Mamba3 output-projection normalization disabled for the first run:

  ```yaml
  is_outproj_norm: false
  ```

- Keep the Dreamer feature interface:

  ```text
  feat_t = concat(flatten(stoch_t), deter_t)
  ```

- Keep the existing RSSM sequence structure:
  - `obs_step(...)` updates posterior state from one real observation.
  - `img_step(...)` updates prior state without an observation.
  - `observe(...)` loops over real sequence timesteps.
  - `imagine_with_action(...)` loops over imagined action timesteps.

## Implementation Tasks

### `rssm.py`

- Replace fake rolling-window context allocation with official Mamba3 cache allocation.
- Replace fake rolling-window context reset with cache-tensor reset.
- Replace fake rolling-window Mamba calls with official Mamba3 cache/step calls.
- Add cache tensor helpers for:
  - detach
  - move to device/dtype
  - stack or unstack over time
- Return the official cache tensors anywhere the current sketch returns rolling-window context.
- Run training sequences through Mamba3 one timestep at a time with the cache.

### `dreamer.py`

- Include Mamba3 context in the agent/model state when `rssm.core=mamba3`.
- Preserve Mamba3 context across online acting steps.
- Start imagination from the posterior Mamba3 context for the matching timestep.
- Pass replay-provided Mamba3 context into training.
- Update replay with the new Mamba3 context after training.
- Check frozen RSSM paths for context passing.

### `buffer.py`

- Store one replay field per Mamba3 cache tensor:
  - `mamba_angle_state`
  - `mamba_ssm_state`
  - `mamba_k_state`
  - `mamba_v_state`
- Sample initial Mamba3 context with initial `stoch` and `deter`.
- Update sampled replay rows with new Mamba3 context after training.

### `trainer.py`

- Attach current Mamba3 context/cache to collected transitions.
- Check video/open-loop logging paths after the RSSM context API is finalized.

### Configs

- Add official-cache Mamba3 fields under `rssm.mamba3`.
- Remove rolling-window-only fields:
  - `context_len`
  - `mlp_hidden_mult`
- Update `size_small_mamba3.yaml` to match the real-cache path.
- Keep the GRU and Mamba3 small experiment configs identical except for RSSM core-specific fields.

## Smoke Checks

- One Mamba3 RSSM step returns finite tensors.
- Official Mamba3 step mode runs forward and backward on the target hardware.
- Mamba3 `observe(...)` returns cache tensors for posterior timesteps.
- Replay can store, sample, and update Mamba3 context.
- Actor imagination uses the posterior context from the matching start timestep.
