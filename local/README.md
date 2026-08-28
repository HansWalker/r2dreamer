# Local Files

Use this directory for archived code, one-off scripts, notes, and other files
that are not part of the active training path. Everything here except this
README is ignored by Git and must not be imported by repository code.

The current local archive is organized as follows:

- `cache/`: generated Python and pytest caches retained instead of deleted.
- `ablations/state_observations/`: the retired vector-observation configs and implementation snapshots.
- `legacy/`: superseded notebook helpers, CLI entrypoints, configs, Docker setup, replaced implementations,
  and archived environments.
- `notebooks/`: the former Colab and recurrent-core notebooks.
- `reference/`: complete upstream STORM and TD-MPC2 checkouts.
- `results/`: the previous DMC backup, evaluation summary, and presentation figures.
- `tests/`: historical test snapshots for the retired module and config layout; they are not part of
  the active verification path.
