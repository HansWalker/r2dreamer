# DMC World Model Comparisons

This comparison framework is based on [R2-Dreamer][r2dreamer]. It trains capacity-matched Dreamer,
STORM, TD-MPC2, LeWorldModel, and Temporal Straightening models on DeepMind Control Suite tasks with
64x64 image observations and optional expert pretraining.

## Code Map

| Path | Responsibility |
|---|---|
| `main.py` | Resolve and validate the matrix once, collect datasets, then train and evaluate each run. |
| `train.py` | Hydra entrypoint for a single training run. |
| `training/trainer.py` | Model creation, resume, expert pretraining, and online scheduling. |
| `training/dreamer.py`, `training/storm.py`, `training/planning.py` | Family-specific updates, replay adapters, and environment policies. |
| `models/<family>/` | Architecture, objectives, and optimizers; Dreamer/STORM variants live in `cores/`. |
| `models/shared/`, `models/planning.py` | Common numerical layers, physical-state head, and latent planning. |
| `buffer.py`, `dmc_expert/replay.py` | Online raw-episode replay and offline HDF5 sampling. |
| `dmc_expert/collection.py`, `storage.py`, `tdmpc2.py` | Expert rollouts, dataset schema/resume, and external expert loading. |
| `scripts/evaluate_dmc.py`, `training/evaluation.py` | Checkpoint evaluation, episode metrics, and physical-state forecasts. |
| `training/protocol.py` | Experiment budgets, provenance, and checkpoint compatibility checks. |
| `envs/` | DMC observations/actions and environment subprocesses. |
| `configs/` | Shared budgets, family recipes, core selectors, and scenario definitions. |

Active code is image-only. Archived state-input experiments, reference checkouts, notebooks, tests,
and prior results stay under ignored `local/`; collection uses the external `TDMPC2_DIR` checkout.

## Setup

This repository is tested with Ubuntu 22.04 and Python 3.10; the pinned wheels also support Python
3.11. The setup script creates or updates the shared `environment/` virtual environment, installs
PyTorch 2.8 for CUDA 12.8, builds Mamba3, and checks the dependency graph, DMC rendering, TD-MPC2
(when configured), and production Mamba kernels. The NVIDIA driver, CUDA 12.8 toolkit, and system
libraries below must already be installed. The initial Mamba source build and checks can take several
minutes.

```bash
sudo apt-get install -y build-essential git libegl1 libglew2.2 libz3-dev python3.10-venv
PYTHON=python3.10 CUDA_HOME=/usr/local/cuda-12.8 bash scripts/setup_dmc.sh
source ../environment/bin/activate
```

Set `PYTHON`, `CUDA_HOME`, or `ENV_DIR` when those paths differ. The compatible Mamba stack is pinned
in [`requirements/mamba3-cu128.txt`](requirements/mamba3-cu128.txt), and its runtime check can be
repeated with `python -m scripts.check_dmc_setup`. Before a full experiment, run
`python -m scripts.smoke_models` to exercise every configured model's update, checkpoint, policy,
and latent-rollout path.

## Run

Expert datasets use the `dmc_expert_hdf5_dense_v1` layout: each dataset directory contains
`metadata.json`, `data.hdf5`, and `progress.json`. The primary experiments use 64x64 RGB observations
on Cartpole Balance Sparse, Reacher Easy, and Ball-in-Cup Catch. Set the local TD-MPC2 checkout
and image-dataset root before collecting or training:

```bash
export TDMPC2_DIR=/absolute/path/to/tdmpc2
export DMC_EXPERT_VISION_DATA_DIR=/absolute/path/to/data/dmc_expert_vision

python3 -m scripts.collect_dmc_expert_data --config-name dmc_expert_collection

python3 train.py --config-name offline_dmc_expert_gru_vision scenario=cartpole_balance_sparse
python3 train.py --config-name offline_dmc_expert_sliding_window_vision scenario=cartpole_balance_sparse
python3 train.py --config-name offline_dmc_expert_mamba3_vision scenario=cartpole_balance_sparse
python3 train.py --config-name offline_dmc_expert_s5_vision scenario=cartpole_balance_sparse
python3 train.py --config-name offline_dmc_expert_hyena_vision scenario=cartpole_balance_sparse
python3 train.py --config-name storm_dmc_transformer_vision scenario=cartpole_balance_sparse
python3 train.py --config-name storm_dmc_sliding_window_vision scenario=cartpole_balance_sparse
python3 train.py --config-name storm_dmc_mamba_vision scenario=cartpole_balance_sparse
python3 train.py --config-name storm_dmc_s5_vision scenario=cartpole_balance_sparse
python3 train.py --config-name storm_dmc_hyena_vision scenario=cartpole_balance_sparse
python3 train.py --config-name leworldmodel_dmc_vision scenario=cartpole_balance_sparse
python3 train.py --config-name temporal_straightening_dmc_vision scenario=cartpole_balance_sparse
python3 train.py --config-name tdmpc2_dmc_vision scenario=cartpole_balance_sparse
```

TD-MPC2 publishes compatible checkpoints for all three tasks; collection downloads them on demand.

Run the complete collection, training, and evaluation matrix from one config:

```bash
./scripts/run_full.sh --dry-run
./scripts/run_full.sh
```

Before committing to the full datasets and training budget, run the same lifecycle with one tiny
Cartpole Balance Sparse dataset and one expert update per model:

```bash
./scripts/run_smoke.sh --dry-run
./scripts/run_smoke.sh
```

For a **fresh, extremely small LeWorldModel/Temporal Straightening training check**, without
the expert dataset, pretrained weights, or checkpoint writes:

```bash
bash scripts/run_tiny_planner_check.sh
```

This defaults to CPU and Cartpole, with roughly 14K/22K parameters, 64x64 images, batches of
8 sequences, 128 offline updates and 128 online updates. Four real random-action episodes
(not expert data) supply offline training and the head's 50% retention source. Online replay
starts empty and receives only each model's own actions. Two disjoint simulator episodes
measure observed-state and open-loop physical errors before/after training and every 64 online
updates. Native losses, optimizer types and learning rates stay unchanged; sizes, schedules,
episode lengths and planner budgets are deliberately tiny. A tripwire forbids physical-head
calls during planning. `PASS` means execution checks passed, not that task learning or full-size
stability is established. Reports include individual-coordinate errors and short policy returns.
Use `--scenario reacher` or `--scenario ball_in_cup`, `--device cuda:0`, `--offline-updates`,
`--online-updates`, or `--output` as needed. Existing output directories are never overwritten.

For the **offline paper-recipe and action-coverage comparison**, targeting about one hour total:

```bash
bash scripts/run_paper_faithful_check.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision --minutes 60
```

This runs six arms sequentially: TS patchwise curvature at 0.1 and 0.01, TS upstream-style
aggregation at 0.1, that aggregation variant with additional intervention data, and LeWM
with/without intervention data. The coverage arms replace half the offline expert windows
with shared action branches. **There is no online training, online retention change, or online
schedule change.** Native objectives, optimizers, spatial goal scoring, and action bounds are
preserved. Production defaults remain patchwise TS until experiment results justify promotion.

The default uses full comparison-model widths and production planning budgets. TS aggregation
adds 2,033,024 active parameters (about 7.229M total versus 5.196M for patchwise TS); it is an
explicit capacity-unmatched restoration control. `--profile small` and `--profile tiny` are
optional size adaptations, not automatic fallbacks or full-size convergence tests.

Before fitting, the runner checks hash-pinned upstream components, collects a common simulator
branch bank with disjoint training/validation/test anchors, and profiles disposable models.
It chooses one fixed update count for all six arms, reserving time for measurements. Calibration
weights are discarded. `--updates N` fixes the common count instead; timing is still reported.
`--minutes` is an estimate, not a deadline: the run does not truncate slower arms and give them
less training. If the estimated budget cannot fit `--min-updates` (default 100), it stops with
an explicit insufficient-budget report before the comparison fits.

Measurements compare the exact native cost on real versus predicted future images, matched/
shuffled/zero actions, persistence, feature spread, source versus production search budgets,
and the actual optimizer-selected sequence. Default policy screening uses three controlled
starts for 100 decisions per arm. This is not the full 50-episode benchmark or a stability
claim; use `--policy-steps 497` for longer controlled trials (two decisions build the real
observation prefix), and expand independent seeds/ordinary-reset evaluation after screening.

Outputs include `report.json`, `summary.txt`, the reusable `branches.pt`, six native checkpoints,
and per-update/per-policy JSONL traces. `COMPLETE` means the experiment executed; lower loss
alone is not evidence that control is repaired. All outputs go to a new directory under `runs/`
unless `--output` is supplied. Existing datasets and checkpoints are read-only.

Pinned source downloads are small and verified against recorded SHA-256 hashes. Use
`--reference-cache PATH --no-reference-download` for an already populated offline cache.
`--skip-reference` is an explicit smoke-test bypass recorded as `NOT_RUN`, never as a parity pass.
These checks execute upstream component code on matched tensors; **they do not reproduce an
original paper task**. Original-task reference execution requires the separately installed
official runtime, original dataset, and released checkpoint.

The separate original-task launcher verifies the released LeWM PushT checkpoint and invokes
the pinned official evaluator. Preflight records missing prerequisites; add `--run` to execute
the original 50-case evaluation once they are available. Its runtime is separate from the
one-hour DMC comparison:

```bash
python -m scripts.run_upstream_reference --prepare-code \
  --python /path/to/official-env/bin/python \
  --checkpoint-dir /data/lewm-pusht --dataset-root /data/stable-wm \
  --output runs/upstream_pusht_preflight
```

The dataset root must contain `datasets/pusht_expert_train.h5`. The launcher downloads only
small pinned source files when `--prepare-code` is supplied; it records setup details in
`manifest.json` and requires the runtime, dataset, and checkpoint to be provisioned separately.
Use a fresh output directory for each preflight or evaluation.

Saved native checkpoints can also be evaluated without fitting:

```bash
bash scripts/run_paper_faithful_check.sh --device cuda:0 \
  --checkpoint runs/previous/temporal_straightening/offline_24000/native.pt \
  --checkpoint runs/previous/leworldmodel/offline_24000/native.pt
```

The loader supports latent-planner archives and diagnostic `training_config` snapshots, records
source hashes, and uses the current TS/LeWM native score while preserving the saved architecture.
It records any score override and never resumes optimizer/replay state. The new scorer retains
TS observed-prefix/time weighting; older terminal-only diagnostic CLIs keep their historical scope.
Checkpoint-only mode performs branch scoring; it does not run the policy or solver trials.

CPU contracts and simulator checks:

```bash
MUJOCO_GL=egl python -m scripts.check_ts_aggregation
MUJOCO_GL=egl python -m scripts.check_paper_faithful_support
MUJOCO_GL=egl python -m scripts.check_paper_faithful_check
PAPER_REFERENCE_CACHE=runs/reference_cache python -m scripts.check_paper_faithful_reference
```

For the **coverage follow-up with broader starts and intermediate evaluations**:

```bash
bash scripts/run_paper_faithful_followup.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision --minutes 60
```

This compares TS patch curvature 0.1 + coverage, TS aggregation 0.1 + coverage, and
LeWM + coverage, each at training seeds 1 and 2. All six fits use 50% expert and 50%
intervention windows, full model widths, the native loss/optimizer recipes, and one
matched update budget. There is no online training or online schedule change.

The common bank has 32 training, nine validation, and nine test anchors sampled over
broader continuous position/angle/velocity ranges, with disjoint seeds and states.
Each anchor has 12 action branches of 25 decisions. Forecasts and native goal costs
are measured at horizons **1, 5, 15, and 25**. **Policy planning stays at horizon 5**
with production solver budgets; the longer forecasts do not silently change control.

Validation forecasts are recorded before training, halfway through, and at the end.
The two trained validation checkpoints also run the same three starts for 25 decisions,
giving a paired control learning curve. Final testing uses separate held-out anchors
and six policy starts for 100 decisions. Zero-action and privileged state-feedback
controls use the identical starts and durations. Test results do not select training
budgets or checkpoints. Intermediate evaluation preserves model/optimizer state and RNG.

Collection and long-horizon evaluation are included in runtime calibration. The shared
update count can be smaller than the previous 1,693-update screen because this run spends
more time evaluating. `--minutes 90` provides a larger total budget; `--updates N` explicitly
fixes the update count and can exceed the time target. If even `--min-updates` (256) does
not fit the estimated runtime, the runner stops before fitting. Models are never silently
shrunk and individual fits are never cut short to meet a deadline.

Outputs under `runs/paper_faithful_followup_<timestamp>/` include `summary.txt`,
`report.json`, `branches.pt`, baseline-control traces, and per-seed/per-arm training logs,
intermediate/final checkpoints, and policy traces. Compare validation learning curves
at equal update counts; final-test return has a different duration and separate starts.
This is a controlled-start screening test, not the full ordinary-reset benchmark or an
original-paper task reproduction. Previous run scripts and defaults remain available.

Follow-up CPU/simulator checks:

```bash
MUJOCO_GL=egl python -m scripts.check_paper_faithful_followup_support
MUJOCO_GL=egl python -m scripts.check_paper_faithful_followup_eval
MUJOCO_GL=egl python -m scripts.check_paper_faithful_followup
```

For the **frozen-checkpoint planning comparison** after the larger coverage follow-up:

```bash
bash scripts/run_paper_faithful_planning.sh \
  --source-run runs/paper_faithful_followup_20260922_092521
```

The source directory must contain its `report.json`, `branches.pt`, and the final
TS-aggregation/LeWM `native.pt` checkpoints for seeds 1, 2, and 3. No expert dataset
is needed. Add `--dry-run` to validate the saved artifacts and print the matrix
without a GPU or simulator evaluation. `--source-run` may also point to a downloaded
copy under `local/reports/`.

The six conditions per seed are TS spatial cost at H5/H15, TS spatial plus
`0.1 * aggregate` cost at H5/H15, and LeWM's existing terminal cost at H5/H15.
The aggregate term uses the already-trained TS head. Both terms retain the current
TS MPC temporal weighting and coordinate reduction; their sum uses one coherent
goal. This implements the combined-cost formula reported in the
[TS paper's long-horizon experiment](https://arxiv.org/html/2603.12231v3), not a new
training loss or a claim of original-task reproduction. The default model cost
remains spatial-only (`aggregate_goal_weight: 0.0`).

All 18 evaluations use frozen final weights, the same 12 validation starts, 200
decisions per trial, and unchanged source solver budgets. H5/H15 candidate scoring
uses all 18 source validation anchors. Zero-action and privileged state-feedback
controls are recomputed for the longer trial. Training and test splits are checked
for artifact integrity but are not evaluated. Select settings on these validation
results, then confirm on fresh held-out starts. This is an evaluation workload,
with no training or fixed six-hour timer; per-condition runtimes are recorded.

Outputs under `runs/paper_faithful_planning_<timestamp>/` include `summary.txt`,
`report.json`, control traces, checkpoint/source hashes, resolved planner settings,
and per-seed/per-condition policy traces. Source artifacts are read-only. Checkpoint
and branch hashes, saved configurations, and simulator compatibility are validated
before evaluation. Partial failures are recorded; existing output directories are
never overwritten.

Planning comparison checks:

```bash
MUJOCO_GL=egl python -m scripts.check_paper_faithful_planning_cost
MUJOCO_GL=egl python -m scripts.check_paper_faithful_planning
```

For the **roughly five-hour Cartpole forecast/online experiment**, continuing the
two completed TS and LeWM checkpoints:

```bash
bash scripts/run_forecast_online.sh \
  --source-run runs/paper_faithful_duration_20260923_020814
```

To test the **small supervised goal-ranking penalty**, use:

```bash
bash scripts/run_goal_ranking.sh \
  --source-run runs/paper_faithful_duration_20260923_020814
```

This runs **only TS and LeWM on Cartpole**, with the previous multi-step
experiment's 2,000 additional offline updates, 1,839 online updates, H5 training,
H25 planners, and original-data taper. Allow roughly **five A100 hours total**;
there is no calibration or time cutoff. The historical comparison is
`multistep_offline_online_20260924_033419`, using the same original offline
checkpoints and seeds. This command does not rerun that control arm.

Each update adds 32 good/bad image pairs from the existing bank's **TRAIN**
anchors. Both images share the same episode goal. A good frame satisfies the
task's physical goal bounds; a bad frame is at least twice the allowed distance
away. The penalty is `0.1 * mean(max(0, 0.1 + d_good - d_bad))`, where distances
are the existing squared image-feature distances expressed per coordinate.
It updates the encoder/projector in their planning-time BN/dropout mode while
preserving their running statistics. The goal is freshly encoded and detached.
Predictor training, inference inputs, model parameters, and the planner are
otherwise unchanged. Weight zero preserves the native update path.

This is an explicit supervised extension of the papers' objectives, with extra
physical-label access, not an exact paper reproduction. It uses no reward/value
learner or additional neural head. The small TRAIN comparison source remains
available even after the main replay's original-data share reaches zero.
Existing branch banks already contain the correct goal image, including
episode-specific Reacher targets; native replay need not store another image.
The same geometry rule supports Cartpole, Reacher, and Ball-in-Cup, but this
launch uses Cartpole only. A bank without eligible training or validation pairs
fails preflight rather than silently omitting the penalty.

Reports under `runs/goal_ranking_offline_online_<UTC timestamp>/` include fixed
512-pair validation ordering/margin scores, pair-image hashes, existing forecast
and control diagnostics, training loss components, supervision inventory, and
sampler state in checkpoints. The pairs are drawn from held-out anchors and are
not 512 independent trials. Both models use identical sampled pairs. The run
name, status, and path print at exit. Lower ranking loss alone is not a repair:
inspect held-out ordering, forecast accuracy, return, and maintenance together.

Append `--dry-run` to validate saved artifacts and label availability without
training or expert-data access. If the expert data moved, append
`--dataset-root /path/to/dmc_expert_vision`. Keep the original duration source;
the later online reports are comparison results, not accepted checkpoint sources.

```bash
MUJOCO_GL=egl python -m scripts.check_goal_ranking
```

To test **a gradual handover from original data to accumulated online experience**, use:

```bash
bash scripts/run_forecast_online.sh \
  --source-run runs/paper_faithful_duration_20260923_020814 \
  --retain-offline
```

This keeps the same two models, Cartpole task, update counts, planners and
before/midpoint/after evaluations. Native batches start at **50% original / 50%
online** after the existing collection warmup. The original share decreases
linearly with update count, reaching **0% original / 100% online on this run's
final update**. The taper spans the requested 1,839 updates, even though this
experiment is a prefix of the longer learning-rate schedule. At 128 sequences
per batch, first/middle/last update allocations are 64/64, 32/96 and 0/128.
Counts round to the nearest whole sequence.

Windows are uniform within each pool. The original pool includes expert training
episodes and the original **training** action branches, weighted by their usable
window counts. This retains the previous implementation's within-original
weighting, not the offline fit's earlier fixed 50/50 expert/branch allocation.
Online experience accumulates throughout the run. Episode boundaries and held-out
splits remain protected. Logs distinguish pool sizes, scheduled/rounded source
shares, and cumulative draws. The separate diagnostic readout keeps its previous
label budget and 50% expert mix, including at the native all-online endpoint.
Native objectives, architectures and learning-rate schedules are unchanged.

Outputs use `runs/offline_online_decay_<timestamp>/`. The newly collected trajectories
are also saved as each model's `online_data.pt`, refreshed with checkpoints and
on handled early exit. Original data files remain read-only. All online data
from this experiment stays available; capacity is increased if needed. The same
rough five-hour allowance applies, with additional original-data reads; there
is no clock enforcement or calibration. The final terminal lines print the run
name and status. Use `--retain-offline --dry-run` to check source compatibility.

```bash
MUJOCO_GL=egl python -m scripts.check_offline_online_replay
MUJOCO_GL=egl python -m scripts.check_forecast_online
```

Use the same A100 environment and original expert dataset. If that dataset moved,
add `--dataset-root /path/to/dmc_expert_vision`; its identity must match the source.
The source's resolved dataset path is used otherwise. No separate offline
pretraining stage is performed. The goal-maintenance run is not a checkpoint source: it saved only
evaluations of the duration checkpoints.

Both models use their original goal scores at H25 throughout collection and
evaluation. Each collects **16,384 raw steps / 8,192 agent transitions** in 16
environments and performs **1,839 native online updates**. This stops partway
through the unchanged 80,000-step/10,000-update schedule, including its 1,024
transition warmup. Without `--retain-offline`, native training uses only fresh online replay; the existing
detached physical readout retains its configured 50% expert data. No reward
learner, physical-state controller, new loss or neural module is added.

Before training, at 8,192 raw steps (788 updates), and at the endpoint, evaluate
the same six validation starts for 200 decisions. Save current selected-plan
replays and rescore the **same initial plans** at every checkpoint. Compare
recursive predictions with one-step predictions supplied the real recent
observations. Real future observations remain diagnostic inputs only. Goal
scores on actual versus predicted futures separate preference errors from
forecast errors. Persistence-normalized errors and feature spread help avoid
mistaking a shrinking latent representation for improved prediction.

The five-hour target is a rough A100 estimate from the previous H25 evaluations,
allowing extra cost for 16-environment online planning. Most time is collection
and evaluation, not gradient updates. Work counts are fixed; there is no timing
calibration, budget feasibility gate or clock cutoff. Runtime depends on hardware.
This is a one-seed diagnostic of online changes, not a completed online benchmark
or proof of a repair. No settings are selected automatically from its results.

Results go to `runs/forecast_online_<timestamp>/` (or `runs/offline_online_decay_<timestamp>/`
with `--retain-offline`): report/summary, zero-action
control, per-model online metrics, before/midpoint/after traces and plan tensors,
milestone checkpoints, and periodic `latest.pt` weights/optimizer snapshots.
These snapshots do not include exact simulator/replay resumption; this runner
has no resume mode. The run name and handled status print at the end, including
failures and interrupts. Original inputs are never overwritten.

```bash
MUJOCO_GL=egl python -m scripts.check_forecast_online
```

`--dry-run` checks source artifacts and resolved budgets without writing outputs
or accessing expert data. A real launch validates expert identity before expensive
evaluation. `--task` also supports Reacher and Ball-in-cup when a matching duration
source exists; the default and current source are Cartpole only.

For the **goal-maintenance comparison using the completed duration checkpoints**:

```bash
bash scripts/run_goal_maintenance.sh \
  --source-run runs/paper_faithful_duration_20260923_020814
```

This freezes the saved TS and LeWM models and evaluates three planning conditions
on the same six validation starts for 200 decisions each: the saved native planner,
the native score with a longer lookahead, and the existing latent tail score over
the final portion of that longer lookahead. There is no retraining, expert-dataset
dependency, online updating, timing calibration, or clock cutoff.

The shared rule averages distance to the same goal image over the final predicted
window, allowing an approach before that window. This is an explicit planning
adaptation, not the original paper objective or a guarantee of stable control.
Native model architectures, losses and planner search settings are retained.
The defaults request at least **0.5 s lookahead** and **0.3 s holding window**,
rounded up using the simulator timestep and action repeat. A saved longer horizon
is preserved. For Cartpole these resolve to H25 and a 15-decision tail, compared
with the saved native H5. When the saved horizon already covers the requested
lookahead, the duplicate native condition is omitted.

`--tasks cartpole_balance_sparse reacher ball_in_cup` uses the same rule for all
three tasks when the source run contains their completed checkpoints and banks.
The supplied `020814` run contains only Cartpole. Requested horizons must fit the
saved branches. `--lookahead-seconds` and `--hold-seconds` explicitly change the
time windows; no settings are chosen from evaluation outcomes.

Add `--dry-run` to validate artifacts and print the resolved matrix without model
evaluation or output writes. Add `--score-only` to score the saved candidate
futures without policy execution. Branch scoring always uses the same longer
forecast horizon across conditions; the report distinguishes this from the
shorter horizon used by `native_saved` when acting. Actual-image and predicted
scores are compared against real task occupancy during the same final window,
with ties averaged and uninformative anchors reported separately. Actual-image
scoring still depends on the learned encoder.

Outputs are in `runs/goal_maintenance_<timestamp>/`: report and summary, matched
zero-action controls, policy traces, and selected-plan diagnostics. Only validation
starts are evaluated. Source checkpoints/banks are read-only; file hashes and
before/after model tensor digests are recorded. Every handled completion, failure
or interrupt prints `Run | <name> | status=<status>`. The runner records partial
results but does not resume them; use a new output directory for a repeat.

CPU/simulator implementation checks (both model families, all three tasks):

```bash
MUJOCO_GL=egl python -m scripts.check_goal_maintenance
```

For the **four-hour Cartpole training-duration comparison**:

```bash
bash scripts/run_paper_faithful_duration.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision \
  --reuse-banks runs/paper_faithful_duration_20260923_002626 \
  --ts-updates 24000 --lewm-updates 28000
```

This runs **two fresh fits on Cartpole balance sparse**, one TS and one LeWM,
with one training seed each. The defaults are **24,000 TS updates and 28,000 LeWM
updates**. Both finish those counts; there is **no timing calibration, runtime
feasibility check or clock cutoff**. Earlier Cartpole measurements put training
at about 86 and 88 minutes. Allowing roughly another hour across the pair for
setup and evaluation gives an approximate four-hour run, about two hours each.
The expanded evaluation has not been timed on the GPU, so this is a rough estimate.
The model widths, native losses, goal-image costs, five-decision
planning horizon and original solver settings remain unchanged. TS retains patch
curvature with no aggregate planning cost. There are no new controller modules or
RL objectives; this experiment remains offline-only.

Each learning-rate schedule uses its declared count from the start, and resume
keeps it unchanged. `--updates N` selects a common fixed count and also skips
calibration. `--minutes` is only a descriptive runtime label in the default fixed
mode; changing it does not shorten or extend training. `--dry-run` prints the two
resolved setups, fixed counts and evaluation settings without starting the run.

The bank-source command above reuses the failed run's Cartpole data; Reacher and
Ball-in-Cup data are not needed. Use `local/reports/...` if using the downloaded
copy, or omit `--reuse-banks` to collect Cartpole data afresh. Bank settings,
file/content hashes, dataset identity, simulator versions and collection code are
validated before reuse. The new run stores its own copy and source provenance;
the source stays unchanged. Reference checks and zero-action controls run again.
Offline batches remain 50% expert and 50% intervention
windows from the prescribed reset/random-roll-in bank.

Forecasts are measured at initialization and at 25%, 50%, 75% and 100% of each
model's update budget. Native control uses the same **four validation starts** for
100 decisions at all four trained milestones. The final test uses **all six
separate test starts**, 200 decisions each. Zero-action controls use matching starts
and durations. Both models see the same evaluation cases; update counts and elapsed
times are recorded. This compares roughly similar time, not identical updates or
strictly equal runtime.
These samples and one seed do not establish robustness across training seeds.

The earlier six-fit attempt stopped at its runtime-budget check. The current
fixed-count test does not use that check. Legacy timing modes remain available
only when explicitly selected with `--budget-mode equal-time` or `equal-updates`;
neither is used by the command above.

Simulator snapshots preserve episode targets, integration state, task RNG and
episode counters. Saved plan probes include the full chosen action sequence,
predicted/actual native costs, real future images, and diagnostic alternatives.
Simulator physical values and rewards never enter the native controller.

Each fit writes complete training checkpoints every 1,000 updates and at validation
milestones. Resume on the same machine/runtime with unchanged code and datasets:

```bash
bash scripts/run_paper_faithful_duration.sh \
  --resume runs/paper_faithful_duration_<timestamp>
```

Resume keeps the original update budget, learning-rate schedule, optimizer state,
both data samplers and random states. It does not extend a finished schedule.
Earlier `paper_faithful_offline_v1` evaluation snapshots are not accepted as exact
resumes. Outputs include `summary.txt`, `report.json`, banks, policy traces, plan
probes and per-fit checkpoints. The run name and report directory print at the end,
including handled failures and interrupts. This run performs no online updates.

```bash
MUJOCO_GL=egl python -m scripts.check_paper_faithful_duration
```

For the earlier **roughly one-hour offline-plus-online diagnostic**, use both small planners
on one scenario (Cartpole by default):

```bash
bash scripts/run_planner_training_check.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision --minutes 60
```

The time target is for both models combined, run sequentially. A short, disposable timing pass
measures training, collection and evaluation, then chooses the SAME offline/online update count
for both models. All timing-pass weights and replay are discarded; actual training starts from
scratch with fixed update budgets and schedules. This is an estimate, not a hard deadline.
`--updates N` skips timing and uses an explicit shared budget (at least 2,048 per phase).
Models use 64-wide representations, two-layer predictors, batches of 128 length-4 sequences,
BF16 neural compute, unchanged native loss/optimizer recipes, and reduced planner search budgets.
Episodes remain 500 agent steps. Online replay starts empty, with 1,024 warmup transitions and
four further agent transitions per update; only the detached head retains 50% expert labels.
No production settings, datasets or checkpoints are modified, and no checkpoints are written.

Fixed expert-held-out windows and separate zero/random-action episodes check physical errors at
initialization, after offline training and four times online. Neither validation source is fitted.
Policy returns use two complete fixed-seed episodes after each phase. `report.json`, `summary.txt`
and per-update `metrics.jsonl` preserve losses, coordinate RMSE/nMSE, gradient norms, timing,
settings and any regression alarms. Terminal progress is throttled to 30 seconds. `PASS` is an
execution/error guard, not proof of task mastery; `REGRESSION` is saved and returns a nonzero exit.

For a **short TS curvature A/B training check** on Lambda:

```bash
bash scripts/run_ts_ablation.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision
```

This runs only tiny Cartpole TS models: legacy flattened curvature versus the corrected per-patch
loss, both from identical fresh initialization. Each receives 1,000 expert updates and 512 adaptation
updates by default (`--expert-updates` / `--online-updates` override these). Expert batches and fixed
simulator replay are identical across arms; all other current training fixes remain enabled. Sixteen
200-step zero/random-action episodes are collected once for adaptation, with separate full-length
simulator validation episodes and held-out expert windows. The replay phase uses the native update
and the head's normal 50% expert retention, but deliberately removes policy optimization and feedback.
There is no timing calibration, checkpoint loading/writing, or policy evaluation. This is an isolation
test, not the production online loop or a reproduction of every historical setting.

The compact `summary.txt`, detailed `report.json`, replay indices and per-arm `metrics.jsonl` are saved
under `runs/ts_ablation_<timestamp>`. Physical RMSE, latent statistics, and gradient clipping are tracked;
nMSE is secondary. Guards check post-offline versus intermediate/final adaptation errors in original
units. If the legacy failure does not reproduce, the comparison explicitly reports `INCONCLUSIVE`.
Even an encouraging result does not establish long-run stability or successful control. No production
model code or settings are changed by this test.

To isolate **action conditioning versus encoder/BatchNorm/readout drift**, use the same launcher:

```bash
bash scripts/run_ts_ablation.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision --mechanisms
```

This skips the old-curvature comparison. It pretrains one corrected tiny TS model for 1,000 updates
once, then restores the identical weights and optimizer moments for three 512-update adaptation
controls: native, frozen BatchNorm, and frozen encoder. Batches, dropout seeds, and the head's online
schedule/50% expert retention are shared. These controls are diagnostic, not production recipe changes.
The same held-out images are decoded with old/new encoder and readout combinations; restoring only
the old BatchNorm buffers separately tests inference-statistic drift without changing current weights.
Crossed-head errors measure compatibility, not whether physical information is irretrievably lost.

Four extra simulator anchors provide zero/-1/+1 action branches at the native five-step Cartpole
horizon. The diagnostic compares predicted and actual encoded action effects, matched versus wrong
actions, and gradients through the action input/encoder. CUDA probes also compare BF16 with FP32 on
the same weights, without retraining. These branches are validation-only and collected once; no policy
optimization or extra head fitting is performed. Results go to `runs/ts_mechanisms_<timestamp>` with
one shared `offline_metrics.jsonl`, per-control metrics, `report.json`, and a compact `summary.txt`.
The previous physical-error guards still apply; this does not establish online control success.

To test **whether TS can fit action effects, separately from physical-head adaptation**:

```bash
bash scripts/run_ts_ablation.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision --fit-isolation
```

This skips both earlier ablation suites. One tiny Cartpole model receives 1,000 expert updates once.
The frozen encoder then caches real zero/-1/+1 simulator branches from four training anchors and
four anchors at a different validation seed. Two predictor/action-encoder controls start with the
same offline weights and optimizer moments: native dropout and diagnostic-only dropout disabled.
Each gets 2,000 updates on the same 60 cached four-observation windows, using the unchanged native
prediction objective and optimizer settings. The curvature term is constant with the encoder frozen.
Training-branch fitting and unseen-seed generalization are reported separately, at h1 and h5.
The action-blind baseline is the mean actual outcome across controls at each anchor, not persistence.
FP32 predictor probes retain the same cached encoder features; they are not separate FP32 training.

Separately, two head-only controls restore the offline head and use cached, unchanging features:
expert-only retention versus the current 50/50 expert/simulator mixture, with the normal online head
LR, warmup and fresh Adam moments. Each gets 512 updates. Expert fitting uses eight cached batches
from the training split; evaluation uses held-out expert windows. Four 200-step zero/random simulator
episodes supply fitting examples and two disjoint episodes supply validation. Per-source and
per-coordinate gradient measurements distinguish loss magnitude from actual gradient conflict.
Neither validation bank is fitted. The short cached pools are fitting diagnostics, not a reproduction
of the production replay distribution or a proposed recipe change.

Override budgets with `--expert-updates`, `--action-updates`, and `--online-updates` (head-only here).
Reports go to `runs/ts_fit_isolation_<timestamp>`: `summary.txt`, `report.json`, shared offline metrics,
and four small per-control metric logs. No checkpoint writes, planner optimization, policy evaluations,
timing calibration, or full-model benchmarks run. `COMPLETE` means execution succeeded; the declared
training-fit criterion is not evidence of long-run online stability or policy success.

To test a **short-rollout training candidate for both TS and LeWorldModel**:

```bash
bash scripts/run_rollout_check.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision
```

Each tiny Cartpole model is trained from scratch for 1,000 expert updates once. Two arms then start
from identical weights and optimizer moments: native one-step prediction, versus a 50/50 mixture
of native one-step and five-step autoregressive prediction. The candidate feeds its own predictions
back through the actual planner rollout, with gradients through all steps. The total prediction
coefficient, regularizers, dropout, optimizer and learning rates are otherwise unchanged.
This is an **experimental extension**, not an upstream reproduction or an enabled production fix.

Encoder/projector features, BatchNorm statistics and the physical head remain fixed during the
2,000 predictor updates per arm. Four training and two disjoint validation simulator seeds supply
identical cached branches for both arms, including zero, opposing constant, and random action sequences. Reports compare
teacher-forced errors, h1/h5 open-loop errors, action response, and predicted versus true-encoded
goal-cost rankings. Physical distances/rewards are scoring-only. The physical head cannot be called
by the fitting/rollout paths. Sparse-reward ties are reported as uninformative, not successes.
The rollout term supervises full-prefix futures; native windows also supervise earlier positions.
This compares objectives, including that target weighting, not a pure teacher-forcing toggle.

No checkpoint reads/writes, extra head fitting, timing calibration, precision sweep or full-model
benchmark runs. This is fixed-data adaptation, **not own-policy online training**. `COMPLETE` means
execution, not repair; assess held-out rollout accuracy and action rankings before a full training run.
Use `--models temporal_straightening` for TS only, or `--expert-updates` / `--fit-updates` to change
the budget. `runs/rollout_training_check_<timestamp>` contains `report.json`, `summary.txt`, and
one offline plus two fitting metric logs per model. Production settings are not changed.

Immediately before the production run, use the fuller preflight. It checks the installed GPU stack,
runs two expert updates and the short online lifecycle for every model, evaluates all thirteen variants,
and validates the resulting checkpoints and metrics:

```bash
./scripts/run_preflight.sh
```

For a **full-size training and resource check** using the already collected datasets:

```bash
bash scripts/run_training_smoke.sh \
  --dataset-root /absolute/path/to/data/dmc_expert_vision --updates 3
```

This runs every production model/scenario in a separate process, preserving model dimensions, batch
sizes, sequence lengths, optimizers, and planning budgets. It performs three expert updates, three
native online updates, and three batched environment collection steps. Online replay is seeded in
memory from complete **training-split** expert episodes so updates can run immediately; this is a
runtime check, not a learning experiment. The dataset is read-only and no model checkpoints are saved.
Only logs and diagnostic reports are written under `runs/training_smoke/`. The terminal ends with a
compact table (one row per run); `summary.txt` contains that same pasteable summary. Full per-phase
details, cold timings, losses, and resource samples remain in `report.json` and worker logs.

The default is one update/collection step; `--updates 3` separates cold compilation/initialization from
subsequent calls for more useful timing estimates. `--rollout-steps N` controls collection separately;
zero skips the simulator. Use `--scenarios reacher` or `--models dreamer/gru storm/mamba3` to narrow the
matrix, and `--dry-run` to inspect it without loading data or models.

`report.json` includes parameter counts, finite-loss/update checks, per-phase GPU allocated/reserved
peaks, sampled process GPU memory (including compute-visible environment children), device-wide GPU
utilization, worker host RAM/CPU usage, effective CPU thread counts, and estimated full-capacity replay
tensor memory. Per-phase CPU seconds and warm `cpu_core_equivalents` (CPU time / wall time, excluding
environment child processes) help distinguish CPU oversubscription from GPU waits. It projects
per-run and serial-matrix **training** time from configured update/environment-step budgets. The
reported range reflects expert data-prefetch overlap, not a confidence interval. Evaluation,
checkpoint I/O, and logging are unmeasured additions; longer contexts and different online trajectories
can change runtime and memory. Memory-only parallel-pair candidates include headroom, but require a
concurrent trial before assuming they are safe or faster. Profile on an otherwise idle GPU.

Measure the same two jobs **serially and concurrently** before enabling parallel training:

```bash
bash scripts/run_training_smoke.sh --dataset-root /absolute/path/to/data/dmc_expert_vision \
  --scenarios ball_in_cup --models dreamer/gru dreamer/s5 \
  --updates 10 --rollout-steps 3 --compare-parallel --output runs/parallel_check
```

`report.json` records both sets of diagnostics, wall-time speedup, and warm-phase throughput ratios;
`summary.txt` includes both sets of rates and memory peaks. Wall times include process startup and
compilation; repeat the check and inspect warm rates before extrapolating. Each worker keeps its own
model, replay, optimizer, seeds, and logs.
The benchmark does not save checkpoints or automatically change the production configuration.

For the current BF16 planner settings, use the focused concurrency check:

```bash
bash scripts/run_concurrency_benchmark.sh --dataset-root /absolute/path/to/data/dmc_expert_vision
```

This tests Dreamer/Mamba3 across three scenarios at once, TD-MPC2 across scenarios with two workers,
and three ball-in-cup pairs: TS + LeWorldModel, TS + STORM/Mamba3, and LeWorldModel + STORM/Mamba3.
It uses the production precision settings (TD-MPC2 FP32, the others BF16) and model sizes. There are
21 worker runs, including nine serial references measured with the same workload; pair references
are shared. It does not rerun the 117-worker matrix, precision comparisons, dependency/data audits,
expert-data collection, or evaluation. Each worker performs 32 expert and 32 online updates;
online updates are interleaved in bursts of four over eight collection calls. Dreamer additionally
continues collection to 64 calls to exercise longer actor histories. Checkpoints are disabled.

Use `--only dreamer`, `--only tdmpc2`, or `--only pairs` to rerun just one section, and `--dry-run`
to inspect the commands without using the GPU. Summaries and JSON are saved under
`runs/concurrency_bf16_check/{dreamer,tdmpc2,pairs}/`; `--output DIR` changes that root.
Worker failures are reported and later sections still run. Use an otherwise idle GPU and compare
warm update/collection rates, not just startup-inclusive speedups. Process-wide utilization averages
and sums of independent memory peaks do not establish available concurrent throughput.
Production uses the measured schedule described below; benchmark trials never automatically
change it. Final evaluations remain serial.

To test **the same model/variant across all three scenarios**, compare a serial baseline with two-
and three-worker queues:

```bash
bash scripts/run_training_smoke.sh --dataset-root /absolute/path/to/data/dmc_expert_vision \
  --scenario-workers 2 3 --updates 20 --warmup-updates 5 --rollout-steps 32 \
  --output runs/scenario_concurrency
```

By default this tests all 13 variants: 39 serial workers, then 39 at each concurrency level (117 total).
Use `--models storm/mamba3 tdmpc2/default` to narrow it. Each trial runs the same selected scenarios:
with two workers, the third scenario starts when a slot becomes free; with three, all start together.
Within each variant, the longest serial jobs start first. Different variants never share a concurrent
trial. The serial baselines are reused, and actual sampled batches must match each scenario's baseline.
No memory prefilter excludes tight triples, including Dreamer; CUDA OOMs are recorded as failures and
later trials continue. A failed trial gets no speedup, and the script exits nonzero if any trial failed.
Use this on an otherwise idle GPU: host OOM or a driver failure can still interrupt the whole check.

The compact `summary.txt` has one row per variant/concurrency level, with serial/concurrent seconds,
wall speedup, estimated warm phase speedups, summed worker memory peaks, and process-overlap fraction.
The memory sum uses the largest N sampled worker peaks, not simultaneously measured GPU occupancy.
Phase speedups simulate bounded queues using warm call times; they are not synchronized phase trials.
Full worker diagnostics and queue start/end times remain in `report.json`. This mode is separate from
pair/compile/storage comparisons, and does not change the production scheduler or enable compilation.

To test complementary pairs, reusing each serial baseline:

```bash
bash scripts/run_training_smoke.sh --dataset-root /absolute/path/to/data/dmc_expert_vision \
  --scenarios ball_in_cup --updates 10 --warmup-updates 3 --rollout-steps 3 \
  --pair dreamer/gru storm/mamba3 \
  --pair temporal_straightening/default storm/mamba3 \
  --output runs/runtime_check
```

The pinned **Torch 2.8 / Triton 3.5** Mamba environment does **not** support CUDA `torch.compile`:
Inductor expects Triton 3.4 APIs, including `triton.compiler.compiler.triton_key`.
Keep `model.compile=false` in this environment; do not downgrade Triton and break the Mamba kernels.
The setup check validates eager execution, not Inductor. CUDA compilation benchmarks check Torch's
declared Triton dependency before launching any workers and report mismatches without an eager fallback.
This dependency check is necessary, not sufficient: successful GPU forward/backward benchmarks are still required.
Only add `--compare-compile` in a separately validated, matching Torch/Triton environment. Use
`--models dreamer/gru --compare-compile` to test compilation alone without repeating the paired trials.

Dreamer's opt-in `model.compile=true` compiles tensor-only encoder, decoder, recurrent-core, posterior,
and prior modules in place. These are shared by expert updates, online training, history reconstruction,
and acting. Sampling, sequence loops, and optimizers stay eager; recurrent CUDA graph capture is not
enabled. Parameter/checkpoint keys are unchanged. Production defaults remain eager until measurements
justify enabling compilation; compile warnings and graph counts are recorded in the worker diagnostics.

For a storage comparison, stage the selected scenario on **local SSD**, not another path on the
same network mount. Keep the original dataset and persistent run outputs:

```bash
bash scripts/run_training_smoke.sh --dataset-root /absolute/path/to/data/dmc_expert_vision \
  --scenarios ball_in_cup --models tdmpc2/default \
  --updates 20 --warmup-updates 5 --rollout-steps 0 \
  --stage-storage /tmp/dmc_expert_vision --output runs/storage_check
```

Check free space and the mount with `df -h /tmp` before copying; `/tmp` is not guaranteed to be local
SSD on every machine. Staging copies only the selected HDF5 and metadata files, verifies SHA-256,
reuses unchanged verified copies, and refuses to overwrite different existing data. It records copy
time separately in `staging.json`; interrupted copies are restarted. No source files are changed.
Use `--compare-storage` instead for an existing copy. The benchmark also checks matching sampled batches. Storage,
compilation, and pair comparisons can be combined in one invocation. Each alternative is compared
separately to the same original-storage, eager, serial baseline. Baselines run first; OS/compiler
caches are not flushed, so these are warm-workload comparisons, not controlled cold-disk benchmarks.
With collection skipped, the projected hours omit collection and are only a training subtotal.

To stage all scenarios for production, run `python -m scripts.stage_dmc_data --source ORIGINAL_ROOT
--target LOCAL_ROOT`, then set `DMC_EXPERT_VISION_DATA_DIR=LOCAL_ROOT` and launch the full run with
`--override stages.collect=false`. Run outputs/checkpoints remain at the configured persistent location.
The local copy must fit on disk; staging all three image datasets can require substantial space.

Temporal Straightening uses 256 candidate trajectories per planning autograd pass and 32 planning
iterations. Model weights are temporarily frozen during action optimization; action gradients,
candidate counts, and horizons are preserved. Action embeddings and masks are reused. Goal images are
encoded once per action decision with current weights; planning never calls the physical readout.
One larger pass is not a guaranteed 2x speedup. Older physical-head-planning timings are not measurements
of the current controller.

```bash
bash scripts/run_planner_benchmark.sh --dataset-root /absolute/path/to/data/dmc_expert_vision
```

This targets TS and LeWorldModel on ball-in-cup, compares FP32 against BF16 mixed precision, interleaves
online updates with real collection, and profiles short deterministic evaluations at batch sizes 5
and 50. It uses production-sized models and saves no checkpoints. Options can be overridden, including
`--scenarios cartpole_balance_sparse reacher ball_in_cup`. The test is not a task-performance evaluation.
`--online-burst N` controls the number of online updates after a collection call; total updates remain
`--updates`. Warmup excludes whole bursts until at least `--warmup-updates` updates have completed.
Evaluation timings separate reset, first-call, and warm-step costs. The compact summary shows
`batch:warm,reset,first` seconds; JSON also separates policy/bookkeeping from environment stepping.
At least two evaluation steps are needed for a warm estimate. Short episodes do not bound late-episode memory.
Both precision cases use the same current planner iterations and candidate batches. The report measures
speed and memory, not equivalent task performance. There is no expert-data collection, dataset audit,
or dependency check. Use `--compare-bf16` with `run_training_smoke.sh` to select other models.

Generate an ordered phase breakdown from a complete report (later reports replace matching cases):

```bash
python -m scripts.estimate_runtime --report runs/scenario_concurrency/report.json runs/planner_bf16_check/report.json
```

`runs/runtime_estimate/schedule.md` and `schedule.json` group concurrent training jobs, keep final
evaluations serial, and separate pretraining, online updates, collection, and evaluation estimates.
For TS, the estimator selects the measured gradient-batch case matching the production setting.
Rows with outdated precision or planner budgets are marked missing rather than used for new estimates.
Evaluation projections count startup/reset once per batch and extrapolate only warm steps.
Older reports without this split use the labelled collection-rate fallback, not reset-inclusive averages.
Unmeasured state-prediction evaluation and checkpoint I/O are explicitly excluded. Existing timings
do not establish a speedup for newer code; only new measurements can update those estimates.

The wrappers use the `environment/` created by `scripts/setup_dmc.sh`. Set `PYTHON` to use another
interpreter. The smoke and full-run wrappers forward additional arguments to `main.py`; all three
configs can also be invoked directly as `dmc_smoke`, `dmc_preflight`, or `dmc_benchmark`.

The orchestrator keeps the terminal focused on stage progress, timing, training metrics, and final
results. Full commands, dependency warnings, and raw tracebacks remain in each collection or run log;
on failure, the useful end of the traceback and the exact log path are printed automatically.
In a terminal (including GNU Screen), each active worker has an in-place progress bar with counters,
ETA, and a small selection of losses. Expert, online, policy-evaluation, and held-out prediction
progress updates arrive at most every 30 seconds, plus stage completion. The parent owns the display,
so parallel workers cannot overwrite each other's rows. Redirected output uses ordinary lines without
terminal escape codes. Startup/model/data/checkpoint details update the live status row instead of
scrolling; redirected output omits these details too. Run labels omit repeated config names and paths.
Results, failures, and completion summaries remain visible. Full scalar metrics remain in each run's `metrics.jsonl` and TensorBoard;
raw worker output remains in `stdout.log` / evaluation logs. The parent also appends a plain-text
`orchestrator.log` under the experiment output directory, without requiring `tee`.

The production matrix runs all thirteen image-model variants on all three scenarios with seed 0.
It collects each scenario dataset once, concurrently (`collection.parallelism=1` selects serial
collection). Dreamer then trains the three scenarios of each variant together, controlled by
`training.scenario_parallelism.dreamer=3`. Variants and seeds do not overlap. These triples fitted the
40 GiB A100 benchmark with little memory headroom; set the value to 2 or 1 for a more conservative run.
TD-MPC2 uses two scenario workers (`training.scenario_parallelism.tdmpc2=2`). The ball-in-cup
LeWorldModel and STORM/Mamba3 runs form the one explicit `training.concurrent_runs` pair; TS and
LeWorldModel never overlap. Remaining families run serially (`training.parallelism=1`). Scenario
groups launch first, followed by the explicit pair and the remaining runs. Final evaluations run
serially after each concurrent training group has finished; periodic checkpoint-selection evaluations
still execute inside their respective training workers. Failures stop active workers, and the existing
resume behavior applies separately to each run.

For the completed datasets, launch the 10,000-expert-update recipe in a fresh output directory:

```bash
export DMC_EXPERT_VISION_DATA_DIR=/home/ubuntu/DMC/data/dmc_expert_vision
bash scripts/run_full.sh --override stages.collect=false --override output_dir=runs/dmc_vision_10k
```

Run directly inside Screen to retain the live display. Do not pipe through `tee` to
`orchestrator.log`: the orchestrator already writes that file. Checkpoints from a different
training budget cannot resume this recipe.

The shared launcher defaults `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1 so concurrent GPU workers
do not create competing large CPU thread pools during replay sampling. Existing environment settings
override these defaults. This applies to both the benchmark and production subprocesses; set these
variables explicitly when invoking `train.py` directly. Unfinished replay episodes are stacked once
per collection step and reused during the following update burst, without caching model states or
changing sampled windows.

Training writes each run under
`runs/dmc_vision/<scenario>/<family>/<variant>/seed_<seed>`, and evaluation writes
`evaluation.json` for `final.pt` and `evaluation_best.json` for `best.pt` in that run directory. Set
the booleans under `stages` to run only part of the lifecycle, add seeds after the initial matrix
succeeds, and select the compute device with
`device`. With `training.resume: true`, rerunning the same command skips completed evaluations,
evaluates finished training runs, and resumes interrupted expert or online training. Set
`training.overwrite: true` to delete existing runs and start them again.

The initial matrix uses one training seed, so it does not estimate training-seed variability.
Additional independent seeds can be run separately without mixing checkpoints:

```bash
python3 train.py --config-name offline_dmc_expert_gru_vision --multirun \
  scenario=cartpole_balance_sparse seed=0,1,2,3,4
```

Every training run names its config explicitly. Resume a checkpoint through the same entrypoint;
`training.online.steps` remains the target total number of environment steps:

```bash
python3 train.py --config-name offline_dmc_expert_mamba3_vision \
  scenario=cartpole_balance_sparse \
  resume_from=/absolute/path/to/latest.pt \
  training.online.steps=80000

tensorboard --logdir ./logdir
```

## Checkpoints

Checkpoint names identify their phase: `pretrain_latest.pt` resumes interrupted expert training,
`pretrained.pt` is the completed expert-pretraining state, `pretrained_best.pt` is its best
validation-return state, `latest.pt` resumes online training, `best.pt` is the best online
validation state, and `final.pt` is the completed online state. Checkpoints and evaluation JSON are
written atomically so an interrupted write does not replace the previous valid file. Production
evaluation reports `final.pt` as the primary result and `best.pt` as a supplementary result. The best
checkpoint is chosen lexicographically by mean validation return, sustained success, and then later
training step. Physical-state prediction is evaluated only for the primary final checkpoint.

Checkpoints retain their original source/runtime fingerprints. New checkpoints separately validate
model settings and the training recipe: logging/save intervals and unrelated source edits do not block
resume. Older checkpoints without compatibility metadata can only be evaluated with their original settings.
Source stays frozen within a running orchestration; restart the process after making changes. Resumed
checkpoints record their parent checkpoint and its provenance. Evaluation has its own version, so metric
fixes can be applied to compatible saved weights without relabeling their training provenance.

Intermediate checkpoint selection uses five policy episodes. All families skip rollout evaluation
during expert pretraining, then evaluate at the start of online training and every 20,000 steps through
80,000 steps. Reported final metrics still use 50 fresh episodes.

## Data And Metrics

Before training, audit all three collected datasets with:

```bash
python -m scripts.check_dmc_data --dataset-root /absolute/path/to/data/dmc_expert_vision
```

Omit `--dataset-root` to use `DMC_EXPERT_VISION_DATA_DIR` or the matrix's default path. This read-only
audit checks the HDF5 schema, completeness, train/held-out splits, every numeric transition, action
bounds, returns, termination/discount consistency, physical goal labels, and exact duplicate
trajectories across splits. Quality summaries cover expert episode returns (mean, population variance,
standard deviation, range, and percentiles), per-agent-step rewards, success/zero-return/perfect-reward
rates, first-success timing, action saturation/variability, and physical-state variability. These describe
the collected expert policy, not the world models. Ten chronological blocks per split and a held-out versus
training summary help reveal collection drift or differing expert performance. Detailed physical-coordinate
and initial-state statistics are included in JSON. Shorter episodes are weighted by their valid timestep
counts for step-level statistics; episode-return statistics weight episodes equally.
By default it reads every valid image from 16 reproducibly sampled episodes per split; add
`--full-images` to read every image, which requires substantially more disk I/O. No GPU or expert
checkpoint is needed. Run after collection has stopped.

Results are saved to `runs/dataset_audit/report.json` and one frame contact sheet per scenario.
`FAIL` exits nonzero for structural/data errors or detected split leakage; `WARN` flags quality
concerns such as mostly static episodes without rejecting them. Inspect the contact sheets as well:
without replaying the simulator, the audit cannot prove that images/actions/states are correctly
time-aligned. Use `--config-name dmc_smoke` for the smoke dataset, or `--override` for matrix overrides.

Each scenario is collected once into one 10,500-episode dataset. Episodes 0 through 9,999 are available
to training, while episodes 10,000 through 10,499 are reserved for evaluation. The RGB array is about
60 GiB before HDF5 compression. Collection also stores simulator state as supervised labels. Every
family receives images only; physical measurements never enter the encoder or policy as inputs.

Collection stores a two-coordinate task relation beside each image: cart position and pole-angle error
for Cartpole, finger-to-target for Reacher, and ball-to-moving-cup-target for Ball-in-Cup. These relations
are diagnostics only. The shared physical-state head is auxiliary in every family, never a controller.
All five families use the same targets within each scenario:

| Scenario | Learned physical targets | Outputs |
| --- | --- | --- |
| Cartpole | Cart position, pole cosine/sine, cart velocity, pole angular velocity | 5 |
| Reacher | Cosine/sine of each joint angle, fingertip-to-target vector, joint velocities | 8 |
| Ball-in-Cup | Cup/ball positions and velocities in the simulator's planar x/z coordinates | 8 |

Reacher angles are encoded on read, removing arbitrary full revolutions from the labels. Cartpole
already supplies cosine/sine. Velocities remain useful history-dependent targets, not quantities that
can be uniquely recovered from one frame. Pixel resolution and occlusion still limit observability.
Existing HDF5 observations and collection metadata are unchanged; no recollection is needed.

Its own Adam optimizer minimizes Smooth L1 state error (beta=1, unit physical-coordinate scales)
on detached observed features, using 256 targets per model update. Features are recomputed in
inference mode after the native update, without changing native RNG or BatchNorm statistics.
New heads use unit physical output scales, including unit sine/cosine scales; narrow expert
variance no longer suppresses the output Jacobian. The output affine is preserved on checkpoint
load, independently of loss scaling. Expert means
and standard deviations remain fixed for evaluation: nMSE keeps its original meaning, including
sensitivity to small expert variance. Online readout updates use LR 3e-5, a 100-update linear
warmup, and fresh Adam moments. Half of the existing 256 labels come from training-split expert
windows encoded by the current model; the other half come from online replay. This is **only an
auxiliary readout change**, independent of the optional native expert replay below.
The separate expert sampler and online head counter are checkpointed for resumption. Its dataset
statistics are not rescanned. Online-only runs without a dataset must explicitly set
`state_head.online.expert_fraction=0`; they use zero mean and unit output scale.
The physical loss updates only the head. Its outputs and weights cannot affect native updates or
action selection. Online replay
computes identical labels from the simulator. The held-out range supplies prediction evaluation data.
Dataset metadata fingerprints the expert checkpoint, collector, external TD-MPC2 source, and collection
runtime so an interrupted collection cannot resume into a mixture of incompatible trajectories.
Completed datasets are reused after checking their task, splits, action repeat, image size, and HDF5
layout. Reuse does not load the expert or require its current checkpoint path, source hash, or runtime
to match, and leaves the original dataset identity and provenance intact.

Evaluate an image-model checkpoint on that held-out dataset. Every family reports fresh policy
return, late-episode success, first-hit/sustained-success diagnostics, and physical-state prediction.
Primary success requires the reward threshold on at least 90% of the final 20% of each episode
(`evaluation.maintenance_occupancy` and `evaluation.maintenance_fraction`). Evaluation protocol
`dmc_evaluation_v11` leads with physical-state RMSE separately for each quantity and prediction horizon,
without dividing by dataset variance or averaging unrelated units. JSON stores the primary scores in
`physical_state_prediction.physical_rmse["<horizon>"]["<coordinate>"]`, with units in `physical_units`.
Positions use metres; linear/angular velocities use m/s and rad/s. Cartpole's `pole_angle` and Reacher's
`joint_angle[0]`/`joint_angle[1]` use shortest wrapped angular errors in radians. Angles are recovered
from the mean predicted sine/cosine coordinates, not by averaging angles across the +/-pi boundary.
Undefined predicted orientation vectors receive the maximum angular error (pi), never a free zero.
The existing `rmse`, `observed_rmse`, and persistence fields retain the raw target-coordinate scores,
including unitless cosine/sine errors, for compatibility and orientation-magnitude diagnostics.
Separate `derived_rmse` and `derived_observed_rmse` report Reacher fingertip position/velocity and
Ball-in-Cup target separation/ball-minus-cup velocity. These use analytic DMC geometry, with no extra
learned outputs; kinematics are calculated per rollout before averaging predictions.
Evaluation uses the head saved in the checkpoint, with no fitting on held-out data. A shared per-token
projection and MLP decode LeWorldModel's and Temporal Straightening's native latent-history windows;
TS retains ordered spatial patches. Dreamer and STORM use temporally conditioned state, while TD-MPC2
uses its stacked-image latent. Forecast history contains only the observed prefix and predicted future
features, never actual future observations. `observed_rmse` separately measures decoding of observed
features, helping distinguish readout error from accumulated dynamics error. JSON records
`readout_history_length`, `readout_updates`, and `readout_examples`. RMSE still depends on the readout,
not only the dynamics; do not interpret it as a decoder-independent measure of world-model quality.
All models now receive the same 64-image prefix and its 63 actions, with no earlier episode history.
They then consume the same recorded future actions, feeding back only predicted internal states.
Physical measurements are decoded for scoring, never fed back into dynamics. Errors are reported after
1, 5, 10, 25, 50, and 100 agent actions. Real future images are used only in the separate observed-state
decoding diagnostic. Native memory mechanisms are unchanged: default STORM still retains its configured
window, planners retain their short histories, and TD-MPC2 constructs frame stacks inside the common
prefix. `persistence_rmse` holds the last decoded observed state constant; it does not receive true
simulator state. `derived_persistence_rmse` provides the corresponding kinematic baseline.
`physical_observed_rmse` and `physical_persistence_rmse` provide the primary physical/angle versions.
`physical_true_persistence_rmse` holds the last true prefix state fixed and supplies a model-independent
baseline (the terminal `hold` values). True states are used only to score this baseline, never as model
inputs. `true_persistence_rmse`/`derived_true_persistence_rmse` retain its raw/kinematic counterparts.
These reporting changes do not alter any native training loss, physical-head loss, or checkpoint weights.
Existing checkpoints can be re-evaluated without retraining; wrapped angle RMSE cannot be reconstructed
from old aggregated cosine/sine RMSE alone.

Prediction windows come only from the held-out episode range, never the training range. The evaluator
rejects overlapping splits and attempted reads from training episodes. The 128-window default budget
is split equally between uniform starts and motion-focused starts. For the latter, select the most
active forecast segment among eight candidate starts in its held-out episode, using squared position
increments scaled by that episode's coordinate ranges (floor 1e-3). These labels affect selection only,
not model inputs or fitting. This favors motion when available; it cannot create movement in stationary
data. Aggregate scores describe this stratified benchmark, not a uniform sample of task experience.
`cohorts.uniform` and `cohorts.motion` separately report all errors and baselines. Every JSON records the
exact `windows` (episode, start, forecast start, cohort, motion score), reproducible from `state_seed`
independently of model family or training seed. Models receive identical windows within each scenario.

Dreamer/STORM use native latent sampling;
`state_samples` defaults to eight rollouts whose decoded physical predictions are averaged before RMSE.
Deterministic planners need one rollout. `state_batch_size` caps concurrent rollout samples, including
these repetitions. Evaluation records the history policy, sample count, and seed.
The `dmc_physical_state_v2` target layout requires fresh v20 training checkpoints. Older heads cannot
be resumed under the new meanings (including Reacher, whose output width stays eight). Existing
datasets remain reusable. The v8 evaluation changes do not require retraining v20 checkpoints; rerun
evaluation to obtain the common-prefix metrics. Do not mix previous prediction scores with v8 scores.
LeWorldModel and Temporal Straightening remain reward-free during representation training;
validation return only selects checkpoints. Recipe 8 / evaluation v10 uses physical goals rendered
into native goal observations, not the physical evaluation head. Goals are specified in
`scenario.goal_observation`, and `jepa_model.goal.source=physical_render_v1` identifies the interface:

- Cartpole: configured cart position in metres and pole angle in radians; default centered/upright.
- Reacher: the current episode's target location, with a fixed configured inverse-kinematics elbow
  branch. No current arm state or future trajectory is used to select the goal.
- Ball-in-Cup: configured cup joint displacement and target-minus-ball offset; default centered cup,
  ball at its target. Point-mass also supports a fixed physical position goal.

A separate, unshared physics copy renders the goal using the same camera and preprocessing as live
images. Only Reacher's task target is copied each reset. Goal construction checks task success and
joint limits once per new goal, never rolls out candidate actions, and does not alter live physics or
RNG. Fixed images are cached; goal embeddings are refreshed each decision as the encoder changes.
Goal images are not inserted into replay or representation/readout training.

LeWM compares the terminal predicted embedding to the goal using summed squared coordinate errors.
New TS configurations use upstream MPC's exponentially weighted trajectory error (`objective=ts_mpc`),
averaging visual tokens/features and retaining the upstream temporal reduction and observed prefix.
CEM and gradient search retain their existing budgets. The physical-head cost and action penalty
remain removed. Saved configs without an objective field retain the old terminal-only behavior.
TS remains an explicit image-only adaptation; rendering a single goal pose does not impose zero
velocity or represent every pose in the task's success region. Goal observations are additional task
specifications for these two families, not live proprioceptive inputs or held-out demonstrations.
Do not mix the new policy returns with evaluations of the old physical-head controller:

```bash
python3 -m scripts.evaluate_dmc \
  --config-name leworldmodel_dmc_vision \
  --scenario cartpole_balance_sparse \
  --logdir /absolute/path/to/the/run \
  --dataset "$DMC_EXPERT_VISION_DATA_DIR/cartpole_balance_sparse"
```

### Readout And Representation Diagnostics

Compare LeWorldModel and Temporal Straightening's `pretrained.pt` and `final.pt`
on identical held-out expert windows, across all three scenarios:

```bash
bash scripts/run_readout_diagnostics.sh \
  --run-root runs/dmc_vision_10k \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision
cat runs/readout_diagnostics/summary.txt
```

No training, checkpoint writes, simulator episodes, or planner optimization run.
The script reads the saved training configurations, checks checkpoint/dataset identities,
and samples 64 common windows per scenario, half uniform and half motion-selected.
Use `--scenarios ball_in_cup` or `--models temporal_straightening` to narrow the run.
`--device cpu` is supported; CUDA is the default.

- **Readout:** Summaries lead with physical-unit RMSE and wrapped angle errors, not aggregate nMSE.
  Compare physical errors from real-image latents against open-loop
  predicted latents at horizons 1, 5, 10, 25, 50, and 100. The CSV includes original-unit
  RMSE, expert normalization scales, and each coordinate's contribution to normalized
  error. Both true-state and decoded-state persistence baselines are included; terminal `hold` means
  true-state persistence. Diagnostic JSON keeps nMSE as a secondary metric; existing error guards and
  training objectives are unchanged.
- **Representation:** Compare centered latent variance, effective covariance rank,
  temporal change, latent forecast errors, and response to zero/random future actions.
  TS patches retain their ordering rather than being averaged away.

The compact `summary.txt`, detailed `report.json` (including exact windows and cohort
breakdowns), and `physical_errors.csv` are saved under `runs/readout_diagnostics`.
Increasing real-image decoding error points to a readout/representation problem;
good decoding but poor forecasting points toward dynamics. Tiny expert scales can
inflate normalized errors, so inspect original-unit RMSE too. Falling variance/rank
and weak action sensitivity are warning signs, not proof of collapse. Sensitivity is
not counterfactual accuracy, and successful expert trajectories do not establish
accuracy on failed online-policy states. Compare latent statistics within each model,
not absolute values across architectures.

Run the CPU regression tests independently:

```bash
../environment/bin/python -m scripts.check_planning_diagnostics
```

These cover analytically known errors/ranks, action-blind predictors, held-out-only
sampling, batch invariance, report serialization, and both real model implementations
without fitting, future-image leakage, or mutation of weights/normalization buffers.

### Fresh Physical-Head Diagnostic

Before another online run, isolate whether LeWorldModel/TS's frozen expert features can
support a usable physical readout. This diagnostic trains new heads, not world models:

```bash
bash scripts/run_fresh_readout.sh \
  --run-root runs/dmc_vision_10k \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision
```

The default checks both Cartpole models. Add `--scenarios cartpole_balance_sparse reacher ball_in_cup`
to check all six checkpoints. Each scenario collects **one shared pool**, reused by both models:
32 expert TRAIN episodes plus eight zero-action and eight random-action training episodes;
validation uses eight different expert TRAIN episodes and four new episodes per simulator policy.
Expert validation episodes were seen during native pretraining, but not during fresh-head fitting.
The final benchmark held-out split is untouched. Episode IDs, seeds, hashes, and exact diagnostic
windows are recorded. Zero/random actions broaden coverage but do not guarantee recovery behavior.

The original encoder/dynamics and normalization statistics stay frozen. Features and fixed-action
forecasts are cached once, with full causal histories and ordered TS patches. A fresh head first
fits a fixed 32-example mixed batch for 1,000 updates. A **separate fresh initialization** then
trains for 5,000 updates using the checkpoint's head batch size and pretraining head LR, without
online warmup. Each batch contains 50% expert, 25% zero-action, and 25% random-action examples;
episodes are sampled uniformly within each source. Only training labels determine physical output
and loss scales (position floor 0.1, velocity floor 1, unit sine/cosine scales). Expert evaluation
std is retained unchanged; the old head's narrow output parameterization is not inherited.

Reports compare the original and fresh heads on fixed training/validation windows, with observed
decoding, action-conditioned forecasts at 1/5/10/100 agent steps, and true/decoded persistence.
They include coordinate RMSE, nMSE, angular/goal-relation errors, false goals, missed goals, and
success/failure counts. `FIT` checks whether every small-batch RMSE / training scale is at most
0.05; it is not a policy-quality threshold. `COMPLETE` only means execution succeeded.
Training has a fixed budget, without validation-based stopping or fitting.

No planner optimization, native updates, online training, checkpoint saving, or production setting
changes occur. A timestamped `runs/fresh_readout_*` directory contains a compact `summary.txt`,
`report.json`, `physical_errors.csv`, and per-model `metrics.jsonl` with every fitting update.
The frozen feature cache lives only in memory. On failure, completed results and `error.log` survive.
Use `--output` to specify a new report directory. This diagnostic does not establish online stability;
its purpose is to choose between a readout repair and broader native training, before paying for either.

CPU regression checks: `python -m scripts.check_fresh_readout`.

### Fixed-Replay Model Adaptation

To isolate representation/statistic drift without any new collection or expensive planning:

```bash
bash scripts/run_fixed_replay_check.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision \
  --run-root runs/dmc_vision_10k
```

Defaults cover cartpole LeWorldModel and TS. Each needs its existing `pretrained.pt` and a
`latest.pt` containing online replay. The script partitions saved replay by episode, reads
32 expert TRAIN episodes plus 8 development-validation episodes, and reserves 8 replay
episodes for validation. These expert validation episodes were seen by native pretraining,
but not by fresh-head fitting; the benchmark held-out split is untouched.

A shared fresh, fixed-unit head fits for 1,000 head-only updates on cached training features,
50% expert and 50% replay. Then three trials each start with the **same** native checkpoint,
native optimizer moments, fitted head, and preselected replay batches:

- `native`: ordinary native updates, with the corrected head and TS clipping guard.
- `frozen_bn`: freeze all BatchNorm running statistics; affine parameters still train.
- `frozen_encoder`: also freeze the image encoder and its output projector, including dropout;
  the predictor, action encoder, prediction projector, and detached head still train.

Each trial performs 256 native updates, retaining the original full-schedule learning-rate
warmup, native batch size, source-episode count, and native losses. Each head update keeps
the existing 128 online plus 128 expert labels. The fixed replay comes from the saved run,
not the adapting model: this is an isolation experiment, **not** a new online benchmark.
It does not establish policy quality or guarantee that 1,000 head updates are sufficient.
The three modes share data within each model; saved replay can differ between models.

Validation at updates 0/64/128/192/256 measures observed decoding and open-loop horizons 1/5,
physical/angle errors, false and missed goals, and latent drift. Native preclip gradient norm
and clipping frequency are recorded. The finite TS native clip is `1.0`; production JSONL
and TensorBoard also receive `grad_norm`, `grad_clipped`, and
`grad_clip_fraction_since_load` (a running fraction for this process, not a persisted counter).
BN/encoder freezes remain diagnostic-only, and no multi-step loss is added.

The timestamped `runs/fixed_replay_*` directory contains a compact `summary.txt`, `report.json`,
`physical_errors.csv`, per-update JSONL, and exact batch indices in each run's `batches.json`.
All checkpoints/datasets are read-only. `COMPLETE` means finite execution, not repaired models;
compare original-unit errors as well as nMSE. Override `--updates`, `--head-updates`, or
`--eval-every` to change only the diagnostic budget. No simulator, planner, dataset audit,
or checkpoint saving is performed. CPU checks: `python -m scripts.check_fixed_replay`.

### Planner Objective Versus Dynamics

Before another training run, separate incorrect forecasts from an incorrect action-ranking
objective using read-only checkpoints:

```bash
bash scripts/run_planner_oracle.sh \
  --run-root runs/dmc_vision_10k \
  --latent-goals
```

Defaults test Cartpole LeWorldModel/TS, `pretrained.pt` and `final.pt`. Eight real simulator
prefixes (two seeds, zero/random roll-in, two offsets) each receive the same 16 candidate
action sequences. All candidates start from copied integration state, task RNG, counters,
and episode geometry. Simulator branches are collected **once per scenario** and reused
across both models and checkpoints. No dataset is needed or audited. There is no fitting,
optimizer step, head evaluation, policy optimization, or checkpoint write.

Three quantities are compared: native latent cost on **predicted** outcomes; that same cost
on encoded **actual simulator** outcomes; and actual cumulative reward over the candidate
horizon. Good forecast/oracle agreement but poor oracle/reward agreement implicates the
goal objective or encoder geometry, even with perfect dynamics. Good oracle/reward agreement
but poor forecast/oracle agreement implicates forecasts. Both can fail. Constant returns
or costs are marked uninformative, not counted as successful action ranking. Candidate-set
regret averages tied selections and is not a claim about optimized or whole-episode policies.
For tied sparse rewards, a separate rank compares oracle latent cost with terminal squared
physical goal relations divided by task tolerances. This is a dense diagnostic proxy, not reward.

The compact `summary.txt` and detailed `report.json` include rank correlations, selection
regret, per-candidate rewards/costs/actions, physical goal relations, open-loop versus
teacher-forced latent errors, and true-latent persistence. Native probes use small real
simulator clip batches with dropout disabled: they report BatchNorm train/eval gaps and
component gradient norms, then restore all running statistics and RNG. These correlated
diagnostic batches are **not** production batches or an online adaptation experiment.

`--scenarios`, `--models`, and `--checkpoints` restrict the work; `--sim-seeds`,
`--rollin-steps`, `--candidates`, and `--batch-size` control only diagnostic sampling/compute.
The checkpoint's native horizon is retained. `--latent-goals` explicitly tests the current
rendered-goal cost on legacy physical-head-controller weights; it does not reproduce old
policy returns. Reports distinguish checkpoint recipe from the current native loss recipe.
Use a new `--output` directory, or accept the timestamped `runs/planner_oracle_*` default.

Standalone CPU checks: `MUJOCO_GL=egl python -m scripts.check_planner_oracle`. These cover
simulator isolation and real-step parity in all three tasks, tied/constant rankings, native
loss/gradient formulas, causal indexing, and optimized versus explicit recursive rollouts.
The formula references are [LeWM's native loss](https://github.com/lucas-maes/le-wm/blob/main/train.py)
and [TS's visual prediction/curvature loss](https://github.com/Agentic-Learning-AI-Lab/temporal-straightening/blob/main/models/visual_world_model.py).
These do not claim full numerical parity with the upstream architectures or validate learning quality.

### True-Future Goal Selection Check

When short random branches give almost identical rewards, test the goal objective without
calling either model's predictor or physical head:

```bash
bash scripts/run_goal_objective_check.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision
```

This pretrains the same tiny Cartpole LeWorldModel and TS encoders used in the rollout
diagnostic for 1,000 expert updates each, then freezes them. No online adaptation, new loss,
planner optimization, checkpoint reads/writes, or production changes. Simulator data is
collected once: 12 mirrored/jittered balanced, boundary, and failure states; 21 candidate
sequences per state; actual futures at horizons 5, 25 and 100. Only endpoint images are rendered.
Horizon 5 matches the current planner; longer lookahead is diagnostic only.

The native terminal latent-goal cost ranks those true futures. Selection is compared with
uniform random choice, actual candidate-best reward, and a physical-distance baseline.
Two candidates come from privileged simulator LQR feedback, recorded as fixed action sequences
shared by both models. They improve coverage of recoverable outcomes, but are **not** a proposed
learned controller. Simulator reward/state labels never enter the model's cost or training.

`summary.txt` reports normalized return, selection regret, reward contrast and recovery rate;
`report.json` includes per-state/action/horizon scores, velocities, cohort breakdowns, configs
and hashes. Flat-reward cases do not count as evidence; unavailable recovery is not an objective
failure. These are constructed states and finite-candidate short branches, not full policy returns.
Reports and offline metrics go to timestamped `runs/goal_objective_check_*`. Defaults do not
repeat any prior online, loss-ablation, or throughput benchmarks.

CPU and real-simulator regression checks: `MUJOCO_GL=egl python -m scripts.check_goal_objective`.

#### Upstream Input-Dropout Correction

Both upstream predictors use `emb_dropout=0` separately from transformer `dropout=0.1`:
[LeWM config](https://github.com/lucas-maes/le-wm/blob/main/config/train/model/lewm.yaml),
[TS config](https://github.com/Agentic-Learning-AI-Lab/temporal-straightening/blob/main/conf/predictor/vit.yaml).
Previously we incorrectly used the transformer dropout probability at the predictor input too.
New training configs restore the upstream setting. Native prediction/regularization losses,
terminal latent goal distance, model sizes, planner budgets and independent physical heads
are unchanged. No goal-metric calibration or physical-label supervision of planning is added.
This is a verified configuration mismatch, **not a demonstrated cure** for goal ranking or
online degradation. Encoder capacity, data coverage and the visual-only TS adaptation still differ
from upstream; those differences are not changed by this correction.

To isolate the correction with the existing tiny Cartpole test:

```bash
bash scripts/run_goal_objective_check.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision \
  --compare-embedding-dropout
```

This fits two arms per model (1,000 expert updates each), checking matching initial weights
and replay sampler state. It collects simulator cases once and scores the same cases with
the unchanged native objective. Baseline/candidate reports and offline metrics are separate;
there are no online runs, checkpoint writes, new planner searches or throughput benchmarks.
Use the ranking/recovery comparisons, not just training loss, to decide whether to proceed.
Lower input dropout can also increase overfitting; improved performance is not guaranteed.

Saved configs lacking `emb_dropout` retain the old training behavior. Existing weights remain
loadable with their saved config and have identical evaluation computations; this change does
not repair already-trained weights. New configs have a different compatibility fingerprint,
so do not silently resume an old run as the corrected experiment.

### Paired Planning-Horizon Check

Test longer lookahead without changing the goal objective or retraining for each horizon:

```bash
bash scripts/run_planning_horizon_check.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision
```

This fits each tiny Cartpole LeWorldModel/TS once for 1,000 expert updates with the corrected
input dropout, then freezes its weights. Each model runs native closed-loop planning at
horizons **5, 15, 25** (0.1, 0.3, 0.5 simulator seconds) for **100 agent steps** from the same
12 constructed balanced/boundary/failure starts. Two real zero-action transitions provide a
common three-frame prefix. Planner caches reset between trials; samples/restarts, iterations,
goals and per-call RNG seeds are identical across horizons. Longer horizons have different
random tensor shapes and greater compute; this is not an equal-compute comparison.

The same weights also rank shared candidate trajectories using **predicted** versus **actual**
future images. All horizons use prefixes of one candidate bank. The summary compares only
anchors with informative returns at every horizon; JSON also includes all-anchor averages,
individual scores and physical baselines. The privileged simulator-feedback candidates are
diagnostic ranking controls only, never supplied to the native closed-loop planners.

There are no online updates, dropout ablations, checkpoint reads/writes, dataset audits or
production changes. The auxiliary physical head is fitted during pretraining but never used
for action selection. Short controlled trials do not establish full-episode success or online
learning stability. `--policy-steps` changes trial length, not training; the runner rejects trials
that would cross the original episode boundary. The terminal prints progress and a compact
six-row summary with returns, tail success, planning time and candidate rankings.

Output: `runs/planning_horizon_check_<timestamp>/{report.json,summary.txt}`, plus one
`offline_metrics.jsonl` per model and `horizon_*/policy_metrics.jsonl` containing each step's
actions, simulator rewards/states and timings. CPU/simulator regression checks:
`MUJOCO_GL=egl python -m scripts.check_planning_horizons`.

### Staged Planner-Recipe Check

```bash
bash scripts/run_planner_recipe_check.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision
```

This is a **diagnostic**, not a production recipe change or an automatic online rerun:

- **Optimizer stage:** one fresh tiny TS fit (1,000 updates), then frozen-weight control
  from six shared Cartpole starts. Compare current production planning settings (16
  restarts, 32 iterations, noisy random initialization), a 100-iteration single-restart
  zero-initialized `tanh` control, and direct train-standardized action optimization
  (Adam 0.1, 100 iterations, zero physical-action initialization/no noise). The latter two match budgets;
  comparisons against current settings change multiple factors. Direct actions are projected
  to DMC bounds, an explicit adaptation from upstream. All three keep terminal latent cost.
- **Reference stage:** each family gets a separate, larger offline fit (3,000 updates).
  LeWM uses a 12-layer/192D encoder, six-layer/16-head predictor and 2,048-wide projectors.
  TS uses a base-32 spatial ResNet with eight channels and a six-layer/16-head predictor.
  Batches are 128/32 respectively, with one sequence per source episode. Five consecutive
  stored actions form each transition, with observations at offsets 0/5/10/15. Action
  normalization uses training episodes only and is shared between fitting and planning.
- **Goal-reaching validation:** select 12 nontrivial start/goal pairs from distinct heldout
  episodes. Replay the recorded prefix and verify that the recorded actions still reach
  the goal after restoring simulator state. Compare expert replay, zero actions and the
  frozen learned planner over ten action blocks. Default pose tolerances are 0.01 m/rad.
  Auxiliary state predictions, rewards and privileged expert actions never score plans.
  A separate candidate-ranking diagnostic compares learned predictions with real futures.

Important limits: the reference uses **224px upsampling of stored 64px images**, not native
224px detail; action repeat two remains, so each block covers ten raw DMC steps. This is
a vision-only local adaptation, not an exact upstream reproduction, equal-parameter test,
or full paper training budget. This historical diagnostic explicitly retains the old terminal
objective, not the corrected TS production MPC objective. LeWM retains local bounded physical-coordinate CEM
proposals, with train-standardized model inputs. Reaching a trajectory pose does not establish sustained
balancing or velocity matching. These differences are printed and recorded, not hidden.

`OFFLINE_CONTROL_OBSERVED` requires at least eight reproducible nontrivial pairs, successful
expert replay on all pairs, learned goal-reaching >=50% and >=25 percentage points above
zero actions, and smaller average closest pose error. This is a screening rule, not statistical
proof. Otherwise the result is `NO_CONTROL_EVIDENCE` or `UNVALIDATED`; neither is called PASS.
Insufficient pair coverage skips the expensive reference fit. No online training is launched.

Use `--stage optimizer` or `--stage reference` to run only one stage without repeating the
other. `--models` selects reference families; `--optimizer-updates`/`--reference-updates`
set the separate training budgets. Existing checkpoint files are neither read nor written.
Results go to `runs/planner_recipe_check_<timestamp>`: compact `summary.txt`, detailed
`report.json`, and per-stage offline/control JSONL logs. CPU/simulator checks:
`MUJOCO_GL=egl python -m scripts.check_planner_recipe`.

### Action-Conditioning Check

```bash
bash scripts/run_action_conditioning_check.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision
```

This follow-up isolates weak action responses without another planner sweep. It fits TS
and LeWM once each for 3,000 offline updates using the **same larger vision-only reference
adapters described above**. Defaults are Cartpole, five stored actions per block and 12
heldout start/goal pairs. It does not change production settings, fit diagnostic heads,
optimize plans, run online training, or write checkpoints. Native offline updates still
include the existing detached auxiliary-head fit; that head is never used by these probes.
Goals are recorded trajectory endpoints, not equilibrium-goal task-success evaluations.

- Replay ten counterfactual action sequences from each identical simulator anchor, once
  for both models. Record real futures at all five horizons. Compare correctly paired,
  shuffled and zero future actions while retaining the same history and targets.
- Separate teacher-forced errors from recursive errors, and measure predicted versus
  real-encoded action responses and goal-cost ranges. Trace opposing actions through
  the action encoder, every predictor block, final normalization and prediction projector.
- Compare FP32/BF16 predictors on identical cached native-precision encoder features.
  The first two pairs also get latent-projection and goal-cost action gradients at every
  horizon, plus an FP32 finite-difference check. Inspect encoder/projector goal geometry
  separately, excluding the exact expert goal from the nonexpert rank statistic.
- For TS, compare mapped identical weights with **actual upstream** action embedding,
  predictor and recursive rollout methods, before and after fitting. Four small source
  files are downloaded once from pinned commit `2c3c7666a69a730042590d548c6731259c4183ac`
  into ignored `local/upstream_ts/`; SHA256 checks reject unexpected contents. Only the
  hard-coded mask device is patched. The adapter supplies cached visual features and
  zero-width proprioception; this is component parity, not whole-model reproduction.
- Initial TS checks also compare the actual upstream visual loss, cached-latent and parameter
  gradients, a matched optimizer step, and MPC cost/action gradients. These use FP32 with
  TF32, AMP and dropout disabled; copies retain optimizer moments. This does not validate
  upstream encoder weights, proprioception, or stochastic dropout equivalence.
- A 256-update fitting control uses disposable predictor copies, the native one-step loss,
  frozen encoder/projector, and disjoint simulator episode splits. Optimizer moments are
  retained but the native adaptation LR schedule starts fresh, avoiding LeWM's exhausted
  pretraining rate. It records actual weight changes and reports teacher-forced
  and recursive errors before/after; endpoint geometry remains a separate diagnostic.
  Set `--fit-updates 0` to skip it, or `--parity-only` to stop before the offline fit.

`--models temporal_straightening` restricts the run to TS. The previous diagnostic did not
save weights, so the fresh offline fits cannot be skipped. Initial upstream mismatch stops
the affected fit early. `COMPLETE` means diagnostics executed, **not** repaired policy quality;
latent MSE is not directly comparable between models. Reports, compact summaries and
offline metrics go to `runs/action_conditioning_<timestamp>`. No broad dataset audit or
previous optimizer/policy sweep is repeated.

Local regression tests: `MUJOCO_GL=egl python -m scripts.check_action_conditioning`.
Upstream parity tests use the cached source (or `TS_UPSTREAM_CACHE=/path/to/cache`);
without it, those tests explicitly skip rather than silently substituting local code.

### Paired Planner Corrections

```bash
bash scripts/run_planner_fix_check.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision
```

This bounded Cartpole test fits one small model per family for 1,000 offline updates, then
compares controllers using the same frozen weights, candidate actions and simulator starts.
It runs no online training and writes no checkpoints. TS upstream inference/training parity
must pass before its fit starts. Reports and compact summaries go to `runs/planner_fixes_<timestamp>`.

The arms are the legacy terminal objective; corrected upstream MPC (TS only); each family's
native objective with a small set of valid rendered goals; and that goal set with a three-step
arrival-tail cost. The last two are **explicit DMC adaptations, not paper reproductions**.
They remain opt-in: production `jepa_model.goal.alternatives` is empty, and `objective=tail`
is not a default. Scenario presets supply alternative cart positions, Reacher elbow branches,
or cup positions. The tail uses one consistent goal across its steps; it encourages staying
near the goal but does not prove zero velocity or sustained task success.

Learned and true-simulator-future candidate rankings use the exact same cost helper. A separate
256-update native-loss fitting control on disposable copies separates weak dynamics from weak
goal geometry, with disjoint fitting/validation simulator seeds. Physical heads never score
plans or supply inputs. Short 100-step intervention rollouts and `COMPLETE` report execution,
not a repaired policy. Budgets are adjustable with `--expert-updates`, `--fit-updates`,
`--policy-steps`, and `--horizon`; no broad dataset audit or old planner sweep is repeated.

Historical endpoint-only objective, horizon and optimizer diagnostics retain `objective=last`
explicitly so their scores do not silently change. Use this paired test for the new MPC/goal-set
comparison. Local checks: `MUJOCO_GL=egl python -m scripts.check_planner_fixes`.

### Native Planner Learning Curve

```bash
bash scripts/run_planner_learning_curve.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision
```

Trains each small Cartpole planner **once from scratch for 5,000 offline updates**, measuring
at 1,000, 3,000 and 5,000. LeWM's cosine schedule spans all 5,000 updates; TS keeps its native
constant rates. Measurements preserve training RNG, model buffers and planner caches. Only
native objectives are used: LeWM terminal distance and TS upstream MPC weighting, with one
physical-rendered goal and no physical-head planning.

Each model then continues from the same weights and optimizer moments for **4,096 environment
steps** in fresh online replay: 262 updates under the unchanged 80,000-step/10,000-update
production schedule, including the usual warmup. This is not a compressed online schedule.
The auxiliary head retains its normal independent expert mixture. No checkpoint is read or
written. Both models run sequentially; this is still a small-model diagnostic, not convergence
training or a full-size benchmark.

All four measurements reuse 16 held-out expert windows and 12 controlled simulator starts.
Reports include one-step/recursive latent errors, persistence baselines, latent spread,
action-candidate rankings, short policy returns, and original-unit physical-state RMSE.
Validation data never enter training. Raw latent MSE can change with encoder scale and is
not directly comparable across models. `COMPLETE` means execution, not successful learning.

Outputs are `runs/planner_learning_curve_<timestamp>/report.json`, `summary.txt`, per-model
`offline_metrics.jsonl`/`online_metrics.jsonl`, and each snapshot's `policy_metrics.jsonl`.
Progress is throttled; the final summary contains every measurement. No parity checks,
planner sweeps, predictor-copy fitting, timing calibration or dataset audit are repeated.
Use `--online-steps 0` for offline only, `--models leworldmodel` for one model, or
`--eval-updates 1000 3000 10000` to change the measurement points and total offline budget.
Local checks: `MUJOCO_GL=egl python -m scripts.check_planner_learning_curve`.

For a longer online continuation without changing the models, offline training budget,
or native online schedule:

```bash
bash scripts/run_planner_learning_curve.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision \
  --eval-updates 5000 \
  --online-steps 32000 \
  --online-eval-steps 4096 16000
```

This skips the earlier offline measurements, not their training updates. It measures at
the offline endpoint, at 4,096 and 16,000 online environment steps, and automatically at
32,000 steps: 262, 1,789 and 3,842 online updates respectively. These are cumulative steps
across 16 environments, including action repeats; the last point covers two full episodes
per environment. Intermediate evaluations reuse the same held-out cases without resetting
training replay, optimizer moments, RNG or planner caches. The report includes actual step
counts and completed training episodes. Policy evaluations remain the same short trials,
not full-episode benchmark scores. There are still no checkpoint reads/writes or extra sweeps.

### Physical Controller Comparison

```bash
bash scripts/run_physical_controller_check.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision
```

Diagnostic-only Cartpole comparison of image-goal planning and physical arrival costs.
Each tiny LeWorldModel/TS model is trained once for 5,000 offline updates, then continues
for 4,096 environment steps under the original online schedule (262 updates). At both
endpoints, controllers use identical frozen native weights, starts, horizons, solvers,
planning budgets and per-call random seeds. Native online collection always uses image
planning, so this tests scoring, not learning with the proposed physical controller.

A separate fresh controller head is fitted for 2,000 updates at each endpoint. It uses
the existing position/orientation/velocity targets, detached three-frame histories, and
equal observed/imagined clips. Only the head changes: native training and the auxiliary
evaluation head remain untouched. Supplemental zero/random simulator episodes and expert
TRAIN episodes are collected/read once and shared across models; controller validation
episodes and policy starts are disjoint. Controller-validation expert episodes may have been
seen during native pretraining, but never head fitting; benchmark held-out windows are a
further evaluation-only check. Training supervision is therefore not matched across controllers.
The extra supervised controller is explicitly an adaptation, not a paper reproduction.

The fixed cost averages cart-boundary error and upright-pole chord error along the horizon,
then adds a last-three-step velocity penalty. Cart limit is 0.25 m, angular scale is
acos(0.995), velocity scales are 0.5 m/s and 1 rad/s, and velocity weight is 0.1.
These are declared diagnostic choices, not tuned or guaranteed stabilization settings.
No goal-image term, reward head, physical-state input, native loss modification, or
checkpoint read/write is added. Physical forecasts can still be wrong or exploited.

`summary.txt` compares paired short policy returns/tail success and candidate choices
using forecast-decoded states, real-image-decoded states, and true simulator states.
`report.json` adds per-coordinate physical RMSE/false goals by source, exact training and
validation clips, hashes, budgets and timings. Per-step controller/policy/native metrics
are JSONL files. `COMPLETE` means execution, not improvement. No old sweeps or audits run.
Use `--online-steps 0` for the offline endpoint only; `--models temporal_straightening`
restricts the test to TS. Outputs default to `runs/physical_controller_<timestamp>`.
Local checks: `MUJOCO_GL=egl python -m scripts.check_physical_controller`.

### Four-Hour Learning Curve

```bash
bash scripts/run_physical_learning_curve.sh \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision
```

Fixed-budget extension of the physical-controller test, estimated at about four hours
**total for both models**, roughly two hours offline and two online including measurements.
LeWorldModel runs offline then online, followed by TS offline then online. Models remain
the same tiny sizes; no architecture, loss, controller-fit budget or planner change is made.
No calibration, timed stopping, old diagnostic sweep or extra dataset audit runs.

| Phase | Budget per model | Measurements |
|---|---|---|
| Offline | 24,000 updates | 1,000, 6,000, 12,000, 18,000, 24,000 updates |
| Online | 20,000 updates / 157,952 environment steps | 4,096, 41,024, 80,000, 118,976, 157,952 environment steps |

The online measurements correspond to 262, 5,000, 10,000, 15,000 and 20,000 updates.
Environment steps include action repeats and are summed across environments. Doubling
the post-warmup data and update budgets preserves the original data/update ratio and
warmup data requirement. Both schedulers span their entire requested phase, without
restarting at measurements. This is an extended diagnostic schedule, not the original
80,000-step production budget. Native image planning still collects the online data;
both controllers are compared on frozen weights at every measurement.

The estimates use the downloaded `planner_online_long_20260921_045640` and
`physical_controller_20260921_071819` timings: about 0.27 seconds per offline update
summed across models, about 0.33 seconds per online update including collection, and
about 140 seconds per paired-model measurement. Including pool setup and ten measurement
points gives roughly 2.1 hours per phase. Allow for I/O and machine-load variation;
the script uses fixed steps, not a four-hour deadline. These estimates apply to defaults.

Outputs: `runs/physical_learning_curve_<timestamp>/summary.txt`, `report.json`,
`learning_curve.csv`, and full JSONL training/controller/policy traces. CSV rows include
native/physical returns, sustained success, and held-out observed/forecast physical RMSE.
At each measurement, `native.pt` saves native weights, optimizer state and config;
`controller.pt` saves the separate fitted controller and its cost settings. These are
reusable diagnostic snapshots, **not exact online-resume checkpoints**: training replay
and live simulator state are not stored. The short test still defaults to no checkpoints.

### True-State Closed-Loop Controller

```bash
bash scripts/run_physical_oracle_check.sh \
  --source-report runs/physical_learning_curve_20260921_170135/report.json
```

Tests the two physical controllers using **true simulator states and futures**, with
the cost, horizon, solver budgets, warm starts, per-call seeds and controlled starts
from the saved report. Only this report is needed: no dataset, checkpoint loading,
head fitting, training, calibration or previous diagnostic sweep runs. The default
is the source test's 100 agent steps and 12 cases, once per solver, not every snapshot.
No production model or objective changes are made.

The existing LeWM CEM and TS Adam/tanh action optimizers are reused. MuJoCo is not
autograd-differentiable: TS gets central finite-difference simulator-state derivatives
(action epsilon 0.001), composed with the existing physical cost's analytic gradient.
This is an approximate-gradient oracle, not a bitwise-equivalent neural rollout.
Private physics copies restore full integration state including warmstarts; the
physics-only path is tested against the normal environment wrapper. See
[MuJoCo state handling](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#integration-state).

MuJoCo rollouts run on CPU. The default `--device cuda:0` keeps the source planner's
tensor/RNG backend; `--device cpu` is also supported but changes random draws.
This is a control-quality isolation, not a speed comparison. `--models leworldmodel`
or `--models temporal_straightening` selects a solver; `--policy-steps` can shorten
debug runs but then returns are not directly comparable with the saved longer tests.

To isolate lookahead, compare horizons 5, 25 and 50 (0.1, 0.5 and 1.0 simulator
seconds with this source report), keeping the physical cost and solver budgets fixed:

```bash
bash scripts/run_physical_oracle_check.sh \
  --source-report runs/physical_learning_curve_20260921_170135/report.json \
  --horizons 5 25 50
```

Each solver/horizon gets fresh environments and planner caches; warm starts are
used only between actions within a trial. Per-call seeds match, but different
tensor shapes mean random candidates are not identical across horizons. The cost's
velocity term still uses the last three predicted steps, not a rescaled tail.
Longer horizons increase both lookahead and optimization difficulty; a negative
result does not rule out a stronger optimizer solving the same cost.

**Runtime:** CEM simulator work grows roughly linearly with horizon, while TS's
finite-difference simulator gradients grow roughly quadratically. From the previous
two-minute H=5 timings, the full sweep may take several hours, mostly TS at H=50.
For a quicker first answer, append `--models leworldmodel` (roughly half an hour),
then run `--models temporal_straightening` separately if needed. These are rough
estimates, not neural-model training times. No dataset audits or training run.

Outputs: `runs/physical_oracle_<timestamp>/summary.txt`, `report.json`, and one
`<model>/horizon_<H>/policy_metrics.jsonl` per trial with every action, true state,
reward and success. The report records each trace path and saves completed trials
as the sweep progresses. A failed trial does not discard other results.
The summary shows each horizon and the saved offline/online physical-controller
endpoints; those saved learned baselines use the source horizon only, not the
longer oracle horizons.
`COMPLETE` means execution, not balancing success. If the true-state controller fails,
cost/horizon/solver limitations remain unresolved; if it works, learned state/dynamics
errors become the next target. Native image-goal planning is not evaluated by this test.

Local contracts: `python -m scripts.check_physical_oracle`.

### Feedback Reference and Planner Cost Comparison

When the true-state planner fails, compare it with a simulator-feedback positive
control before another training run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
python -m scripts.diagnose_controller_reference \
  --source-report runs/physical_learning_curve_20260921_170135/report.json \
  --output runs/controller_reference --device cpu --horizons 5 25 50
```

This evaluates a bounded local feedback controller on the source report's saved
starts for its original policy length. At each fresh initial anchor, it then
scores feedback-generated and solver-selected action sequences using the same
true futures and physical cost. CEM uses its executed elite-mean sequence; the
gradient planner uses its selected final restart. Each solver plans only once
per horizon, so this is shorter than repeating the closed-loop horizon sweep.
The default horizon list is just `5`.

A lower-cost feedback sequence identifies a missed feasible plan at that anchor.
The opposite ordering shows that the tested cost can prefer a different sequence;
it does not establish global optimality or explain an entire closed-loop failure.
The feedback baseline uses privileged simulator state and a different control
objective, so it is a diagnostic reference, not an image-model benchmark result.
CPU/CUDA candidate draws differ even with the same seed.

Outputs include `report.json`, `summary.txt`, feedback traces, complete compared
sequences, source hashes, and package versions. No images, checkpoints, training,
or dataset samples are used. The output directory must be new.
Local contracts: `python -m scripts.check_controller_reference`.

### Short Online Check From Expert Checkpoints

Before repeating long online runs, test the Cartpole LeWorldModel and Temporal Straightening
expert checkpoints using their original model sizes, batches, planner budgets, precision, and schedules:

```bash
bash scripts/run_online_checkpoint_smoke.sh \
  --run-root runs/dmc_vision_10k \
  --dataset-root /home/ubuntu/DMC/data/dmc_expert_vision
```

By default, each model runs three independent branches from the same expert checkpoint:
native online-only training, 50/50 native expert/online training, and that same mixture with
256 additional head-only calibration updates. Each branch collects 4,096 environment steps into
empty on-policy replay and performs 262 native updates under the production schedule. It skips
expert pretraining and checkpoint writes. The original 10,000-update learning-rate schedule
is preserved, not shortened to the smoke budget. `--env-steps` changes only the stopping point;
`--models` and `--scenarios` select other LeWorldModel/TS cases. Completed expert checkpoints and
matching held-out datasets are required; incompatible or online checkpoints are rejected.

Use `--native-expert-fractions 0 0.5`, `--calibration-updates 256`, and `--policy-episodes 1`
to set these budgets explicitly. The original checkpoint is read-only. Fixed expert windows and
complete pretrained-policy, zero-action, and random-action validation episodes are collected once
per model, then reused across branches. A complete policy episode on the same independent seed
also runs after each branch. These few episodes diagnose behavior, not reliable policy rankings.

Calibration uses separate zero/random training episodes and training-split expert labels, never
validation data. Native features and BN statistics stay frozen during calibration. Physical scales
are fixed from these training states with floors of 0.1 m for positions, 1 for velocities, and unit
scale for trigonometric coordinates; both output and loss conditioning use those scales. The
initial affine migration preserves predictions, while the original expert evaluation std stays fixed.
This is an **unvalidated diagnostic candidate**, not a new production default. Extra head updates
are reported separately; native batches, source-episode counts, losses, and optimizer counts do not grow.

Prediction-preserving checkpoint migration is checked before training. Expert and fixed simulator windows are
scored before, every 64 updates, and after, with context 64 and horizons 1/100, without fitting. Reports include
original-unit RMSE, unchanged expert-normalized nMSE, representation diagnostics, online losses,
and the source checkpoint identity. A timestamped `runs/online_checkpoint_smoke_*` directory holds
`summary.txt`, `report.json`, and per-worker logs/metrics; `--output` must name a new directory.
PASS includes observed/forecast regression and false-goal guards, not proof that long-run degradation is solved.
REGRESSION distinguishes finite execution with degraded predictions from an execution failure.
By default each observed/forecast coordinate RMSE must stay below `3 * max(initial RMSE, 0.01)` in original units,
and mean nMSE below `9 * max(initial nMSE, 1)`, at every diagnostic snapshot. These are explicit
smoke alarms, not significance tests or model-selection criteria. `--rmse-floors` accepts a JSON map
of coordinate-specific original-unit floors. `UNVALIDATED` means no validation failure states were
available, or the observed head falsely predicted success on more than `--max-false-success-rate`
(default 20%) of actual failures in any sampled cohort. Goal-relation errors and failure counts are logged.
Diagnostic windows never supply gradients,
and the declared online schedule runs to its stopping point even if a diagnostic flags regression.
There is no repeated serial/concurrent benchmark or full dataset quality audit.

Run `python -m scripts.check_online_checkpoint_smoke` for CPU regression checks using tiny models
and simulated observations; the full-size CUDA/environment check must run on the training server.
`python -m scripts.check_online_repairs` checks gradient invariance, replay budgets/resumption,
frozen-native calibration, and disjoint validation end to end. For a minimal execution-only check,
use `--native-expert-fractions 0 --calibration-updates 0 --policy-episodes 0`; this cannot validate failure-state quality.

Native retention is opt-in via `training.online.expert_fraction` (default 0). For LeWorldModel/TS,
0.5 means 168 expert plus 168 online sequences, from eight episodes each, in one native forward
including BatchNorm. The head independently receives its usual 128 expert plus 128 online labels,
not an accidental 75/25 mix. Both expert sampler states are saved for exact sampling resumption.
The comparison validator rejects silently mixing native-retention protocols within one result matrix.

## Training Budget

Every family receives 10,000 expert updates and 10,000 online
updates, sampling from 16 source episodes in each world-model or representation update. Dreamer and
STORM use one 64-frame sequence per source episode (batch 16); TD-MPC2, LeWorldModel, and Temporal
Straightening use twenty-one native four-frame clips per source episode (batch 336). Online
replay has a shared rolling capacity of 20,000 transitions. Updates require both the 1,024-transition
warmup and 16 usable source episodes; the scheduler then catches up to the update budget.
LeWorldModel and Temporal Straightening collect
with their own planners and continue their unchanged representation objectives; they receive no actor,
critic, reward, or behavior-cloning loss. Tasks, action spaces, adjacent-state target counts, update
counts, and replay capacity are matched; family-specific objectives and planners remain separate.
Dreamer and STORM also use 512 starting states for each imagined controller update; their native ways
of obtaining those starts remain different.

Each update has 1,008 adjacent-state targets. Dreamer additionally trains 16 initial-state targets,
for 1,024 total RSSM state targets; none are masked out. Dreamer and STORM sample 1,024 observations
per update, while the short-horizon planning models sample 1,344. It is not an equal-compute budget.
Training logs and evaluation JSON
record both observation and dynamics-target counts. Use a fresh training output directory for this
recipe; existing collected datasets remain compatible.

## Models

### Actor Objective Regression Checks

Run `python -m scripts.check_policy_objectives` for CPU checks of saturated-action
scores/gradients, STORM return targets, expert-to-online updates, and checkpoint compatibility.
No dataset, environment rollout, or checkpoint writes are required.

Dreamer and STORM retain pre-tanh samples for online actor scoring; STORM also bootstraps
transition returns from the next-state value (recipe 3 corrections).

Training recipe 7 adds fixed-unit fresh-head initialization and finite TS native gradient clipping
to recipe 6's planner/readout repairs. Recipe 8 additionally isolates physical readout from control
and supplies physically rendered latent goals. Recipe 9 fixes TS's `cos` curvature reduction:
compute directions per visual patch before averaging, as upstream does, instead of flattening
patches into a single motion vector. Static-patch masking and the finite empty-mask guard remain.
This corrects loss/gradient weighting; it does not establish that the observed online degradation
is repaired. Old weights still load for readout-only diagnostics;
their original policy returns require the original controller/code. Neither old expert nor old online
checkpoints silently resume the new production recipe. To test the new controller with existing
pretrained weights, `run_online_checkpoint_smoke.sh --latent-goals ...` explicitly installs the current
scenario's goal specification and records the override. This diagnostic never writes checkpoints or
claims that the loaded weights were pretrained under the current recipe. Use `--native-expert-fractions 0
--calibration-updates 0` for a single native-training trial without extra head calibration.
Keep corrected training results in a separate output directory; datasets remain reusable.

Run `MUJOCO_GL=egl python -m scripts.check_physical_goals` for CPU head-isolation tests and actual
DMC rendering/parallel collection/reset/update/evaluation checks. These are standalone tests, not
additional work inside the production training loop.

Run `python -m scripts.check_state_normalization` for CPU checks of loss conditioning, legacy
prediction/gradient preservation, optimizer migration, and all five families' checkpoint paths.

Run `python -m scripts.check_training_contracts` for small-model Dreamer/STORM/TD-MPC2 checks of
native optimizer coverage, actual component updates, target EMAs, physical-label isolation,
checkpoint continuation, and recurrent output/gradient equivalence. Mamba3 update checks use CUDA
when available and otherwise report skips. No data files, simulator, or checkpoint writes are needed;
these checks are not part of the training launch path and do not establish long-run learning stability.

The five Dreamer variants share their convolutional encoder/decoder, posterior, prior, and losses;
only the deterministic sequence core changes. The five STORM variants share their
Conv-BatchNorm-ReLU encoder, transposed-convolution decoder, observation-only posterior, and training
losses; only the sequence core changes. Dreamer sliding attention streams through the episode with
a 64-step rolling KV cache. The default STORM Transformer retains its fixed 16-step policy context;
STORM sliding attention streams through the whole episode with a 64-token KV window. Mamba3 and
S5 carry fixed-size recurrent states until the episode ends. Hyena trains with causal FFT
convolutions and streams with an exact rolling 64-token filter history.

The learning/control budget is centered on 5.25M parameters without inactive padding. It includes
encoders, dynamics, prediction heads, controllers, and decoders whose losses train the representation.
Temporal Straightening's detached visualization decoder is optional and reported separately, as are
frozen target-network copies. Neither counts toward the matched budget. The remaining same-task
spread comes mostly from action-conditioned modules, including TD-MPC2's five Q networks.

Run the complete image-model and scenario matrix after changing any architecture setting:

```bash
python3 -m scripts.model_size_report
```

The standalone report breaks each model into encoder, dynamics, decoder, prediction heads, and
controller, plus the physical-state head, auxiliary weights, and frozen copies. The detached head is
auxiliary for every family; it no longer counts toward the two goal planners' learning/control budget.
It checks component proportions
against each family's reference implementation, using the scratch-ResNet variant for Temporal
Straightening. It fails if a component's relative share differs by more than 10%, a same-task budget
spread exceeds 50K parameters, a recurrent pair gap exceeds 2K, or a model moves more than 50K away
from the shared target. These size checks do not run inside training.

Native architectures are unchanged by the controller correction. TS now has approximately 5.196M
learning/control parameters, plus its separately reported approximately 53K physical head. This is
about 54K below the 5.25M target, with a maximum same-task spread of 57,273 parameters. The existing
strict 50K target/spread checks flag this; they have not been relaxed and the model has not been
resized to hide the accounting change.
Its six predictor layers retain roughly
equal attention and feedforward parameter allocations. Visualization is disabled by default;
`jepa_model.decoder.enabled=true` adds 1,526,355 separately reported parameters. Its reconstruction
losses still receive detached latents and do not train the encoder or predictor.

The Dreamer and STORM variants use the shared architecture budget in `configs/dmc_model.yaml`, while
retaining their family-specific input transforms, normalization, and activation functions. Dreamer's prior,
posterior, and recurrent settings live in `configs/model/_base_.yaml`; the corresponding STORM
settings live in `configs/storm_dmc.yaml`. Their implementations remain separate under
`models/dreamer` and `models/storm`.

- [TD-MPC2][tdmpc2] keeps its three-frame pixel stack, encoder, random-shift augmentation, SimNorm latent model,
  distributional reward and five-critic losses, Gaussian policy prior, target critics, and MPPI planner.
- [LeWorldModel][leworldmodel] keeps its train-from-scratch image ViT, three-frame autoregressive prediction, AdaLN-zero
  action conditioning, learned projectors, SIGReg, and CEM planner.
- [Temporal Straightening][temporal-straightening] keeps its visual encoder,
  causal action-conditioned prediction, stop-gradient targets, visual cosine-curvature objective,
  and gradient-based action planning. Its detached reconstruction decoder remains available for
  visualization but is disabled in the comparison runs. Removing the proprioceptive branch is a
  deliberate deviation; the remaining visual prediction term retains its previous effective weight.

The goal planners use latent distance to a separately rendered physical goal: terminal-only for
LeWM and upstream weighted MPC scoring for new TS configurations. During action
optimization native weights are fixed, but gradients pass through predicted latents to candidate
actions; no physical decoder is involved. Planning and evaluation share the same autoregressive rollout implementation.
Temporal Straightening computes action gradients in batches of at most
`jepa_model.planner.gradient_batch_size=256` candidate trajectories. This bounds planning memory
without changing the number of environments, restarts, iterations, or future steps, and preserves
independent candidate gradients: sum candidate costs, never divide them by environment/restart count.
The batch size is an execution setting, not a checkpoint recipe.

Dreamer, STORM, and the goal planners use BF16 mixed precision for CUDA neural computation, including
Dreamer's history reconstruction and acting. TD-MPC2 defaults to FP32 after BF16 slowed warm updates
in the A100 benchmark; `tdmpc2_model.use_amp=true` remains available for experiments. Parameters and
optimizer state remain FP32; curvature, SIGReg,
physical readouts, goal costs, TD targets, and action optimization remain FP32. S5 retains complex64
dynamics and Hyena retains FP32 FFTs. CPU runs use FP32. Disable AMP with `model.use_amp=false`
(Dreamer), `jepa_model.use_amp=false` (TS/LeWorldModel), or `tdmpc2_model.use_amp=false` (TD-MPC2).
STORM retains its existing `storm_model.use_amp` and `actor_critic.use_amp` switches. Precision is part
of the run identity; these settings are not intended to silently resume an older-precision experiment.

The shared trainer does not impose one optimizer or update rule on every family. Dreamer keeps its
joint world-model/actor-critic update; STORM keeps one separate world-model and imagined actor-critic
update per shared update; TD-MPC2 keeps its joint latent-model/Q update,
separate policy update, and soft target-Q update; LeWorldModel keeps AdamW and its prediction plus
SIGReg objective; and Temporal Straightening keeps separate Adam/AdamW optimizers for its encoder,
predictor, action encoder, and optional visualization decoder. The common fixed-update expert phase,
64x64 images, and matched parameter budget are deliberate comparison adaptations rather than claims
about the original paper defaults. Sequence lengths remain family-specific where the reference recipes
differ, so batch sizes equalize adjacent-state targets while source-episode diversity is held constant.

The production configs use the predeclared `dmc_frozen_defaults_v20` hyperparameter protocol. Recurrent
variants inherit one unchanged family recipe: Dreamer uses LaProp at `4e-5`, batch size 16, and AGC
0.3; STORM uses Adam at `1e-4` for the world model and `3e-5` for the actor-critic, batch size 16, and
its reference gradient limits. TD-MPC2 uses its `3e-4` reference learning rate,
while LeWorldModel and Temporal Straightening retain their native optimizer separation. Core widths are
set only by the shared parameter budget. These defaults must be frozen before production runs and must
not be adjusted for individual tasks or variants after observing results.

Dreamer, STORM, and TD-MPC2 share `reward_discount: 0.99` for expert value targets, online returns,
and reward-based planning. This sets an effective reward horizon of 100 agent steps, or 200 physics
steps with the shared action repeat of two. Override `reward_discount` to change all three together;
family-specific discounts are rejected by the recipe check. Imagination and planning rollout lengths
remain model-specific. LeWorldModel and Temporal Straightening do not optimize discounted rewards.

All Dreamer variants use a tanh-squashed Gaussian actor instead of clipping Gaussian samples.
Behavior cloning and imagined policy updates use its transformed log probability; entropy includes
the tanh correction. This is a deliberate policy adaptation, with unchanged parameter counts and
standard-deviation settings. Existing Gaussian-policy Dreamer checkpoints require a fresh training run;
the expert datasets remain reusable.

STORM's expert critic uses the same bootstrapped lambda returns, symlog two-hot loss, and slow-critic
regularization as its online critic; only the expert actor uses behavior cloning. Truncated sequences
bootstrap rather than treating time limits as terminals. Its continuous-policy entropy includes the
tanh transformation. AMP overflows retry only the affected update (at most 32 attempts); update
budgets, schedulers, target networks, and auxiliary heads advance only after a successful optimizer
step. These recipe and target changes require fresh v20 training checkpoints, but collected datasets are reusable.

For easier code reading, inline tensor shape annotations are provided. See
[`docs/tensor_shapes.md`](docs/tensor_shapes.md).

## Headless rendering

If you run DMC on a headless machine, set `MUJOCO_GL` for offscreen rendering. **Using EGL is recommended** as it accelerates rendering and simulation throughput.

```bash
# For example, when using EGL (GPU)
export MUJOCO_GL=egl
# (optional) Choose which GPU EGL uses
export MUJOCO_EGL_DEVICE_ID=0
```

More details: [Working with MuJoCo-based environments](https://docs.pytorch.org/rl/stable/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)

## Code formatting

If you want automatic formatting/basic checks before commits, you can enable `pre-commit`:

```bash
pip install pre-commit
# This sets up a pre-commit hook so that checks are run every time you commit
pre-commit install
# Manual pre-commit run on all files
pre-commit run --all-files
```

## Citation

If you find this code useful, please consider citing:

```bibtex
@inproceedings{
morihira2026rdreamer,
title={R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation},
author={Naoki Morihira and Amal Nahar and Kartik Bharadwaj and Yasuhiro Kato and Akinobu Hayashi and Tatsuya Harada},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Je2QqXrcQq}
}
```

[r2dreamer]: https://openreview.net/forum?id=Je2QqXrcQq&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)
[tdmpc2]: https://github.com/nicklashansen/tdmpc2
[leworldmodel]: https://github.com/lucas-maes/le-wm
[temporal-straightening]: https://github.com/Agentic-Learning-AI-Lab/temporal-straightening
