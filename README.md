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

Immediately before the production run, use the fuller preflight. It checks the installed GPU stack,
runs two expert updates and the short online lifecycle for every model, evaluates all thirteen variants,
and validates the resulting checkpoints and metrics:

```bash
./scripts/run_preflight.sh
```

The wrappers use the `environment/` created by `scripts/setup_dmc.sh`. Set `PYTHON` to use another
interpreter. The smoke and full-run wrappers forward additional arguments to `main.py`; all three
configs can also be invoked directly as `dmc_smoke`, `dmc_preflight`, or `dmc_benchmark`.

The orchestrator keeps the terminal focused on stage progress, timing, training metrics, and final
results. Full commands, dependency warnings, and raw tracebacks remain in each collection or run log;
on failure, the useful end of the traceback and the exact log path are printed automatically.

The default production matrix first runs all thirteen image-model variants on all three scenarios with
seed 0. It collects the three independent scenario datasets concurrently on the shared GPU, then trains
and evaluates every model one at a time, scenario by scenario. Set `collection.parallelism=1` to collect
serially. Training writes each run under
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
resume. Older checkpoints without compatibility metadata still require their original training settings.
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
for Cartpole, finger-to-target for Reacher, and ball-to-moving-cup-target for Ball-in-Cup. LeWorldModel and
Temporal Straightening derive these relations from the shared physical-state head for planning, rather
than learning duplicate goal outputs. All five families use the same targets within each scenario:

| Scenario | Learned physical targets | Outputs |
| --- | --- | --- |
| Cartpole | Cart position, pole cosine/sine, cart velocity, pole angular velocity | 5 |
| Reacher | Cosine/sine of each joint angle, fingertip-to-target vector, joint velocities | 8 |
| Ball-in-Cup | Cup/ball positions and velocities in the simulator's planar x/z coordinates | 8 |

Reacher angles are encoded on read, removing arbitrary full revolutions from the labels. Cartpole
already supplies cosine/sine. Velocities remain useful history-dependent targets, not quantities that
can be uniquely recovered from one frame. Pixel resolution and occlusion still limit observability.
Existing HDF5 observations and collection metadata are unchanged; no recollection is needed.

Its own Adam optimizer minimizes standardized state MSE on detached observed features, using 256
targets per model update. Training-split mean and standard deviation are fixed before pretraining and
saved with the head and its optimizer. Online-only runs use zero mean and unit scale. No physical-state,
reward, or planning loss from this head updates the native representation or dynamics. Online replay
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
`dmc_evaluation_v8` reports supplementary physical-state RMSE separately for each physical coordinate
at each prediction horizon, in its original units. It includes constant coordinates and does not divide
by dataset variance or average unrelated units into one score. JSON stores these values in
`physical_state_prediction.rmse["<horizon>"]["<coordinate>"]`, for example `"1"` and `"velocity[0]"`.
Reacher's angle coordinates are `cos(position[0])`, `sin(position[0])`, and the corresponding joint-1
pair. Cosine/sine are unitless; other positions and velocities retain metres, radians, and seconds.
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
validation return only selects checkpoints. Their planners minimize the fixed DMC success geometry
predicted from latent state, with no goal image supplied at evaluation time:

```bash
python3 -m scripts.evaluate_dmc \
  --config-name leworldmodel_dmc_vision \
  --scenario cartpole_balance_sparse \
  --logdir /absolute/path/to/the/run \
  --dataset "$DMC_EXPERT_VISION_DATA_DIR/cartpole_balance_sparse"
```

## Training Budget

Every family receives 5,000 expert updates and 10,000 online
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
auxiliary for Dreamer, STORM, and TD-MPC2; it is part of the controller budget for the two goal planners.
It checks component proportions
against each family's reference implementation, using the scratch-ResNet variant for Temporal
Straightening. It fails if a component's relative share differs by more than 10%, a same-task budget
spread exceeds 50K parameters, a recurrent pair gap exceeds 2K, or a model moves more than 50K away
from the shared target. These size checks do not run inside training.

Temporal Straightening has 5,248,804 to 5,249,009 learning/control parameters across the three scenarios:
about 19.6% encoder, 79.4% dynamics, and 1.0% physical readout. Its six predictor layers retain roughly
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

The goal planners use task-relation outputs of the detached physical readout. During action
optimization its weights are fixed, but gradients can pass through it and the dynamics to candidate
actions. Planning and evaluation share the same autoregressive rollout implementation.
Temporal Straightening computes action gradients in batches of at most
`jepa_model.planner.gradient_batch_size=32` candidate trajectories. This bounds planning memory
without changing the number of environments, restarts, iterations, or future steps, and preserves
the full-batch cost normalization. The batch size is an execution setting, not a checkpoint recipe.

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
