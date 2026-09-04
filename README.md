# DMC World Model Comparisons

This comparison framework is based on [R2-Dreamer][r2dreamer]. It trains capacity-matched Dreamer,
STORM, TD-MPC2, LeWorldModel, and Temporal Straightening models on DeepMind Control Suite tasks with
64x64 image observations and optional expert pretraining.

## Instructions

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

### DMC expert experiments

Expert datasets use the `dmc_expert_hdf5_dense_v1` layout: each dataset directory contains
`metadata.json`, `data.hdf5`, and `progress.json`. The primary experiments use 64x64 RGB observations
on Point Mass Easy, Reacher Easy, and Ball-in-Cup Catch. Set the local TD-MPC2 checkout
and image-dataset root before collecting or training:

```bash
export TDMPC2_DIR=/absolute/path/to/tdmpc2
export DMC_EXPERT_VISION_DATA_DIR=/absolute/path/to/data/dmc_expert_vision
export DMC_EXPERT_EVAL_DATA_DIR=/absolute/path/to/data/dmc_expert_eval_vision
export TDMPC2_POINT_MASS_CHECKPOINT=/absolute/path/to/pointmass-easy.pt

python3 -m scripts.collect_dmc_expert_data --config-name dmc_expert_collection

python3 train.py --config-name offline_dmc_expert_gru_vision scenario=point_mass
python3 train.py --config-name offline_dmc_expert_sliding_window_vision scenario=point_mass
python3 train.py --config-name offline_dmc_expert_mamba3_vision scenario=point_mass
python3 train.py --config-name offline_dmc_expert_s5_vision scenario=point_mass
python3 train.py --config-name offline_dmc_expert_hyena_vision scenario=point_mass
python3 train.py --config-name storm_dmc_transformer_vision scenario=point_mass
python3 train.py --config-name storm_dmc_sliding_window_vision scenario=point_mass
python3 train.py --config-name storm_dmc_mamba_vision scenario=point_mass
python3 train.py --config-name storm_dmc_s5_vision scenario=point_mass
python3 train.py --config-name storm_dmc_hyena_vision scenario=point_mass
python3 train.py --config-name leworldmodel_dmc_vision scenario=point_mass
python3 train.py --config-name temporal_straightening_dmc_vision scenario=point_mass
python3 train.py --config-name tdmpc2_dmc_vision scenario=point_mass
```

TD-MPC2 publishes compatible checkpoints for Reacher and Ball-in-Cup. Its published checkpoint set
does not include Point Mass, so that task requires a locally trained 5M TD-MPC2 state checkpoint
exposed through `TDMPC2_POINT_MASS_CHECKPOINT`.

Run the complete collection, training, and evaluation matrix from one config:

```bash
./scripts/run_full.sh --dry-run
./scripts/run_full.sh
```

Before committing to the full datasets and training budget, run the same lifecycle with one tiny
Ball-in-Cup dataset and one update per model:

```bash
./scripts/run_smoke.sh --dry-run
./scripts/run_smoke.sh
```

Both wrappers use the `environment/` created by `scripts/setup_dmc.sh`. Set `PYTHON` to use another
interpreter. They forward any additional arguments to `main.py`, whose configs can also be invoked
directly with `python3 main.py --config-name dmc_smoke` or `dmc_benchmark`.

The orchestrator keeps the terminal focused on stage progress, timing, training metrics, and final
results. Full commands, dependency warnings, and raw tracebacks remain in each collection or run log;
on failure, the useful end of the traceback and the exact log path are printed automatically.

The default matrix runs all thirteen image-model variants on all three scenarios. For each scenario,
it creates the training and held-out evaluation datasets, then trains and evaluates every model before
moving to the next scenario. Training writes each run under
`runs/dmc_vision/<scenario>/<family>/<variant>/seed_<seed>`, and evaluation writes
`evaluation.json` beside each checkpoint. Set the booleans under `stages` to run only part of the
lifecycle, add values under `seeds` for independent replicates, and select the compute device with
`device`. With `training.resume: true`, rerunning the same command skips completed evaluations,
evaluates finished training runs, and resumes interrupted expert or online training. Set
`training.overwrite: true` to delete existing runs and start them again.

Run each reported experiment with independent training seeds. Log directories include the seed, so a
Hydra multirun does not mix checkpoints:

```bash
python3 train.py --config-name offline_dmc_expert_gru_vision --multirun \
  scenario=point_mass seed=0,1,2,3,4
```

Every training run names its config explicitly. Resume a checkpoint through the same entrypoint;
`training.online.steps` remains the target total number of environment steps:

```bash
python3 train.py --config-name offline_dmc_expert_mamba3_vision \
  scenario=point_mass \
  resume_from=/absolute/path/to/latest.pt \
  training.online.steps=80000

tensorboard --logdir ./logdir
```

Checkpoint names identify their phase: `pretrain_latest.pt` resumes interrupted expert training,
`pretrained.pt` is the completed expert-pretraining state, `pretrained_best.pt` is its best
validation-return state, `latest.pt` resumes online training, `best.pt` is the best online
validation-return state, and `final.pt` is the completed online state. Checkpoints and evaluation
JSON are written atomically so an interrupted write does not replace the previous valid file. Final
evaluation selects one exact checkpoint name from the model config and never falls back across phases.

The image models keep each family's visual design rather than routing pixels through shared state
MLPs:

- Dreamer uses its convolutional encoder and spatial convolutional decoder.
- STORM uses its Conv-BatchNorm-ReLU encoder and transposed-convolution decoder with Transformer,
  sliding-window attention, Mamba3, S5, or Hyena sequence dynamics.
- LeWorldModel uses a compact train-from-scratch ViT and its decoder-free latent objective.
- Temporal Straightening combines spatial ResNet tokens with its native proprioceptive embedding and
  reconstructs the visual tokens with a convolutional decoder.
- TD-MPC2 uses its four-layer pixel encoder and random-shift augmentation.

At the default 10,000 episodes, each RGB array is about 57 GiB before HDF5 compression. Collection
also stores the simulator state for the held-out physical-state probe. Temporal Straightening alone
receives its configured four-value proprioceptive subset; target-relative state remains excluded.

Collection stores a two-coordinate task relation beside each image: mass-to-target for Point Mass,
finger-to-target for Reacher, and ball-to-moving-cup-target for Ball-in-Cup. LeWorldModel and
Temporal Straightening train a detached readout for this label; it does not alter their encoder or
predictor objective. Collect a separate held-out set for physical-state prediction metrics:

```bash
export DMC_EXPERT_EVAL_DATA_DIR=/absolute/path/to/data/dmc_expert_eval_vision
python3 -m scripts.collect_dmc_expert_data \
  --config-name dmc_expert_collection_eval_vision
```

Evaluate an image-model checkpoint on that held-out dataset. Every family reports fresh policy
return and task success as well as physical-state prediction. LeWorldModel and Temporal
Straightening remain reward-free during representation training; reward only selects checkpoints.
Their planners minimize the fixed DMC success geometry predicted from latent state, with no goal
image supplied at evaluation time:

```bash
python3 -m scripts.evaluate_dmc \
  --config-name leworldmodel_dmc_vision \
  --scenario point_mass \
  --logdir /absolute/path/to/the/run \
  --dataset "$DMC_EXPERT_EVAL_DATA_DIR/point_mass_easy"
```

The image cohort is the comparison. Each family retains its native input branches, visual frontend,
objective, planner, and update recipe, while tasks, action spaces, datasets, and total capacity are
matched.

All experiment configs enter through `train.py`. The shared lifecycle in `training/trainer.py`
handles pretraining, online scheduling, evaluation, logging, checkpoints, and resume state.
Each module under `training/` contains only its family-specific construction and update hooks.

The `scenario` selection keeps each DMC task, dataset directory, action shape, success geometry, and
planning horizon together. Active benchmark values are `point_mass`, `reacher`, and `ball_in_cup`.

Only the Mamba3 configs require the Mamba runtime, but the setup script installs one shared
environment so every comparison model runs against the same PyTorch and DMC dependencies.

The first five commands use the Dreamer training recipe and share the Dreamer encoder, posterior,
prior, decoders, and losses; only the deterministic sequence core changes. The next five commands
use the native STORM recipe and share the STORM encoder, observation-only posterior, and training
losses; only the sequence core changes. Dreamer sliding attention streams through the episode with
a 64-step rolling KV cache. The default STORM Transformer retains its fixed 16-step policy context;
STORM sliding attention streams through the whole episode with a 64-token KV window. Mamba3 and
S5 carry fixed-size recurrent states until the episode ends. Hyena trains with causal FFT
convolutions and streams with an exact rolling 64-token filter history.

Trainable capacity is centered on 5.25M parameters without inactive padding. Frozen target critics
and value networks are reported separately because they are optimizer state, not independently
trainable capacity. The remaining same-task spread comes mostly from action-conditioned modules,
including TD-MPC2's five Q networks.

Run the complete image-model and scenario matrix after changing any architecture setting:

```bash
python3 -m scripts.model_size_report
```

The report breaks each model into encoder, dynamics, decoder, prediction heads, and controller. It
checks each image model against the component proportions of its own reference implementation, then
fails if a component differs by more than 10%, a same-task spread exceeds 65K parameters, a recurrent
pair gap exceeds 2K, or a model moves more than 60K away from the shared target.

The Dreamer and STORM variants use the shared architecture budget in `configs/dmc_model.yaml`, while
retaining their family-specific input transforms, normalization, and activation functions. Dreamer's prior,
posterior, and recurrent settings live in `configs/model/_base_.yaml`; the corresponding STORM
settings live in `configs/storm_dmc.yaml`. Their implementations remain separate under
`models/dreamer` and `models/storm`.

The additional configs bring the comparison to five model families: Dreamer, STORM,
LeWorldModel, Temporal Straightening, and TD-MPC2. Dreamer has five sequence-core variants and STORM
has five, so there are thirteen configurations in total. Eleven have native reward-driven online control;
LeWorldModel and Temporal Straightening remain offline representation and goal-planning methods.

- [TD-MPC2][tdmpc2] keeps its pixel encoder, random-shift augmentation, SimNorm latent model,
  distributional reward and five-critic losses, Gaussian policy prior, target critics, and MPPI planner.
- [LeWorldModel][leworldmodel] keeps its image ViT, three-frame autoregressive prediction, AdaLN-zero
  action conditioning, learned projectors, SIGReg, and CEM planner.
- [Temporal Straightening][temporal-straightening] keeps its visual and proprioceptive encoders,
  decoder, causal action-conditioned prediction, stop-gradient targets, visual cosine-curvature
  objective, reconstruction, and gradient-based action planning.

LeWorldModel and Temporal Straightening are kept as goal-conditioned representation models. They do
not receive added reward or value heads, and reward-only online training is disabled for them. Their
learned latent dynamics can be compared with the same held-out physical-state prediction metric.
Their small task-relation readout is trained on detached latents and used only to expose each DMC
task's native success region to the planner.

The shared trainer does not impose one optimizer or update rule on every family. Dreamer keeps its
joint world-model/actor-critic update; STORM keeps separate world-model and imagined actor-critic
updates at one of each per collected transition; TD-MPC2 keeps its joint latent-model/Q update,
separate policy update, and soft target-Q update; LeWorldModel keeps AdamW and its prediction plus
SIGReg objective; and Temporal Straightening keeps separate Adam/AdamW optimizers for its encoder,
predictor, action encoder, and decoder. The common 16-epoch expert phase, 64x64 images, and matched
parameter budget are deliberate comparison adaptations rather than claims about the original paper
defaults. Replay batch, sequence, and online update settings remain
family-specific where the reference recipes differ.

All active STORM variants live under `models/storm`. Collection loads an external TD-MPC2
checkout from `TDMPC2_DIR`; neither training path imports an upstream reference checkout from this
repository. The retired state-input ablation, reference copies, old notebooks, tests, and prior
experiment artifacts live under `local/`, which is excluded from version control.

For easier code reading, inline tensor shape annotations are provided. See
[`docs/tensor_shapes.md`](docs/tensor_shapes.md).


## DMC environments

The active comparison targets DeepMind Control Suite with image observations.

| Environment | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous | 80K | DeepMind Control Suite with image inputs. |

Choose an image config explicitly. The `scenario` group keeps the task, dataset, and action shape
together.

```bash
python3 train.py --config-name offline_dmc_expert_gru_vision scenario=point_mass
```

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
