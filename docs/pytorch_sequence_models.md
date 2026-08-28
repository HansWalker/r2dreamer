# PyTorch Sequence Model Implementations

This document lists the sequence-modeling methods under consideration that
already have an open-source PyTorch implementation. "Official" means the code
was released by the paper's authors or their organization. Community ports are
identified separately.

| Method | PyTorch implementation | Source | Integration note |
| --- | --- | --- | --- |
| Sliding-window attention | [Longformer](https://github.com/allenai/longformer) | Official | Implemented as STORM's local causal-attention variant in `models/storm/cores/sliding_window.py`. |
| S4D | [S4](https://github.com/state-spaces/s4) | Official | The repository includes a standalone PyTorch S4D layer intended for use in other projects. |
| S5 | [s5-pytorch](https://github.com/i404788/s5-pytorch) | Community | Implemented locally with the paper's MIMO HiPPO initialization and an exact recurrent state. |
| Hyena | [Safari](https://github.com/HazyResearch/safari) | Official | Implemented locally with the official implicit filter, FFT sequence path, and a bounded recurrent history. |
| CKConv | [CKConv](https://github.com/dwromero/ckconv) | Official | The core layer is PyTorch, but the original project uses an older dependency stack. |
| Gated Linear Attention | [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) | Official | Provides training-ready PyTorch layers with Triton kernels. |
| RetNet | [TorchScale](https://github.com/microsoft/torchscale) | Official | The RetNet layer is available in PyTorch but is packaged inside a larger model library. |
| Mamba with sliding-window attention | [Samba](https://github.com/microsoft/Samba) | Official | Directly combines Mamba and local attention. The sequence layers should be extracted from the language-model framework. |
| xLSTM | [xLSTM](https://github.com/NX-AI/xlstm) | Official | Provides native PyTorch execution as well as optional optimized kernels. |
| Dilated TCN | [TCN](https://github.com/locuslab/TCN) | Official | Small, conventional PyTorch implementation with few external dependencies. |
| Liquid networks and CfC | [Neural Circuit Policies](https://github.com/mlech26l/ncps) | Official | Provides `CfC` and `LTC` as standard PyTorch recurrent modules. |

S5 and Hyena are now available for both Dreamer and STORM. The safest remaining
additions are S4D, a dilated TCN, and CfC. GLA, RetNet, Samba, and xLSTM have
usable implementations but require more adaptation or additional runtime
dependencies.

## World Model Baselines

These are full world-model baselines rather than replacement sequence layers. They are available
through `leworldmodel_dmc_vision`, `temporal_straightening_dmc_vision`, and `tdmpc2_dmc_vision`.

| Method | PyTorch implementation | What to compare |
| --- | --- | --- |
| LeWorldModel | [le-wm](https://github.com/lucas-maes/le-wm) | Uses a train-from-scratch image ViT with the action-conditioned JEPA predictor, SIGReg, and CEM. |
| Temporal Straightening | [temporal-straightening](https://github.com/Agentic-Learning-AI-Lab/temporal-straightening) | Applies stop-gradient prediction and latent-trajectory curvature to spatial image tokens. |
| TD-MPC2 | [tdmpc2](https://github.com/nicklashansen/tdmpc2) | Uses its native pixel encoder, latent dynamics, distributional critics, policy prior, and MPPI control path. |
