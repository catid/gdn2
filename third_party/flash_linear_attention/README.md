# Vendored KDA kernel snapshot

This directory contains the Kimi Delta Attention (KDA) layer and Triton kernels
from [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention),
commit `9e82a80`. The snapshot is included so the KDA baseline used by the
benchmark is inspectable alongside the optimized GDN2 kernels.

The files retain their upstream MIT license. They import the normal `fla`
package for shared modules and utility kernels; install FLA from the pinned
commit in `requirements.txt` when running the benchmark. The benchmark uses
`fla.layers.kda.KimiDeltaAttention`, which dispatches to these upstream KDA
kernels when that package is installed.
