# Static GB10 checks

`gb10check parse` reads saved ptxas entry records, optional cuobjdump resource
records, PTX and SASS. Configuration supplies launch threads, dynamic shared
memory, target and explicit thresholds per entry. Missing resources, unknown
geometry, compiler errors and target drift fail. Zero GPU devices are needed.

`gb10check compile` runs the same checks inside a GPU-free Linux container.
It supports standalone CUDA (`kind: cuda`), PTX (`kind: ptx`) and Triton AOT
(`kind: triton`) entries. Both `sm_121` and `sm_121a` are accepted. The
[CUDA 13 compiler guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-compiler-driver-nvcc/index.html)
lists both targets. Use a compatible image and preserve its exact digest
alongside your experiment output.

The default parse-only occupancy fallback models 48 SMs, 65,536 registers per SM,
48 warps per SM, 24 blocks per SM, 102,400 shared bytes per SM and 1,024
reserved shared bytes per block. Hardware properties can be overridden in
configuration. The rounded fallback omits subpartition and barrier limits;
it is an approximation that can overestimate occupancy. Compilation additionally
builds the CPU-only wrapper around the installed NVIDIA `cuda_occupancy.h`,
which accounts for more allocation rules. Neither queries a GPU. See the
[Blackwell tuning guide](https://docs.nvidia.com/cuda/archive/13.1.0/blackwell-tuning-guide/index.html)
for register and occupancy context. Kernel execution remains necessary to
validate correctness and performance.

A Triton entry supplies `source`, `function`, `signature`, `constants`,
`attrs`, `options`, `kernels` and `thresholds`. `attrs` keys are argument
names. The compiler target is `GPUTarget("cuda", 121, 32)`. Launch threads
must equal `num_warps * 32` and dynamic shared memory must match static
metadata. The adapter reads `asm` and metadata and reassembles PTX using
ptxas; it never reads properties that initialize CUDA handles. The input
module must contain only AOT definitions and no runtime or device setup.
Optional Triton must be installed in the chosen container image.

Ordering rules accept regex arrays: `required`, `forbidden`, `sequences`
(ordered arrays), and `unchanged_counts` against a specified baseline file.
The optional `barrier_cut` takes `scratch_parameter` and `reset_parameter`.
It conservatively checks all represented direct PTX paths from scratch
loads to L2 discards, including conditional branches and loop backedges.
Predicated barriers are not treated as guaranteed cuts. Ambiguous scratch
provenance, multiple entries, calls and indirect branches are rejected.
The later-reset check and SASS sequence checks are local ordering diagnostics.
They cannot establish scratch ownership, synchronization across blocks,
launch-stream ordering or memory-model correctness.
