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
The image must also contain `nvcc`, `ptxas`, `cuobjdump`, `nvdisasm`, a C++
compiler and the CUDA occupancy header. A runtime image can contain the CUDA
compiler but omit `nvdisasm`; in that case SASS inspection fails and the
report records the failed command.

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

Compiler reports retain `mode: STATIC_NO_GPU` and add `build_provenance`.
It records the canonical configuration digest, configuration file hash,
hashes of each declared source and ordering baseline, executed argument arrays
(including failures), toolkit versions, the installed occupancy header hash,
the occupancy wrapper source hash, target and requested/observed platform.
Additional files can be declared in an `inputs` array at configuration or
entry level; every declared file must stay within the input directory.
The Triton AOT API arguments are recorded separately from subprocess commands.

The launcher inspects the local image before compilation. When an immutable
image ID is observable, it records that ID and available repository digests
and runs by the ID. Otherwise image identity is `unknown`; a digest supplied
in the reference is recorded as declared, without implying verification.
Image inspection retains only identity and platform fields.

This is build provenance, not reproducible-build proof. Transitive headers,
imported Python modules, compiler caches and image contents are not independently
audited. Matching these fields does not prove bit-identical output, GPU
correctness, or performance. The occupancy formulas and GPU-free checks are
unchanged.
Argument arrays cover processes issued by the adapter; compiler-internal
subprocesses, including those inside Triton, are not traced.

## Explicit device API validation

The checked-in [GB10 verification record](../examples/gb10check/verified-gb10.json)
records 92/92 device-API cases and two complete matching clean builds on one
GB10. The record is a sanitized summary bound to private raw-report hashes;
its PASS has the narrow meanings defined in this document.

This command is opt-in and requires a Linux GB10 host, CUDA 13 development
tools and an explicit device ordinal:

    gb10check validate-device --cases examples/gb10check/device-cases.json \
      --device 0 --out /path/to/new-device-report --timeout 600

The output directory must be new and outside the checkout and case directory.
The command compiles 33 deterministic kernel variants for 92 bounded requests.
It records actual compiled registers, static shared bytes, barriers and spills;
register-pressure inputs are requests, not assertions about allocated registers.
The matrix covers every multiple of 32 threads from 32 through 1024, register
pressure boundaries, static and dynamic shared-memory boundaries, ordinary and
opt-in limits, launch bounds, both occupancy flags, carveout preferences and
invalid requests. These fixtures assume GB10 shared-memory limits; observed
device properties are retained and differences fail rather than being adjusted.

The helper selects only the supplied ordinal, verifies compute capability 12.1,
calls cudaGetDeviceProperties and cudaFuncGetAttributes, and queries
cudaOccupancyMaxActiveBlocksPerMultiprocessor (or its WithFlags variant).
The installed cuda_occupancy.h receives the observed device/function properties,
the queried preferred carveout and maximum dynamic shared-memory limit, and the
barrier count from the compiled ptxas record. Cache flags are mapped from the
observed cacheModeCA attribute to the header's corresponding partitioned-cache
setting; partitioned caching does not affect GB10. Each case runs in a fresh
helper process, so per-function attributes cannot leak between cases.

Reports compare blocks per SM and retain the original CPU approximation and its
signed error. Invalid requests must be rejected by attribute setters or both
calculators. Thread counts outside the device or compiled function's explicit
maximum, and dynamic shared-memory requests above the function attribute, are
rejected by separate launch-validity guards: CUDA's occupancy API can still
return a residency estimate for a request that the function cannot launch.
Unexpected driver errors fail. A mismatch remains visible and returns a
failing command. Runtime API results are residency estimates, not
measured occupancy, GPU kernel correctness, or speed. This distinction follows
the [CUDA runtime occupancy API](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-runtime-api/group__CUDART__OCCUPANCY.html).
The helper never launches a kernel or allocates a device buffer. Selecting a
device and querying function attributes can initialize a CUDA context.

Use --build-only to compile and inspect ptxas output without any device query.
That report is labeled BUILD_ONLY_NO_GPU and contains no device-validation
results. The existing parse and compile commands retain their GPU-free behavior.

## Two clean builds

    gb10check repeat-build --config examples/gb10check/compile-pass.json \
      --image example/toolkit@sha256:REPLACE_WITH_LOCAL_DIGEST \
      --out /path/to/new-repeat-report --timeout 900

Repeat-build supports standalone CUDA entries and requires a locally observable
immutable image ID. The image must include nvdisasm. It snapshots the complete
input directory (at most 512 files and 32 MiB), hashes and marks the snapshot
files read-only, and mounts it read-only into two new containers. Both builds
use the same in-container input and output paths, the same image ID, and no
devices or network. Snapshot drift before, between or after builds fails.

The compiler's dependency output supplies include-closure hashes for each CUDA
entry and the occupancy wrapper. Reports retain exact compiler argument arrays,
toolchain versions, header and tool source hashes, target, platform, image
identity and exact cubin, PTX, SASS and occupancy executable hashes. The final
repeat-report.json lists differences without stripping paths, whitespace,
symbols or other bytes. Missing artifacts and incomplete builds cannot pass,
even if both fail identically. Matching two builds is bounded repeatability
evidence, not a general reproducible-build proof or hardware qualification.
Triton and standalone PTX repeats are refused because this mode does not claim
to capture their full Python/assembler include closure.

## Operator runbook

Run only on the explicitly selected standalone GB10 system. Keep its existing
management connection and local recovery path available. Choose a time when
the serving owner accepts an additional CUDA context; no service stop or
configuration change is required by these commands. Preserve outputs outside
the checkout and do not automatically retry a failed device query.

1. Verify the selected device with nvidia-smi -i 0 and identify its GB10 model,
   current driver, existing workloads and available memory.
2. Verify the development toolkit contains nvcc, ptxas, the CUDA occupancy
   header and a host C++ compiler. Record the checkout revision and file hashes.
3. Run the build-only command into a new output directory. Inspect report.json
   and compile.log before allowing the explicit device-query command.
4. Run validate-device with --device 0 and the 92-case fixture into a second
   new directory. Keep report.json, every per-case JSON observation, compiler
   output, generated kernels, dependency closure and executable hashes.
5. Inspect all mismatches and API errors. PASS concerns residency-model agreement
   only; it does not authorize serving changes. Keep the CPU approximation
   errors in the result, including nonzero errors.
6. Run repeat-build separately with an immutable local toolkit image, without
   GPU access. Preserve both raw builds and repeat-report.json.

No command discovers another machine, opens SSH, enumerates alternate devices,
modifies driver settings, launches kernels or promotes a runtime.
