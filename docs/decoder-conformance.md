# Dense decoder conformance

The checked-in [verification record](../examples/conformance/verified-runs.json)
records a pinned `gr8` synthetic CPU run (4/4 tests) and the pinned TensorFold
EXL3 CUDA suite on one GB10 (62/62 tests, no skips). It contains hashes and
bounded counts rather than local paths or raw logs. These results qualify the
dense decoder paths described below, not model inference or performance.

`lab conformance` checks **dense decoder conformance only**. It provides a
dependency-free CPU reference, synthetic fixtures, and pinned external runner
plans. PASS never means model correctness, model quality, collective correctness,
whole-engine correctness, checkpoint qualification or improved performance.

```sh
lab conformance cpu
lab conformance cpu --format exlr8
lab conformance fixtures
lab conformance manifest --runner tensorfold-cuda
lab conformance manifest --runner gr8-dense
lab conformance cuda
```

`cpu` and `fixtures` return 0 for PASS and 1 for FAIL. `manifest` returns 0 for
PLAN_ONLY; it executes nothing. `cuda` returns 3 with UNAVAILABLE and zero
executed tests: there is no packaged local GPU API adapter. CPU arithmetic is
never presented as a simulated GPU run. JSON is printed to stdout; callers may
save it under their chosen result directory. Commands require no model files,
checkpoint downloads, CUDA, PyTorch, NumPy, Rust or network access.

## Exact checks and bounds

The CPU suite checks all 65,536 16-bit states for each of the EXL3 `3inst`, `mcg`
and `mul1` codebooks against fixed fp16 little-endian SHA256 oracles. It checks
the EXLR8 MCG table against both FP32 and E4M3 byte oracles. The MCG half-value
sum is rounded to fp16 before conversion to FP32. EXLR8 then clamps to +/-448
and rounds to `float8_e4m3fn`, round-to-nearest with ties-to-even; this E4M3
rounding is part of the format. Stock EXL3 MCG is a negative control: 64,372
states have different FP32 bytes and it must be rejected.

An independent exponent/significand rounding implementation checks every state
against the table-search implementation. All 126 positive finite E4M3
intervals test the exact halfway value and its adjacent float64 neighbors,
with both signs (756 inputs). Eight explicit clamp/saturation/signed-zero
inputs bring the finite input total to 66,300. Both signed zeros are retained.
NaN and +/-infinity fixture inputs are rejected; the generated MCG values are
all finite. This fixture policy makes no assertion about upstream conversion
of nonfinite input. E4M3 codes `0x7f` and `0xff` decode as NaN.

EXL3 fixtures cover all integer widths 1..8 for three codebooks and half widths
1.5, 2.5, 3.5 for `mul1` (27 tiles). EXLR8 fixtures cover K2/K3/K4 and both entry
sides in 256x256 matrices (6 matrices). The packed byte pattern is versioned
integer arithmetic, independent of random-library versions. Inputs and exact
decoded bytes have SHA256 records; the canonical fixture set has its own hash.

The EXLR8 path uses swapped little-endian 16-bit words, MSB-first K-bit symbols
and circular symbol history. A separate little-endian 32-bit sliding-span
decoder agrees on 24 fixtures of 256 states. For integral K these are equivalent
representations. Tests reject unswapped words, reversed bits, and omitted E4M3
rounding. Tile stream positions map to the tensor-core 16x16 permutation.
Output-side entries store `[input_tile][output_tile_within_128]`; input-side
entries store `[output_128_block][input_tile_within_128][output_tile_within_128]`.
The assembled matrix is transposed into HF `(out,in)` orientation.

CPU exactness is checked in the rotated domain before Hadamard reconstruction
and scale multiplication. Synthetic matrix dimensions are limited to 128 or
256; no arbitrary model allocation exists. CPU PASS does not qualify fp32
Hadamard/GEMM accumulation or the gr8 engine. Those are separate external checks.

## Pinned external oracles

| Upstream | Immutable revision | External entry point |
|---|---|---|
| [TensorFold](https://github.com/ashhart/TensorFold/tree/db281878ddb836fd0df510d8771ecb7e0fe47d26) | `db281878ddb836fd0df510d8771ecb7e0fe47d26` | `tests/cuda/test_exl3_linear.py` |
| [gr8](https://github.com/kindlingai/gr8/tree/d40ab1d8e010e125890b9220f3cf1cd80a4a8e06) | `d40ab1d8e010e125890b9220f3cf1cd80a4a8e06` | `engine/loader-rs/tests/{make_exlr8_fixture.py,exlr8_fixture.rs}` |
| [EXLR8](https://github.com/kindlingai/exlr8/tree/d56e3ecf7aa3104be0b77b59b076406562fd20c5) | `d56e3ecf7aa3104be0b77b59b076406562fd20c5` | `loader/exlr8.py` |

Manifests include exact source-file hashes, license-file hashes, dependencies,
argument arrays, working directories and per-command deadlines (600 seconds
per runner). Placeholders must be resolved to absolute paths; the manifests are
reviewable plans, not shell strings or an execution API. Use independent
disposable checkouts. Pin HEAD and check the tracked tree plus entry-point file
hashes before execution:

```sh
lab conformance check-sources --upstream tensorfold --source /absolute/disposable/TensorFold
lab conformance check-sources --upstream gr8 --source /absolute/disposable/gr8
lab conformance check-sources --upstream exlr8 --source /absolute/disposable/exlr8
```

The checker does not import code. It rejects pin drift, modified tracked files,
symlinked pinned paths and source files larger than 1 MiB; git calls have 15-second
timeouts. It does not inventory untracked files or attest imported dependencies.
Keep disposable checkouts free of untracked Python modules and record the
environment/image identity. Pin validation is not an execution verdict.

TensorFold's `test_decode_is_bit_exact` compares CUDA decoder fp16 bit patterns
against its CPU format reference, with zero tolerance. Its
`test_matches_the_float64_reference` separately checks relative L2 matmul error
below 0.002, including input fp16 rounding. The test module also checks row
invariance, dtypes, bias, stored/strips layout and split-K behavior. These test
categories must be reported separately. CUDA preflight must succeed and all
tests must execute with zero skips, failures or errors; a module-level "CUDA
only" skip does not qualify anything. Full-module execution includes larger
synthetic matrices than this CPU suite and may compile CUDA code.

The gr8 recipe generates a seeded fixture with the pinned EXLR8 loader and runs
four Rust tests. It compares 11 dense matrices across TRELLIS, INT4, MXFP8,
FP8_BLOCK, FP8_ROW, BF16 and FP32 using FNV-1a over decoded FP32 little-endian
bytes, plus full byte comparisons where small. It checks block/whole equality,
entry-alignment rejection, parallel decode equality and INT4 HF orientation.
The trellis coverage is K4 on both 256x256 sides, K3 output-side and K2
input-side at 128x128. This is bounded dense loader coverage; no expert/model
or whole-engine claim follows. The Rust source uses `slice::as_chunks`, requiring
[Rust >=1.88](https://doc.rust-lang.org/std/primitive.slice.html#method.as_chunks).
Cargo build/toolchain availability remains externally verified.

Hash the resulting safetensors fixture before and after the Rust run (`lab
sha256 FILE`). Keep generated fixture hashes distinct from fixed reference
hashes. The generator overwrites its fixture path: use only a disposable gr8
checkout and place Cargo's target directory inside a separate result directory.
The full upstream fixture is not bundled.

## Standalone GB10 execution handoff

The packaged manifest and source checker perform no SSH, container or serving
action. Check the selected standalone GB10 and any existing service state before
GPU qualification; no multi-node cluster is required.

Run the stdlib CPU suite first; it is portable to ARM64 Linux. For a GPU run,
copy/install this package and use disposable directories created with
`mktemp -d`, separate from any serving TensorFold checkout. Resolve
the chosen manifest with an isolated Python environment or an ARM64-compatible
image whose digest is recorded. Read-only checks should confirm the target
hostname, GPU/driver/toolchain and available memory. Do not use a cluster lock,
change clocks, stop/restart serving, expose a port or install into the active
serving environment as part of this decoder plan.

Run CUDA tests only in an operator-owned idle maintenance period with the recovery path for
the existing service already available. The runner deadline is at most 600
seconds; apply it to the owned subprocess group, preserve stdout/stderr and
return codes, and terminate only that group on timeout. Record all skips and
errors, source pins, dependency versions, fixture hashes and GPU identity.
Perform a read-only service health check afterwards. A CPU PASS and an external
PLAN_ONLY manifest do not authorize claiming a GPU PASS. No model weights or
whole-engine launch are required by either manifest.

## Oracle provenance and credit

On 2026-10-10, public pinned sources were read by immutable URL and their hashes
checked before the CPU oracle calculation. Exhaustive tables and 27 EXL3 tile
hashes were produced by pinned TensorFold `format.py` using NumPy; the six
EXLR8 rotated-domain matrix hashes were produced by pinned EXLR8
`tiles_in_major`/`tiles_out_major`, not by this package's implementation.
Environment: Python 3.12.14, NumPy 2.5.3, PyTorch 2.14.1 on macOS CPU. Only the
hash results and independent format arithmetic are retained here. Public source
files, models and weights are not vendored.

Credit: Turboderp / ExLlamaV3 (MIT, copyright 2025) for EXL3 trellis/codebook/tile
format; Ash Hart / TensorFold contributors (Apache-2.0, with MIT EXL3 attribution
in `format.py`); Kindling AI / gr8 contributors (MIT, copyright 2026 g6 engine
contributors) for the dense loader conformance fixture; Kindling AI EXLR8's
loader retains Turboderp's MIT license in `loader/LICENSE`. Existing project
MIT notices remain applicable. The new independent code is Apache-2.0; external
implementations retain their own licenses and are never relicensed here.
