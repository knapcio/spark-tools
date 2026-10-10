# Static kernel checks

Run from the checkout root:

```sh
gb10check parse --config examples/gb10check/pass.json
gb10check parse --config examples/gb10check/fail.json
```

Runtime: under one second on macOS/Linux; no GPU, VM or toolkit needed. The first
command exits 0 with `status: PASS`, 16 registers, zero spills, 256 static
shared bytes and estimated 100% warp occupancy at 128 threads. The second
exits 1 with `status: FAIL`: 128 total spill bytes and 110,592 dynamic shared
bytes exceed the configured budgets. Both complete JSON outputs are saved
as `expected-pass.json` and `expected-fail.json`.

The failing command intentionally returns a nonzero exit code. When chaining
commands in a shell with `set -e`, handle that expected result explicitly.

The saved ptxas/resource/SASS fixtures are **illustrative synthetic compiler
outputs**, not measurements from an executed GPU. They make parsing and
failure policies reproducible. `copy.cu` and `pressure.cu` are actual small
standalone kernels for the optional compile path; compiler versions can
produce different register and spill counts. The failing kernel's explicit
dynamic shared allocation always exceeds the configured user budget.

For static compilation on Linux, use a local Docker installation with a
compatible toolkit image; no VM setup is needed. On Apple silicon, install Colima and a Docker CLI
using your preferred package manager. Start an arm64 VM locally:

```sh
colima start --arch aarch64 --cpu 4 --memory 8
```

Choose **any Linux arm64 CUDA 13 toolkit image** containing `nvcc`, `ptxas`,
`cuobjdump`, `cuda_occupancy.h`, Python 3.10+ and a C++ compiler. Confirm its
`nvcc --list-gpu-code` includes `sm_121` and `sm_121a`. No private image is
required or referenced. Obtain/build the image yourself under its licence;
the project does not provision a VM or download an image automatically.
The container command uses `--pull never`; the image must already be local.

```sh
gb10check compile --config examples/gb10check/compile-pass.json \
  --image "$CUDA_ARM64_IMAGE" --out "$EXTERNAL_OUTPUT/pass"
gb10check compile --config examples/gb10check/compile-fail.json \
  --image "$CUDA_ARM64_IMAGE" --out "$EXTERNAL_OUTPUT/fail"
```

Set `CUDA_ARM64_IMAGE` to your toolkit image and `EXTERNAL_OUTPUT` to a fresh
directory outside the checkout. Runtime: roughly 10-60 seconds once the VM
and image are available; initial setup may take several minutes. Expected:
passing resource checks for the copy, failing shared-memory budget for the
pressure case. Inspect `report.json`, ptxas, PTX and SASS in that directory.
These are compile-only checks; there is no GPU execution part in this example.
Inputs and the two checker packages are mounted read-only; compiler output is
the only writable project mount. The command also works from an installed
wheel, with the example input files supplied separately.

See [static checks](../../docs/gb10check.md) for PTX and Triton AOT manifests,
ordering rules and the limits of static analysis.

On x86 Linux, choose a matching Linux amd64 toolkit image and add
`--platform linux/amd64`. A DGX Spark uses the default arm64 platform.
