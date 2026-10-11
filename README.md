# Spark Tools

A simulator and lab for testing inference configurations, flags, schedulers,
MoE layouts and kernel changes for NVIDIA GB10 / DGX Spark. Most checks run on
a laptop without a GPU; the optional device validator compares CUDA's runtime
residency answer with compiler resources and the CPU occupancy model on one
Spark. Four Sparks are not required.

| Tool | What it does | GPU needed? | Model dependence |
|---|---|---|---|
| `gb10check` | Checks compiler resources, repeat-build provenance, PTX/SASS ordering and optional CUDA residency API results | Static: no; device validation: one GPU | Model-agnostic; GB10 target and explicit launch geometry |
| `cyclesim` | Predicts decode-cycle cost and throughput | No | Needs timing calibration; includes a GLM-5.3 TP4 example |
| `moereplay` | Replays routing, placement and critical-rank costs for caller-defined MoE geometry | No | Geometry is configurable; each geometry needs its own timing fit |
| `lab conformance` | Exhaustively checks EXL3 and rounded-E4M3 EXLR8 decoding against pinned synthetic fixtures | CPU suite: no; TensorFold CUDA runner: one GPU | Decoder/layout scope only; no weights or model download |
| `lab observations` | Normalizes pinned vLLM/TensorFold measurements and builds exact-cell train/holdout lookups | No | Versioned runtime adapters; never extrapolates |
| `lab experiment` | Verifies local evidence manifests and produces bounded machine/readable reports | No | Recomputes tool decisions; does not run workloads |
| `lab` replay and inspection | Replays scheduler/rank faults; inspects EXL3/EXLR8 checkpoints and TensorFold sources/endpoints | Usually no | Runtime adapters are explicit; scheduler timing needs calibration |
| `abharness` | Compares A/B/A-prime measurements and checks near ties | Client: no | Model-agnostic; caller supplies instrumented observations |
| `qpanel` | Compares task passes against an A/A2 null calibration | Client: no | Model-agnostic; caller supplies tasks and scoring |

The `common.atomic_publish` module provides crash-safe, create-only publication
for small immutable byte records. It writes a unique same-directory temporary
file, makes its contents durable, publishes without replacing an existing path,
and validates identical retries. Interrupted temporary files are ignored and
left to an explicit cleanup policy.

## Five-minute quickstart

From the checkout, with Python 3.10+ on macOS or Linux:

```sh
python3 -m venv ../spark-tools-env
. ../spark-tools-env/bin/activate
export PYTHONDONTWRITEBYTECODE=1
python3 -m pip install -e '.[test]'
gb10check parse --config examples/gb10check/pass.json
cyclesim predict --config examples/cyclesim/calibration.json
cyclesim whatif --config examples/cyclesim/calibration.json \
  --kernel-ms draft_graph=-1.50970368
python3 -m examples.cyclesim.smoke
python3 examples/moereplay/run.py
lab acceptance --before examples/lab/before.prom --after examples/lab/after.prom --k 2
lab conformance cpu --format exlr8
lab observations import --adapter tensorfold-native-v1 \
  --input examples/observations/tensorfold-streaming.json \
  --metadata examples/observations/tensorfold-native-import.json
lab experiment check examples/experiment/manifest.json
```

The checker reports `PASS`; the other commands print predictions or counter
summaries as JSON. These core examples need no GPU, Docker or endpoint. Each tool
has `--help`. Use `lab schedreplay --help` or `lab rankharness --help` for the
replay commands. Rank replay needs the optional CPU PyTorch/Gloo and NumPy
dependencies: `python3 -m pip install -e '.[ranks]'`.

`cyclesim` prediction, sensitivity and target records carry a readable
calibration label plus canonical calibration and optional kernel-table
SHA-256 digests. This makes archived results traceable without embedding local
paths in output.

Optional EXL3/TensorFold checks use resources you supply. Inspection and the
CPU conformance suite do not download checkpoint payloads:

```sh
lab exl3 inspect /path/to/exl3-model
lab exlr8 inspect-hf kindlingai/glm-5.3-exlr8-k2-k3-k4
lab tensorfold source-manifest /path/to/TensorFold --require-exl3
lab tensorfold probe --endpoint http://127.0.0.1:8000/v1 --compare-draft
lab conformance manifest --runner tensorfold-cuda
lab conformance manifest --runner gr8-dense
```

## Platforms

`gb10check` runs anywhere with Docker and a compatible CUDA toolchain image:
macOS via Colima, Linux natively, and on a DGX Spark. Static compilation needs
no GPU. Use an arm64 image for Apple silicon or DGX Spark; x86 Linux accepts
an amd64 image with `--platform linux/amd64`. Saved-output parsing needs only
Python.

The EXL3 inspector reads `config.json` and bounded safetensors headers; it does
not load weights or initialize CUDA. The EXLR8 inspector checks a Hugging Face
release's LFS identities, manifests, variants, index-slice coverage, every
expert-index header and bounded samples of other safetensors headers without
downloading weight payloads. The TensorFold source
manifest hashes and classifies CUDA, Triton, native and Python sources. Its
endpoint probe validates the OpenAI chat API and can compare deterministic
serial and drafted output without saving response text. See
[EXL3, EXLR8 and TensorFold](docs/exl3-tensorfold.md).

`cyclesim`, `moereplay` and `lab` run in Python on macOS/Linux. Scheduler and
file utilities use the standard library; rank replay adds CPU PyTorch/Gloo and
NumPy. `abharness` and `qpanel` run from any machine that can reach an
OpenAI-compatible endpoint. The endpoint's server may need a GPU.

The external conformance manifests pin exact TensorFold, `gr8` and EXLR8 source
revisions and verify selected source hashes before execution. The `gr8` path
uses small synthetic fixtures on CPU. The TensorFold CUDA path and
`gb10check validate-device` need one available GPU; neither needs four Sparks
or model weights. A PASS is decoder or residency-API evidence, not a throughput
or end-to-end model-quality result.

## Examples and limits

Each example has a command, expected output and runtime:
[static checks](examples/gb10check/), [cycle model](examples/cyclesim/),
[MoE routes](examples/moereplay/), [scheduler](examples/schedreplay/),
[rank faults](examples/rankharness/), [file/counter utilities](examples/lab/),
[A/B/A-prime and near ties](examples/abharness/),
[quality calibration](examples/qpanel/), and
[EXL3/TensorFold adapters](examples/exl3-tensorfold/),
[decoder conformance](examples/conformance/),
[runtime observations](examples/observations/), and
[evidence manifests](examples/experiment/).
The observation examples include a [recorded exact-cell Qwen3.8 Flash Next
TensorFold panel on one GB10](examples/observations/qwen38-gb10/); it is a
reproducible evidence example, not a cross-system speed claim.

Static checks cannot establish GPU correctness or speed. Replay leaves out
real allocator, network, cache and asynchronous execution effects. The six
included cycle validation observations have 3.71% maximum absolute error
across three protocols. Earlier throughput comparisons reached 19.39% error at one
stream and 17.85% at two/four streams. These numbers describe one setup, not
accuracy guarantees. New models need measurements; endpoint fixtures do not
measure a real model. See [calibration](docs/calibration.md),
[endpoint usage](docs/endpoints.md) and [scope](docs/scope.md).

Legacy MoE replay keeps the original 256-expert, top-8, four-rank calibrated
example. Version 2 traces accept explicit expert count, top-k, world size and
optional per-expert bytes. Timings are unavailable until a fit is bound to the
exact geometry; legacy GLM timings are never reused for another shape. Layout
and collective assumptions do not establish serving support.

EXL3 inspection establishes header/layout compatibility only. EXLR8 remote
inspection does not download or recompute weight payloads. The separate CPU
conformance suite exercises all decoder states and synthetic dense layouts,
but not model inference. TensorFold source inventory does not compile kernels,
and its API probe does not establish throughput, quality, GPU correctness or
multi-rank safety. Observation calibration supports only identical recorded
cells and deliberately refuses interpolation or extrapolation. Those broader
claims require separate measurements on the target runtime and hardware.

By knapcio. Apache-2.0 with retained upstream MIT notices. See
[CREDITS.md](CREDITS.md), [NOTICE](NOTICE), [LICENSE](LICENSE) and
[CHANGELOG.md](CHANGELOG.md) for attribution and release history. See the
[export policy](docs/export-policy.md) for release checks.
