# Packaged scope

This export assembles the existing CPU tools and retains the portable cycle/
MoE adaptations already prepared for packaging. Scheduler fast policy, rank
fault harness, file/counter utilities, crossed paired-cycle decisions,
first-divergence T0 logic and quality difference arithmetic come from the
existing labs and tests. Changes are package imports, CLI adapters, explicit
configuration, anonymised calibration labels and public documentation.

The common utilities include create-only atomic publication for small immutable
byte records on macOS and Linux. It establishes file and directory durability,
no-overwrite publication and exact-byte validation; it is not a transaction
across multiple files, a replicated log or an orphan-file cleanup service.

The scheduler retains its original text-only, bounded policy and fixed
execution contracts. It is not a general scheduler emulator. Timing tables
can be replaced, but allocator, graph, preemption, multimodal and device
behavior remain outside the model. The rank harness uses local CPU Gloo;
it does not emulate a GPU collective transport.

Excluded: deployment and restoration controllers, machine discovery, remote
execution, credentials, private logs/prompts, raw profiler captures, traffic
importers, native-image extraction/pinning, model-specific rank adapters,
model weights, serving runtime patches and GPU test execution. The
mac-tools bundle duplicates the individual labs; only its reusable portable
utilities are additionally exposed. No extra live demo server is supplied:
the examples are the existing test fixtures and saved outputs.

The installed tool commands are `gb10check`, `cyclesim`, `moereplay`, `lab`,
`abharness` and `qpanel`. The `lab` command delegates `schedreplay` and
`rankharness` to the existing replay CLIs, exposes `exl3`, `exlr8` and
`tensorfold` adapters, and provides `sha256` and `acceptance` file/counter utilities.
`spark-export-audit` is the release check.

The EXL3 adapter validates format declarations, bounded safetensors headers,
trellis geometry, scale layouts and codebook marker values without reading
weight payloads. It does not evaluate quantization quality or execute a kernel.
The EXLR8 adapter pins a remote Hugging Face revision, validates LFS weight
identities and manifests, recomputes the release set ID, checks index-slice
coverage, checks every expert-index header and samples other headers with
bounded byte ranges. It does not download or
hash weight payloads and does not execute the mandatory E4M3-rounded decoder.
The TensorFold adapter inventories a supplied source/install tree and probes a
supplied OpenAI-compatible endpoint. It never discovers machines, changes a
runtime or stores generated text. Draft/serial equality covers only the exact
probe request and is not a general correctness proof.

MoE replay preserves the existing 256-expert, top-8, four-rank geometry and
packed-weight shape. Its synthetic example uses assumed routes and anonymised
numeric timing priors; it does not qualify a runtime speedup. Scheduler replay
packages the existing fast text-only policy, policy sweep and conditional
comparison. Native-image execution, model-specific rank adapters and private
trace/data extraction are excluded. CPU rank faults exercise local Gloo
processes and do not measure GPU memory or GPU collective behavior.

The static compiler path requires a user-supplied compatible CUDA toolkit
image. Parse-only examples and CPU tests work without it. CUDA compilation,
GPU correctness remains outside the CPU smoke check. The TensorFold adapters
were additionally checked against a live single-node TensorFold deployment on
GB10: its source/EXL3 inventory and a bounded draft-off/draft-on API probe
passed. The public EXLR8 GLM-5.3 release passed the metadata/manifest/header
inspection without a weight download. Neither result is a throughput, quality
or multi-node qualification.
