# Packaged scope

This export assembles the existing CPU tools and retains the portable cycle/
MoE adaptations already prepared for packaging. Scheduler fast policy, rank
fault harness, file/counter utilities, crossed paired-cycle decisions,
first-divergence T0 logic and quality difference arithmetic come from the
existing labs and tests. Changes are package imports, CLI adapters, explicit
configuration, anonymised calibration labels and public documentation.

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
`rankharness` to the existing replay CLIs and exposes `sha256` and `acceptance`
for file/counter utilities. `spark-export-audit` is the release check.

MoE replay preserves the existing 256-expert, top-8, four-rank geometry and
packed-weight shape. Its synthetic example uses assumed routes and anonymised
numeric timing priors; it does not qualify a runtime speedup. Scheduler replay
packages the existing fast text-only policy, policy sweep and conditional
comparison. Native-image execution, model-specific rank adapters and private
trace/data extraction are excluded. CPU rank faults exercise local Gloo
processes and do not measure GPU memory or GPU collective behavior.

The static compiler path requires a user-supplied compatible CUDA toolkit
image. Parse-only examples and CPU tests work without it. CUDA compilation,
GPU correctness, real endpoint evaluation and Linux execution have not been
validated by the local Mac smoke check.
