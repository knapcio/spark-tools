# Changelog

All notable changes to Spark Tools are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-10-10

### Added

- Configurable MoE route schema v2 with explicit expert count, top-k, world
  size, optional expert footprint, generic TP/EP layouts and geometry-bound
  timing fits. The legacy schema and calibrated GLM-5.3 example remain exact.
- Versioned vLLM and TensorFold observation importers, including native
  response/SSE telemetry and isolated Prometheus intervals, plus explicit
  train/holdout exact-cell calibration with no interpolation or extrapolation.
- Exhaustive CPU decoder conformance for EXL3 trellis codebooks and EXLR8
  rounded E4M3, synthetic K2/K3/K4 dense-layout fixtures, negative controls
  and pinned external TensorFold CUDA and `gr8` runner manifests.
- Offline experiment manifests that hash local evidence, recompute existing
  A/B/A-prime and quality decisions, and produce bounded JSON/Markdown reports
  without running workloads or contacting endpoints.
- `gb10check validate-device` for comparing compiler resources, the CPU
  occupancy approximation and CUDA's runtime residency API on a selected GB10.
- `gb10check repeat-build` for two clean GPU-free CUDA builds with toolchain,
  dependency-closure and artifact comparison records.
- GitHub Actions coverage for Python 3.10/3.13 on Linux, Python 3.13 on macOS,
  optional CPU rank tests, export auditing, distributions and installed-wheel
  examples.

### Reliability

- Observation adapters reject mixed or incomplete intervals, counter resets,
  failed/disconnected/preempted requests, unsafe inputs and mismatched native
  companions. Missing runtime fields remain explicitly unavailable.
- Generic MoE geometry cannot inherit the legacy model's timing prior; missing
  geometry-specific fits produce unavailable timing rather than an estimate.
- Decoder fixtures are hash-pinned and external runners verify exact upstream
  revisions and selected source files before execution; no model weights or
  third-party runtime source are bundled.
- Build and device reports distinguish static compilation, API-estimated
  residency and measured performance. No residency result is presented as
  achieved occupancy or speed.
- Prometheus histogram deltas must remain cumulative and count-bounded after
  interval subtraction; individually valid but mutually impossible snapshots
  are rejected.
- Device launch validation now checks both thread limits and the observed
  function dynamic-shared-memory limit independently of occupancy API output.
- Python 3.10 test installs include the `tomllib` compatibility dependency,
  and serialized bootstrap intervals are stable across supported Python
  versions.

## [0.2.0] - 2026-10-10

### Added

- Header-only EXL3 checkpoint inspection with bounded configuration, header
  and codebook-marker reads.
- Metadata-only EXLR8 release inspection for Hugging Face, including immutable
  revision pinning, LFS identity and `set_id` checks, width manifests, TP/rank
  index-slice coverage, every expert-index header and bounded samples of other
  safetensors headers.
- TensorFold source manifests for CUDA, Triton, native and Python components.
- A bounded TensorFold OpenAI-compatible endpoint probe with optional
  deterministic draft-off/draft-on comparison.
- Documentation and public examples for EXL3, EXLR8 and TensorFold.

### Reliability

- Remote inspection is restricted to Hugging Face HTTPS hosts, strips
  credentials on cross-host redirects and enforces response-size limits.
- New adapters reject symlinks, oversized or malformed inputs, incomplete TP
  rank sets, missing hash sidecars and incompatible EXLR8 E4M3 metadata.
- Reports omit model output, endpoint addresses, credentials and local source
  paths.

## [0.1.0] - 2026-10-10

### Added

- Initial public release of the decode-cycle simulator and CPU replay lab.
- Static GB10 compiler-resource, occupancy and PTX/SASS ordering checks.
- MoE route replay, scheduler replay and CPU/Gloo rank-fault harnesses.
- A/B/A-prime performance screening, near-tie checks and a null-calibrated
  quality panel.
- Portable examples, calibration fixtures, attribution, packaging and a
  fail-closed export audit.

[0.3.0]: https://github.com/knapcio/spark-tools/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/knapcio/spark-tools/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/knapcio/spark-tools/releases/tag/v0.1.0
