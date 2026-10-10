# Changelog

All notable changes to Spark Tools are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

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

[0.2.0]: https://github.com/knapcio/spark-tools/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/knapcio/spark-tools/releases/tag/v0.1.0
