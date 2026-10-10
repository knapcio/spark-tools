# Calibration and model limits

cyclesim is the existing decode-cycle cost model, packaged without changing
its prediction arithmetic. It is pure Python and runs on macOS and Linux
without a GPU. It consumes an explicit schema-5 calibration configuration.

Cycle time is the sum of category mean expected-work costs plus a signed
closure residual. Expected committed tokens are `1 + sum(prefix_survival)`;
per-stream throughput is `1000 * committed / cycle_ms`. Prefix survival is
unconditional, finite and nonincreasing. Category and kernel aliases share
one time/traffic ledger, and edits must satisfy the configured DRAM floor.

The GLM-5.3 TP4 example contains anonymised dimensions and numerical
coefficients from the existing calibration. Its single-stream baseline is
73.482 ms for prose and 83.423 ms for code. Six held-out timing observations
span three protocols; the final two use the alternate NVFP4 head. Their
largest absolute cycle error is 3.71%. This subset does not establish a
universal bound. Historical throughput transfer errors reach 19.39% for
single-stream and 17.85% for concurrent observations.

Routing occupancy, category traffic, unmeasured concurrency acceptance and
single-stream draft-pass costs retain their original assumptions. Empirical
head and draft-width changes use separately measured paired deltas. New
confidence thresholds, policies or reduced single-stream draft widths require
identified observations. The shape ledger retains the source architecture
and layer counts; it is not a general model architecture parser.

The projection and glue examples are engineering hypotheses compared with
retained measured percentages from a different configuration. Do not add the
percentages or interpret the comparison as a release decision.

Saved-profile ingestion retains automatic graph segmentation, varying draft
passes, interval-union busy time and ordinal rank comparison. Explicit
bindings identify head precision and working-set bytes. Strict ingestion
checks local manifests, hashes, counters, descriptors and capture coverage.
The synthetic profile and companion examples contain no measured raw traces.

See [the runnable examples](../examples/cyclesim/README.md) for commands,
expected outputs, runtime, assumptions and the original regression fixtures.
