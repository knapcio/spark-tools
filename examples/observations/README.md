# Observation adapter fixtures

These fixtures follow pinned vLLM and TensorFold source formats. All numbers,
request IDs, generated fixture content and artifact hashes are synthetic.
They are adapter contract tests, not performance measurements.

See [the observation guide](../../docs/observations.md) for commands, metadata,
alias policy, timing definitions and explicit calibration splits. The vLLM
split fits the two 64-input-token training rows and validates the third; the
128-input-token holdout is intentionally unavailable. Prometheus fixtures
include every pinned histogram bucket and native/compatibility aliases with
identical values, to exercise the explicit mirror policy.

`tensorfold-{before,after}-no-http.prom` models the native Spark export shape
without `requests_total`. It must report failure visibility unavailable unless
paired with a matching versioned client response capture. The streaming JSON
fixture is a versioned wrapper with composed final telemetry and client SSE
timings; its client rate and server rate deliberately differ.
