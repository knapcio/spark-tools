# Versioned runtime observations

`lab observations` imports local, pinned runtime measurements and builds an
empirical lookup table using an explicit disjoint train/holdout split. It uses
the Python standard library, has no endpoint option and performs no network or
subprocess calls. Generated text, API replies' choices, endpoints, credentials,
HTTP key labels, source paths and unrelated log lines never enter its reports.
Input files are bounded regular files; direct symlinks are rejected. Output
files are created exclusively, so an existing report is never overwritten.

## Import contract

Each adapter accepts raw source bytes plus a separate JSON metadata file with
`schema: "spark-observation-import"`, `version: 1`, `adapter`, `source_id`,
`runtime`, `config_sha256`, `model_sha256`, `workload` and `request_ids`. See the
three `*-import.json` examples. Source and request IDs are opaque tokens, not
prompts or URLs. Request IDs must be unique and supplied for every source row,
including a rejected request. Configuration and model identities are SHA256
digests. These identities are operator-declared; the importer validates their
form and pin, not the runtime installation or workload isolation itself.

`runtime.id` is `vllm`, `tensorfold` or `tensorfold-native-zig` as appropriate;
`runtime.source_revision` must match the adapter's pinned revision. TensorFold
may use the explicit `+mia-<40-hex-recipe-revision>` suffix. Optional
`binary_sha256`, `image_sha256` and `patches_sha256` bind installed artifacts.
`workload` contains an opaque `id` and positive integer `concurrency`. Native
response engine, thinking, sampling and draft settings must agree across the
capture; their hash is retained in runtime identity.

The normalized output uses `schema: "spark-observations"`, `version: 1`, a
versioned `adapter`, `identity`, `source_sha256`, `recognized_fields`,
`missing_fields`, `counts.accepted/rejected`, `requests`, `aggregates` and
`scope`. Metrics carry a value, explicit unit and timing/counter basis.
Missing data stays unavailable. The whitelist is independent of arbitrary
source keys. Duplicate JSON keys, nonfinite values, malformed counts and
inconsistent telemetry are rejected before export. CLI errors contain a fixed
diagnostic and do not repeat source content or local filenames.

## Pinned adapters

| Adapter | Upstream source | Required source format |
|---|---|---|
| `vllm-bench-serve-v1` | [vLLM serve.py at 276fbcff](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/benchmarks/serve.py) | Detailed benchmark JSON arrays `input_lens`, `output_lens`, `ttfts`, `latencies`, `itls`, `errors`; `duration`, `completed`, `failed`, successful token totals |
| `tensorfold-native-v1` | [TensorFold openai.zig at db281878](https://github.com/ashhart/TensorFold/blob/db281878ddb836fd0df510d8771ecb7e0fe47d26/zig/src/server/openai.zig), [chat.zig](https://github.com/ashhart/TensorFold/blob/db281878ddb836fd0df510d8771ecb7e0fe47d26/zig/src/server/chat.zig) | One nonstreaming OpenAI response object or JSON array of responses, with `usage`, `tensorfold`, `speculative` |
| `tensorfold-prometheus-v1` | [TensorFold metrics.zig at db281878](https://github.com/ashhart/TensorFold/blob/db281878ddb836fd0df510d8771ecb7e0fe47d26/zig/src/server/metrics.zig) | Complete before/after Prometheus snapshots with pinned histogram buckets and stable labels |

The vLLM adapter validates equal array lengths, completion counts, successful
input/output token totals, and any supplied throughput totals. Failed requests
are excluded and counted only when their output length and timings are zero;
successful telemetry on a failed request rejects the capture. `decode_rate` is
`(output_tokens - 1)/(latency - ttft)`, measured at the client through completion.
`inter_token_latency` retains individual streaming event intervals in seconds;
chunked events are not assumed to equal individual model tokens.
`aggregate_output_rate` is successful output tokens divided by whole-run wall
seconds. These quantities cannot be substituted for each other.

The native adapter preserves TensorFold's reported `tokens_per_second`, whose
audited numerator is output tokens minus one and whose denominator is finish
minus prefill completion. This denominator is not available in the response;
the importer checks it cannot exceed total submit-to-finish time, without
inventing a decode interval or interpreting TTFT as prefill completion. It
checks speculative ratios against integer counts. Native responses do not
provide client ITL arrays or whole-run wall duration. Total response duration
divided by speculative rounds is not exported as a decode cycle cost.

The native adapter also accepts an explicit `spark-tensorfold-native-capture`
version 1 wrapper with a `responses` array. Each item contains `response`
(composed final `usage`, `tensorfold` and `speculative` telemetry) and `client`
with `client_ttft_seconds`, `client_e2e_seconds`, `itls_seconds` and
`stream_token_events`. Client times must be finite/nonnegative and ordered;
content-event count must be positive and no larger than completion tokens,
with exactly one fewer intervals than events. Their sum cannot exceed E2E
minus TTFT. `client_ttft`, `client_latency`, `client_decode_rate` and
`client_time_per_output_token` use basis `client-sse-monotonic`. The latter
two use output tokens minus one and E2E minus TTFT only when both are positive.
They do not replace native `decode_rate`. ITL has basis
`client-sse-monotonic-content-event`: an SSE event may carry several tokens.
Raw OpenAI responses never supply trusted client metrics; only the explicit
versioned wrapper does. The wrapper can omit all text and choices.

Prometheus imports require metadata `interval` with `isolated: true`,
`drained: true`, matching `workload_id`, and positive `duration_seconds`.
Running/waiting/live-token gauges must be zero at both ends. All finished
request latency, TTFT, decode, prefill and TPOT histogram count deltas must
match supplied request IDs. Completion count comes from request latency
histogram counts. When HTTP status counters exist, their successful counts
must match and failed count deltas reject the interval. The native Spark
exporter can omit the HTTP family; failure visibility is then explicitly
`unavailable`, rather than inferred from a latency histogram. Disconnects,
preemptions, changed labels or counter resets reject the interval.
The adapter retains a hash of HTTP label identity, never the key label itself.
Single-token requests without TPOT observations are not a complete interval
for this adapter and require a separate capture.

An optional `--client-capture` versioned native streaming wrapper proves all
interval response objects succeeded. Its complete request count and sums of
input, output, drafted, accepted and round counters must exactly match the
Prometheus interval. The report retains the companion digest and sets failure
visibility to `client-response-proof`; a failing or unmatched companion rejects
the import. Isolation and correspondence still rely on declared capture identity.

Alias policy defaults to `strict-v1`: simultaneous semantic aliases reject
the capture. The explicit `tensorfold-pinned-mirrors-v1` policy accepts the
pinned native/compatibility mirrors only after exact equality of all samples,
label sets and histogram buckets. It selects the first native family and
reports the selected name plus verified mirrors. A partial or conflicting
mirror fails. No metric is summed across aliases.

`pooled_decode_rate` has unit `token/request-second`, because summed request
decode time overlaps under concurrency. It is not per-request decode rate or
aggregate wall throughput. `decode_seconds_per_round` similarly means summed
request decode seconds divided by matched finished-request round counters,
not GPU kernel duration. It is emitted only for the verified counter interval.
Acceptance means accepted draft tokens divided by drafted tokens. None of
these counters establish position survival, a kernel cost or a GLM/cyclesim
calibration.

## Explicit empirical calibration

`spark-observation-split` version 1 requires nonempty `train_request_ids`,
`holdout_request_ids` and a list of supported `metrics`. Both sets must be
unique, disjoint and cover exactly every accepted request across input reports.
Repeated request IDs, source IDs or raw source digests reject calibration;
renaming one capture cannot make it an independent holdout.

Calibration cells match adapter, full runtime artifact identity, config/model
digests, workload/concurrency, exact input/output token counts, metric, unit and
timing-basis hash. Each supported cell needs both training and holdout rows.
Its prediction is the training arithmetic mean. Reports preserve the explicit
IDs and per-holdout observed/predicted values, signed/absolute errors and
percentage error (null when the observed value is zero). Cells with no training
or no holdout return `unavailable` and a null estimate. Missing request metrics
are explicitly unavailable. Prometheus interval summaries provide no individual
request token dimensions and cannot be used for this request-level calibration.

The output schema is `spark-observation-calibration` version 1 with method
`exact-cell-mean-v1`, split/source hashes, cells and unavailable measurements.
The Python `lookup(calibration, cell)` API returns a supported estimate only for
the identical full cell; all other inputs are unavailable. There is no curve
fit, regression, interpolation, extrapolation, kernel attribution, GLM model or
cyclesim parameter inference. Holdout errors describe the supplied capture,
not uncertainty bounds or qualification for other workloads.

## Offline examples

From the repository root, write fresh files outside the fixture directory:

```sh
lab observations import --adapter vllm-bench-serve-v1 \
  --input examples/observations/vllm-bench-serve.json \
  --metadata examples/observations/vllm-import.json --output ../spark-observations-vllm.json
lab observations calibrate --input ../spark-observations-vllm.json \
  --split examples/observations/split.json
lab observations import --adapter tensorfold-native-v1 \
  --input examples/observations/tensorfold-native.json \
  --metadata examples/observations/tensorfold-native-import.json
lab observations import --adapter tensorfold-native-v1 \
  --input examples/observations/tensorfold-streaming.json \
  --metadata examples/observations/tensorfold-native-import.json
lab observations import --adapter tensorfold-prometheus-v1 \
  --before examples/observations/tensorfold-before.prom \
  --input examples/observations/tensorfold-after.prom \
  --metadata examples/observations/tensorfold-prometheus-import.json
```

The checked-in fixtures use pinned upstream field names and histogram layout;
their numerical values and artifact hashes are synthetic, not Spark benchmark
results. A real capture must carry its own source/runtime/config/model identity.
