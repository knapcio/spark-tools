# Scheduler replay fixtures

From the checkout root:

```sh
lab schedreplay run --config configs/scheduler.json \
  --workload examples/schedreplay/newcomer.jsonl
lab schedreplay run --config configs/scheduler.json \
  --workload examples/schedreplay/agents.jsonl
```

Expected: two completed requests for the newcomer fixture and four for the
agent-turn fixture. Reports include TTFT, stalls, admission reservations,
queue depth and unsupported timing regimes. `expected-newcomer.json` and
`expected-agents.json` contain the complete reports. Runtime: under one
second each on macOS/Linux. No GPU or endpoint.

These are the existing synthetic test workloads, without prompt text. The
existing fast scheduler and two-deep serial device model are preserved.
`configs/scheduler-timing.json` contains only anonymised fit values and
neutral sample labels. Shapes and policy constraints remain those of the
original lab; changing timings does not establish support for a new engine.
Native-image comparison and private traffic extraction are excluded.

The existing policy sweep is also available. This small example runs OFF and
fair policies with two decode intervals:

```sh
lab schedreplay sweep --config configs/scheduler.json \
  --workload examples/schedreplay/newcomer.jsonl \
  --chunks 4096 --decode-steps 0,40 --admissions on
```

Expected: three cases, each with two completed requests. The conditional
one-decoder/cold-prefill comparison is indeterminate for this newcomer fixture.
Runtime: under one second; no GPU. Unsupported timing regimes withhold rankings.
