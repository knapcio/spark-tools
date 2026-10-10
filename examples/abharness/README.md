# A/B/A-prime and near-tie fixtures

From the checkout root:

```sh
python3 -m abharness decide --prereg examples/abharness/prereg.json \
  --input examples/abharness/panels.json
python3 -m abharness near-tie --input examples/abharness/near-tie.json
```

Expected: PASS with 0.300 ms mean prose cycle saving, and PASS_NEAR_TIE with
one divergent row and a 0.100 nat first-divergence margin. Complete outputs
are in `expected-decision.json` and `expected-near-tie.json`.
Runtime: under one second each on macOS/Linux. No GPU or endpoint.

These are the existing synthetic decision tests: three panels, each with ten
prose and ten code observations; A=8 ms, A-prime=12 ms, B=9.7 ms. Each prompt
is paired before averaging. The original crossed paired bootstrap, 0.2 ms
prose saving gate, 99.5% committed-token floor and 0.5% code regression cap
are retained. The preregistration binds that fixed rule and the source hashes.

The T0 fixture retains the original 30-row, three-repeat cold-cache evidence
contract and two A top-five references. It tests first-token equality,
repeat stability and a <=0.5 nat margin at the first divergence. All evidence
here is synthetic. Real runs need native token IDs and instrumented cache/
cycle observations; a generic HTTP response does not supply those guarantees.
See [endpoint usage](../../docs/endpoints.md) for the request adapter.

Install the package and the dependencies used by the test commands:

```sh
pip install -e '.[test]'
```

The console command `abharness` accepts the same arguments after installation.
To freeze your cycle rule before collecting observations, use
`python3 -m abharness preregister --config configs/ab-fixture.json --prereg
"$EXTERNAL_OUTPUT/plan.json"`. The destination and its `.sha256` companion must
both be new files. The numeric rule is fixed; unsupported thresholds are
rejected before the plan is created. The near-tie command evaluates its own
saved evidence; run both checks when both gates are required.

The retained bracket-screen fixtures can also be run directly:

```sh
python3 -m pytest tests/test_abharness_screening.py
```

Expected: 10 passed, under two seconds on macOS/Linux, no GPU or endpoint.
These tests use the existing synthetic six-prompt native trajectory bank.
`abharness.screening.bracket(before, after, candidate, det_pair_stats, ...)`
compares the candidate separately with each surrounding baseline, preserving
the profile thresholds below. It is a library screen with
`release_gate=false`; it does not change any runtime configuration. The
three-panel `decide` command above retains its separate paired-mean rule.

### Screening input contract

Each arm is a nonempty dictionary with the same prompt keys. Each observation
contains the fields in this complete, synthetic one-prompt example:

```python
from copy import deepcopy
from abharness.screening import bracket, det_pair_stats

before = {'probe': {
    'metadata': {'kind': 'prose', 'i': 0, 'pair_key': 'probe', 'thinking': False},
    'token_ids': [1, 2],
    'top_logprobs': [
        [{'token_id': n, 'logprob': -.1*n} for n in (1, 2, 3, 4, 5)],
        [{'token_id': n, 'logprob': -.1*rank}
         for rank, n in enumerate((2, 1, 3, 4, 5), start=1)],
    ],
    'cycles': 100, 'committed_tokens': 300, 'seconds': 1.0,
    'committed': 3.0, 'cycle_ms': 10.0, 'tps': 300.0,
    'time_source': 'client_decode_wall',
}}
after = deepcopy(before)
candidate = deepcopy(before)
candidate['probe'].update(seconds=.9, cycle_ms=9.0, tps=300/.9)
result = bracket(before, after, candidate, det_pair_stats, exact=True, profile='glue')
assert result['decision'] == 'PASS'
```

`metadata` must match across paired arms; `kind` selects `prose`, `code` or
`structured`, and `thinking=True` additionally selects the `tau` thinking
subgroup. `time_source` must match in each pair. `token_ids` must be nonempty
lists of native, nonnegative integer IDs (booleans are invalid). The two
baseline sequences must match exactly.

`cycles` and `committed_tokens` are positive integer counts. `seconds` is
elapsed decode time in seconds, `cycle_ms` is mean cycle time in milliseconds,
`committed` is mean committed tokens per cycle, and `tps` is tokens per second.
All measurements must be finite and positive. The caller supplies consistent
counts and derived measurements from the same observation interval; this API
does not collect them or check their arithmetic consistency. Throughput ratios
use total `committed_tokens / seconds`, not the supplied per-prompt `tps`.

For a divergent pair, `top_logprobs` has one row per baseline token. At the
first divergence its row must contain five distinct native token IDs and
finite log probabilities in nats. Its first entry must be A's selected token
with the maximum log probability. B's actual selected token must occur in
that row, and `logprob(A) - logprob(B)` must be at most 0.5 nats. An absent
reference or a length-only divergence gives HOLD; malformed top-five evidence
is rejected. Identical sequences do not require a top-five reference.

### Screening profiles

Every applicable condition must hold against both surrounding baselines.
Ratios below are candidate/baseline. CIs use 4,000 deterministic paired
bootstrap draws with sorted endpoints 100 and 3899 (zero based).

| Profile | Meaning and thresholds |
| --- | --- |
| `glue` | Cycle-overhead screen: prose mean saving >= `saving_ms` (default 0.15 ms), saving CI lower bound > 0; code mean per-prompt cycle regression <= 0.5%; each kind's mean committed ratio >= 0.995. |
| `async` | Asynchronous-overhead screen with the same thresholds as `glue`. |
| `trim` | Cycle-trimming screen: every kind's mean per-prompt cycle regression <= 0.5% and mean committed ratio >= 0.995; no minimum prose saving. |
| `tau` | Throughput screen: prose ratio CI lower bound > 1; code and the additional thinking subgroup ratio >= 1. |
| `k4` | Throughput screen: code and structured ratios >= 1.01 with CI lower bound > 1; prose ratio >= 0.997 with CI lower bound >= 0.995. |

Unknown profiles are rejected. With `exact=True`, any candidate token
divergence gives HOLD. Otherwise, the first-divergence near-tie rule above
applies. Invalid observations raise `ValueError` before scoring.

### Applicability and T0

This is an offline screen, not release qualification. It has no minimum
sample count or mandatory category coverage; absent categories are not tested.
The one-prompt example demonstrates the schema, not statistical adequacy.
Thresholds describe the retained profiles and do not establish applicability
to another model or workload.

Screening permits a first-token near tie and checks one reference at the
first divergence. The stricter `compare_t0` draft/target gate requires 30 rows,
three stable repeats, first-token equality, cold-cache reset/hit evidence,
matching requests and two frozen A references reproducing A's full sequence.
Its target-neutral policy requires exact native token equality. A screening
PASS does not substitute for that T0 gate or the separate cycle CLI decision.
