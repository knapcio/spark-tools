# Offline experiment manifests and reports

`lab experiment check MANIFEST` verifies explicit local evidence and all requested
adapters. `lab experiment report MANIFEST --out report.json` exports the same
checks plus safe summaries; `--markdown report.md` adds a readable view. Both
report destinations are create-only. Neither command executes a workload,
contacts an endpoint, launches a compiler/container, discovers files, or fetches
missing evidence. A failed tool outcome can coexist with integrity PASS.

The [small example](../examples/experiment/manifest.json) has a fixture static
kernel report and a simulated prediction. Run it with:

```sh
lab experiment check examples/experiment/manifest.json
lab experiment report examples/experiment/manifest.json --out report.json --markdown report.md
```

## Version 1 contract

The root contains exactly these required fields:

| Field | Contract |
| --- | --- |
| `schema`, `version` | `spark-experiment`, integer `1` |
| `identity` | Object containing opaque experiment `id` |
| `model`, `runtime` | Objects with opaque `id`, `revision`; optional `sha256` |
| `arms` | 1-16 objects with unique `id` and `config_sha256` |
| `artifacts` | 1-64 explicit references described below |
| `analyses` | 0-32 adapter requests with unique `id`, `tool`, `inputs` |
| `evidence` | Optional list of up to 128 declarations |

Opaque identifiers use 1-96 ASCII letters, digits, dots, underscores or hyphens,
starting with a letter or digit. Use public aliases and `unknown` for identities
you cannot establish. URLs, IP-address identities, common credential prefixes,
raw configuration objects and free-form prose are refused. All SHA-256 values
are 64 lowercase hexadecimal characters. Identity and per-arm configuration
hashes are operator declarations; the checker does not load model weights or
infer running configurations from them.

Each artifact has unique `id` and `path`, plus `kind`, `sha256`, `schema`, `origin`,
and optionally `arm` referencing a declared arm. `path` is relative to the
manifest's directory. Absolute paths, traversal, empty segments, backslashes,
colon-containing paths, symlinks and repeated file identities are refused.
References are opened through directory descriptors without following symlinks;
reads verify the same inode, size and timestamps before/after. No sibling files
or glob patterns are discovered. SHA-256 validates the full referenced bytes.

Origins are separate from epistemic status: `fixture`, `recorded`, `simulation`,
or `static`. A fixture result remains a fixture even when an original tool uses
the word PASS or measured. Recorded means the operator supplied a saved record;
the checker cannot authenticate its collection or prove GPU execution.

These schema strings identify the adapter's accepted existing-tool shape. They
are not added to or substituted for the original artifact's native schema.

| Kind | Adapter schema |
| --- | --- |
| `abharness.prereg` | `abharness-prereg-v1` |
| `abharness.panels` | `abharness-panels-v1` |
| `qpanel.prereg` | `qpanel-prereg-v1` |
| `qpanel.run` | `qpanel-run-v1` |
| `prereg.digest` | `sha256-text-v1` |
| `cyclesim.report` | `cyclesim-report-v1` |
| `gb10check.report` | `gb10check-report-v1` |
| `exl3.report` | `exl3-report-v1` |
| `exlr8.report` | `exlr8-report-v1` |
| `tensorfold.source` | `tensorfold-source-v1` |
| `tensorfold.probe` | `tensorfold-probe-v1` |

## Existing decision adapters

An analysis's inputs map role names to artifact IDs. Extra/missing roles,
incorrect artifact kinds, unknown references and repeated input IDs are refused.

| Tool | Input roles | Behavior |
| --- | --- | --- |
| `abharness` | `prereg`, `prereg_digest`, `panels` | Verifies original frozen sources/rule and recomputes paired-cycle decision |
| `qpanel` | `prereg`, `prereg_digest`, `A`, `A2`, `B` | Verifies original frozen sources/rule, arm/item identity, and recomputes null-calibrated item-pass decision |
| `cyclesim` | `report` | Preserves original decisions/status strings and a bounded numeric summary |
| `gb10check` | `report` | Preserves static report/entry statuses, including occupancy estimates |
| `exl3`, `exlr8`, `tensorfold` | `report` | Preserves supplementary status only |

For panel adapters, explicitly list both the preregistration JSON and its existing
`.sha256` sidecar as separate artifacts. The sidecar contents must equal the JSON
artifact digest. The saved source map must exactly match the unchanged installed
tool source files. A copied old plan against changed tool code is rejected.
The JSON's original fixed rule is validated; panel/run `prereg_sha256` must match
the manifest-bound preregistration. No supplied summary PASS is trusted.

`abharness.panels` retains the original `A`, `Aprime`, `B` shape: each contains
three complete 20-row panels, 10 prose and 10 code observations with unique
identities and positive finite cycle/throughput/committed values. The adapter
calls the original fixed paired bootstrap and gates, preserving their values.

Each quality run must declare its original arm (`A`, `A2`, `B`), also specified
in its artifact's `arm`. Runs must be COMPLETE, contain nonempty unique ordered
item keys, match the preregistered count and task keys when present, and carry
Boolean scores/truncations plus string content and response hashes. The
recomputed decision uses the original saved item scores and original null-pass
arithmetic. It does not rescore responses or claim the expected answers are
correct. Raw outputs are used only by the original changed-output comparison.

cyclesim supports existing prediction arrays, validation summaries, profile
reports and numeric comparison summaries. It preserves only recognized original
tool-owned status literals, decisions and named numeric metrics. Unsupported
status strings fail closed; copied arbitrary text cannot enter a public report.
The report does not rerun simulation or turn estimates into measured results.
GB10's STATIC_NO_GPU outcomes remain static even if the saved status is PASS.
Supplementary checkpoint/source/API probes do not qualify throughput, model
quality, kernels or a runtime promotion.

## Evidence and output boundary

Each optional evidence declaration contains unique `id` and `status`, with
optional artifact ID, metric, numeric/Boolean value and opaque unit. Supported
statuses are `measured`, `estimated`, `assumed`, `missing`, `operator-declared`.
Measured/estimated declarations need an artifact. Measured also requires
`recorded` origin; a simulation or fixture cannot be relabeled a measured
hardware observation. Missing declarations cannot have a value. These remain
explicitly marked manifest declarations: the checker does not prove a declared
scalar equals a metric in an arbitrary saved report.

JSON output separates `integrity`, `tool_decisions`, artifact origins and the
five evidence buckets. Original tool statuses remain literal and scoped to their
tool. Identity has its own operator-declared basis. There is no aggregate
promotion verdict, threshold optimizer or winner selection. Check/report return
0 when integrity and adapter validation succeed, even if a tool's decision is
FAIL/KILL/HOLD. Invalid manifests, drift or unsafe I/O return argument-error code
2 and sanitized diagnostics.

Reports omit artifact paths, absolute paths, endpoints, raw configuration,
prompts, response content, task/item identifiers, saved errors and arbitrary
metadata. JSON/Markdown output is assembled from fixed allowlists; optional
operator metadata must use safe public aliases. Do not put secrets into IDs.
Only output files explicitly requested by the operator are created.

## Resource limits

The manifest is at most 256 KiB; each artifact at most 8 MiB, and referenced
artifact bytes total at most 32 MiB. JSON has at most 32 nesting levels and
200,000 values per document; duplicate keys and all nonfinite numbers (including
overflowing exponents) are refused. Panel shapes are fixed; quality panels are
at most 10,000 items. cyclesim arrays have at most 4,096 rows and summaries at
most 8,192 metrics/statuses. GB10 entry arrays have at most 4,096 rows.

Large raw profiler traces, model shards and directory inventories must be
reduced with their existing tools first. The manifest references the resulting
bounded local summary and its declared origin. No missing evidence is inferred.
