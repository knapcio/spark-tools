# Configurable offline MoE geometry

`moereplay` defaults are preserved: conversion without a geometry argument
writes `moe-routes v1`, with 256 experts, top8, four ranks, uint8 data and the
existing manifest hashes. Replay of those traces preserves the v1 report and
the historical GLM-5.3 timing prior. The existing synthetic example and its
saved report remain v1 fixtures.

Explicit geometry creates `moe-routes v2`. Its route files contain unsigned
16-bit expert IDs in little-endian order, including expert 65535. The maximum
expert count is 65536; top_k must be in `[1,num_experts]`. Geometry contains:

```json
{"num_experts":512,"top_k":4,"world_size":8,"expert_bytes":null}
```

`expert_bytes` is optional. When supplied, it is the full unsharded packed
weight and scale footprint for one expert, in bytes. It must divide evenly
across each requested TP layout. When absent, footprint outputs are null and
marked unavailable; the GLM weight dimensions are never inferred. Geometry is
immutable in the Python API and has a SHA-256 fingerprint of its canonical
JSON representation, including `expert_bytes` (null when unspecified).

List the presets and their provenance with `moereplay presets`. `glm53-legacy`
records the legacy geometry, historical weight footprint and limited TP4
timing provenance. `synthetic-small` (8/top2/two ranks) and `synthetic-wide`
(512/top4/eight ranks) are geometry-only fixtures. Neither establishes model,
kernel or serving support. All v2 traces require explicit geometry-bound
fits for timing, including a v2 trace using the legacy geometry preset.

Convert a caller-supplied schema-1 capture with routes matching that geometry:

```sh
moereplay convert-schema1 capture.jsonl --source-status assumed \
  --preset synthetic-wide --model synthetic-model --out route-output
moereplay check route-output --preset synthetic-wide
moereplay replay route-output --layout TP8 --layout EP2xTP4 --layout EP8 \
  --bootstrap 200
```

`--geometry` accepts a JSON object or a geometry JSON file and is mutually
exclusive with `--preset`. Both conversion commands support it. On `check`
and `replay`, the option asserts equality with the stored geometry; it does
not relabel a trace. A capture containing a `geometry` field must agree with
the conversion geometry. Converters default to `unspecified-model` for
explicit geometry unless `--model` is supplied. The conversion source status
describes the capture, independently of a preset's provenance.

Layout spellings are `TPn`, `EPn`, and `EPnxTPn`. The Python API also accepts
`Layout(name, ep, tp)` objects or `{name,ep,tp}` mappings. EP times TP must
equal world_size; EP cannot exceed num_experts. Uneven expert counts use
balanced capacities that differ by at most one expert. Default layouts cover
valid world-size factors. The first selected layout is the comparison
baseline. `--layout` selects generic v2 analysis even for a v1 trace.

Without a fit, v2 replay still produces held-out frequencies, placements,
per-rank distinct counts, routed occupancy and union-growth analysis.
Timing and timing deltas are null, with status `unavailable`. No historical
TP4 prior or EP bytes scaling is transferred to a new geometry. Collective
timing and runtime compatibility remain unknown. Generic reports do not
export cyclesim deltas because geometry does not establish compatibility
with that simulator's runtime model.

A supplied timing file must bind the complete geometry and its fingerprint:

```json
{
  "schema":"moe-cost-fit v2",
  "geometry":{"num_experts":8,"top_k":2,"world_size":2,"expert_bytes":null},
  "geometry_fingerprint":"<SHA256 of canonical geometry JSON>",
  "layouts":[{"name":"TP2","ep":1,"tp":2},{"name":"EP2","ep":2,"tp":1}],
  "per_layout_M":[{
    "layout":"TP2","M":2,"a_us":2,"b_us_per_distinct_local_expert":3,
    "status":"estimated","points":[{"D":2},{"D":4}]
  }]
}
```

Each coefficient line is bound to layout and physical row count M. Geometry,
fingerprint or layout mismatch is rejected. Missing layout/M cells stay
unavailable rather than falling back to another fit or the legacy prior.
Compute the fingerprint with `Geometry.from_dict(data).fingerprint` from
`moereplay.geometry`. The numerical example above is synthetic, not a
measurement. Matched local route hash, local slots, per-row occupancy,
expert multiplicities, duplicate-free routes, measured status and declared
matched-route validation are still required for calibration eligibility.
Timing estimates and sampling intervals do not establish serving speed.

External training traces must match model, layers and the complete geometry;
request IDs must be disjoint from the test trace. Route-stage, holdout,
padding and decode step checks remain in effect for v2.
