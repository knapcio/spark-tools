# Endpoint adapters

`configs/endpoint.json` and `configs/qpanel-endpoint.json` use the placeholder
`http://127.0.0.1:8000` and `example-model`. Supply your endpoint, model and
request bodies. Bases with or without a final `/v1` are accepted. If the
endpoint needs authentication, set the environment variable named by
`token_env`; the default is `OPENAI_API_KEY`. Never put tokens in files.

The request adapter makes a standard non-streaming chat completion request:

```sh
abharness request --config configs/endpoint.json --out "$EXTERNAL_OUTPUT/response.json"
```

The existing quality HTTP transport can run a caller-supplied exact-answer
panel. Freeze the configuration before collecting any arm:

```sh
qpanel preregister --config configs/qpanel-endpoint.json \
  --prereg "$EXTERNAL_OUTPUT/quality-plan.json"
qpanel run --prereg "$EXTERNAL_OUTPUT/quality-plan.json" \
  --arm A --out "$EXTERNAL_OUTPUT/A.json"
```

Collect A2 and B with the same plan, then call `qpanel decide` with the three
outputs. Configure each arm's endpoint in advance. A2 must use the same
baseline model under an externally arranged shape/load perturbation; keep
foreground bodies and scoring fixed. B must identify the candidate model or
runtime. A response ending with `length` is recorded as truncated and scored
by the exact-answer checker, following the existing rule. Network errors,
missing choices or invalid finish reasons stop collection.

The null threshold is max(absolute null net item loss, binomial standard error
in items). It is a descriptive tolerance, not a general statistical proof of
quality equivalence. The package does not verify deployment identity, enforce
exclusive traffic, create background load or change a server configuration.
Preregistration detects accidental configuration/source drift; it is not a
signature or protection against deliberate editing of saved results.

For the cycle harness, freeze `configs/ab-fixture.json` with `abharness
preregister`; collect the existing three-panel numeric schema in
`examples/abharness/panels.json` using your own instrumentation. Bind the
plan's SHA256 as `prereg_sha256` before deciding. Those measurements are
**decode cycles and committed tokens**, not HTTP request latency. T0 needs
native IDs, repeat stability, cold-cache evidence and saved top-five A
references. The portable HTTP helper alone cannot produce these observations.
