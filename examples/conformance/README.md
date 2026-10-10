# Dense decoder conformance examples

These synthetic examples assert **dense decoder conformance only**. They contain
no weights, private endpoints or GPU execution evidence. Run from an installed
spark-tools package or the repository root:

```sh
lab conformance cpu
lab conformance fixtures
lab conformance manifest --runner tensorfold-cuda
lab conformance manifest --runner gr8-dense
lab conformance cuda
```

`expected-cpu.json` is a deterministic stdlib CPU PASS. `fixtures.json` records
27 EXL3 tiles and six EXLR8 rotated-domain 256x256 matrices with exact hashes.
The two runner JSON files are PLAN_ONLY; neither runs code or reports GPU PASS.
`cuda` exits 3 (UNAVAILABLE), because no local GPU API adapter is bundled.
`verified-runs.json` separately records completed external pinned runs: 4/4
`gr8` CPU tests and 62/62 TensorFold CUDA tests on one GB10.

See [the runbook](../../docs/decoder-conformance.md) for exact byte contracts,
separate matmul tolerance, immutable upstreams, credits and the standalone GB10
handoff. Model/quality/collective and whole-engine correctness remain
unverified by these examples.
