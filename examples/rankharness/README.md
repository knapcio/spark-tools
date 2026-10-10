# CPU rank fault fixtures

Install the `ranks` extra, then run from the checkout root. Set
`EXTERNAL_OUTPUT` to a fresh directory outside the checkout:

```sh
lab rankharness run examples/rankharness/toy_toggle.py \
  --cases none,invalid_one --repeat 2 --out "$EXTERNAL_OUTPUT/toggle"
lab rankharness run examples/rankharness/rank_local_bug.py \
  --repeat 2 --timeout 0.5 --out "$EXTERNAL_OUTPUT/local-input"
```

Expected: `none` is PASS and `invalid_one` is FALLBACK; the deliberately unsafe
local-input fixture yields DIVERGED for `guarded` and HANG for `skipped`.
Exit 0 means observed verdicts match the fixture's expected verdicts, including
expected failures. Output includes `matrix.json`, per-rank JSON/logs and a
summary. Runtime: about 5-20 seconds per command on macOS/Linux; startup cost
varies with PyTorch. No GPU or endpoint. CPU PyTorch/Gloo and NumPy are needed.

The files are the existing toy-toggle and skipped-collective test fixtures.
The harness reaps local rank process groups after each case. CPU storage and
Python retention tests do not measure GPU allocation. Model-specific worker
adapters are omitted. The smoke script reports SKIP if the optional CPU
transport dependencies are absent.
