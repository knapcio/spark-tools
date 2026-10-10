# Synthetic expert routes

After `pip install -e .`, run from the checkout root:

```sh
python3 examples/moereplay/run.py
```

Expected: four held-out decode steps, one layer and 16 rows, with TP4,
EP2xTP2 and EP4 comparisons, contiguous/round-robin/greedy placements,
rank weight bytes, estimated costs and paired saving intervals. `expected.json`
is the complete deterministic report. Runtime: under one second on macOS/Linux.
No GPU, endpoint, PyTorch or model download is needed.

`routes.jsonl` and `training.jsonl` are the existing synthetic replay test
fixture written in the existing schema-1 interchange format. Each row routes
to eight of 256 experts. Training and test request identities are disjoint.
The runner converts both inputs into temporary hashed route directories and
removes them after replay. To use the installed CLI directly, set
`EXTERNAL_OUTPUT` to a fresh directory outside the checkout:

```sh
moereplay convert-schema1 examples/moereplay/routes.jsonl \
  --source-status assumed --model example-model --out "$EXTERNAL_OUTPUT/routes"
moereplay convert-schema1 examples/moereplay/training.jsonl \
  --source-status assumed --model example-model --out "$EXTERNAL_OUTPUT/training"
moereplay check "$EXTERNAL_OUTPUT/routes" --decode-only --min-steps 4
moereplay replay "$EXTERNAL_OUTPUT/routes" \
  --train-trace "$EXTERNAL_OUTPUT/training" --bootstrap 200
```

The preserved model supports fixed 256-expert/top-8 geometry and four ranks.
The packed weight dimensions and timing prior are specific to that example
shape; this port does not generalize the model. `moereplay/prior.json` contains
only anonymised timing numbers and neutral source labels. `--cost-fit` accepts
caller-supplied per-layout fits. Communication and cache effects are omitted.
Synthetic routes remain assumed evidence and cannot export qualified runtime
deltas. Estimates and confidence intervals do not establish production speed.
