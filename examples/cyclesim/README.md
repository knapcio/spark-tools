# Decode-cycle simulator

This is the existing pure-Python cost model and profile tooling, packaged with
an anonymised GLM-5.3 TP4 calibration. It estimates decode-cycle costs and prefix
acceptance, with bounded edits to measured or assumed budgets. It runs on
macOS and Linux without a GPU, model download or endpoint.

Run from the checkout root after `pip install -e .`:

```sh
cyclesim predict --config examples/cyclesim/calibration.json
cyclesim whatif --config examples/cyclesim/calibration.json \
  --kernel-ms draft_graph=-1.50970368
python3 -m examples.cyclesim.compare
python3 -m examples.cyclesim.validate
python3 -m examples.cyclesim.smoke
```

Without an installation, replace `cyclesim` with `python3 -m cyclesim`.
Runtime: under one second per command; the complete smoke check takes about
one second. No GPU. Expected baseline: 73.482 ms per cycle, 2.322 committed
tokens per cycle and 31.59 tokens/s per stream. `expected-prediction.json`
contains the full output. The prose FP8 projection hypothesis predicts +2.10%;
the glue-lite hypothesis predicts +0.41%. The deterministic comparison is in
`expected-comparison.json`.

Every prediction, sensitivity row and target row includes a human-readable
`calibration_label`, the calibration schema and a canonical SHA-256 digest.
The digest is stable across JSON whitespace and key ordering, so archived runs
can be tied back to their exact calibration content without recording a local
file path. Predictions made with an ingested kernel table also include its
canonical digest.

The existing projection example assumes two 6144 by 12288 projections per
cycle and 100 GB/s effective bandwidth for BF16 and FP8. FP8 includes two-byte
per-output-channel scales. The resulting 1.50970368 ms saving is applied to
the existing draft budget; it is an engineering assumption, not a measured
kernel cost. Glue-lite assumes a 0.300 ms reduction in host/sampling time.
`fp8-projection.json` and `glue-lite.json` record those hypotheses.

The retained measured comparison fixture reports FP8 projection gains of
3.6-3.9% prose and 5.0-5.4% code, and glue-lite gains of 0.43% prose and 1.0%
code. Its source references are in `measured.json`. These observations and
this baseline used different configurations. Compare each lever separately;
do not add percentages or infer a release decision.

`holdout.json` retains six anonymised timing observations from three protocols
in the existing calibration report. The final two use the alternate NVFP4
head; the other four use BF16. The original fit/holdout membership is retained
using anonymous prompt labels. `expected-validation.json` reports errors
separately by protocol. Across these six values the maximum absolute cycle
error is 3.71%; that is not a universal error bound. Historical throughput
transfer errors in the calibration reach 19.39% for single-stream and 17.85%
for concurrent observations. Concurrency, routing and draft-pass prices remain
assumptions where matched observations are unavailable.

The model accepts the source schema 5, supports c=1/2/4, and retains its
measured policy/head restrictions. `--c 1,2,4 --domain prose,code` produces a
matrix. `sensitivity` and `targets` retain the original analyses. New policies,
confidence thresholds or reduced single-stream draft lengths need identified
observations. The shape ledger retains the source architecture and layer
counts; it is not a general model architecture parser.

The existing regression fixtures are also runnable examples:

- [Profile fixture](profile-fixture/README.md): four synthetic rank traces,
  graph descriptors, counters and hashed companions; expected 4 ranks and
  16 kernel rows.
- [Companion fixture](companion-fixture/README.md): the same validation contract
  without profiler traces; expected 0 ranks and 0 kernel rows.

Ingestion uses the original graph segmentation, interval union and ordinal
rank comparison. Kernel sums can exceed busy time when streams overlap.
Traffic, bandwidth and cache-residency estimates retain their original limits.
Use `python3 -m cyclesim.profile --help` for saved-file ingestion,
`python3 -m cyclesim.build_ledger --help` for explicit shape configuration,
`python3 -m cyclesim.calibrate --help` for fitting normalised observations, and
`python3 -m cyclesim.validate_profiles --help` for saved reference comparisons.
The capture planner only writes local request bodies or hashes local files.
It sends no requests. Its model name is a placeholder.
