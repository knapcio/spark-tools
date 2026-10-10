# Qwen3.8 Flash Next on one GB10

This is a sanitized, recorded example for one exact TensorFold/recipe/model
cell. It demonstrates the observation and calibration pipeline; it is not a
portable performance promise. The fixture contains no endpoint, hostname,
local path, prompt text, generated text, model weight or absolute server
counter snapshot.

The run used the public MiaAI-Lab TensorFold recipe at the revision recorded in
`preset.json`, a pinned TensorFold native binary/image, the public
`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` snapshot, one NVIDIA
GB10, greedy sampling, thinking off, drafting on, warm prompt cache and 128
output tokens. Each c1 cell contains five train and five holdout requests;
each c2 cell contains the same five paired repetitions on two concurrent
slots. All 120 scored responses succeeded and matched their isolated
Prometheus intervals.

| Workload | Per-request client decode median | Interval aggregate output rate | Client TTFT median | Draft acceptance |
|---|---:|---:|---:|---:|
| prose, short, c1 | 52.54 tok/s | 50.55 tok/s | 0.103 s | 63.6% |
| prose, short, c2 | 43.02 tok/s | 83.09 tok/s | 0.130 s | 63.6% |
| prose, long, c1 | 50.68 tok/s | 43.20 tok/s | 0.455 s | 63.7% |
| prose, long, c2 | 38.96 tok/s | 64.06 tok/s | 0.705 s | 63.7% |
| code, short, c1 | 82.77 tok/s | 78.01 tok/s | 0.106 s | 84.9% |
| code, short, c2 | 67.23 tok/s | 124.49 tok/s | 0.132 s | 83.6% |
| code, long, c1 | 81.90 tok/s | 63.59 tok/s | 0.461 s | 78.7% |
| code, long, c2 | 64.84 tok/s | 90.23 tok/s | 0.920 s | 78.3% |

Per-request client decode rate uses output tokens minus one divided by client
time after the first content event. Interval aggregate output rate uses all
completion tokens divided by the complete isolated wall interval. Under
concurrency they answer different questions and should not be compared as if
they had the same denominator.

`preset.json` summarizes the identities and cells. Each `*.observations.json`
contains normalized per-request telemetry, each `*.interval.json` contains the
matched counter deltas, and each `*.calibration.json` contains a disjoint
train/holdout exact-cell mean with holdout errors. The raw SSE captures and
absolute Prometheus snapshots are deliberately not published.

These results apply only to the recorded hashes, prompt token counts, output
length, cache state, concurrency and settings. Generalization to another
prompt, context length, runtime revision, checkpoint, Spark or kernel is
unavailable. No result attributes time to an individual kernel or establishes
EXL3/EXLR8 support for this checkpoint.
