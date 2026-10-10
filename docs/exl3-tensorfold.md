# EXL3, EXLR8 and TensorFold adapters

These adapters answer four narrow questions without pretending to qualify a
whole serving stack:

1. Is a checkpoint structurally EXL3 according to its configuration and
   safetensors headers?
2. Is a remote EXLR8 release internally consistent at the repository,
   manifest, LFS identity and sampled-header level?
3. Which TensorFold runtime and kernel sources are present in a checkout or
   installed package, and are EXL3 CUDA/loader sources included?
4. Does a TensorFold endpoint satisfy the expected OpenAI chat contract, and
   does one deterministic request return identical text with drafting disabled
   and enabled?

## Inspect an EXL3 checkpoint

```sh
lab exl3 inspect /models/example-exl3 --out exl3-report.json
```

The command reads root-level `*.safetensors` headers, `config.json`, and an
optional `quantization_config.json`. It validates trellis widths, Hadamard
geometry, input/output scales and `mcg`/`mul1` marker values. It reads only four
payload bytes per codebook marker; weight tensors are not loaded.

A `PASS` means that every detected EXL3 group is structurally readable by the
implemented format contract. It does not mean that the model architecture is
supported by TensorFold or ExLlamaV3, that the quantization is accurate, or
that a CUDA kernel will run.

## Inspect EXLR8 without downloading the weights

```sh
lab exlr8 inspect-hf kindlingai/glm-5.3-exlr8-k2-k3-k4 \
  --header-samples 10 --out exlr8-report.json
```

The command pins the requested revision to an immutable commit, obtains the
repository's LFS SHA-256 identities and sizes, validates every width manifest,
recomputes the EXLR8 `set_id`, checks required expert files and TP/rank index
slices, and range-reads every expert-index header plus a bounded sample of
other safetensors headers. It also checks that index metadata explicitly
declares MCG codebook values rounded to E4M3.

On the public GLM-5.3 release this transferred about 9.6 MB of metadata while
accounting for roughly 833 GB of weight files. A `PASS` does not hash the weight
payloads locally, decode the trellis, validate numerical quality, compile `gr8`
or exercise its TP3-TP7 RoCE runtime. EXLR8 is deliberately separate from the
EXL3 inspector: a stock EXL3 decoder that omits the format's E4M3 rounding is
not compatible.

## Inventory TensorFold sources

```sh
lab tensorfold source-manifest /path/to/TensorFold --require-exl3 \
  --out tensorfold-sources.json
```

The source can be a checkout or an installed layout containing
`src/tensorfold`, `python/tensorfold` or `tensorfold`. The report uses relative
paths and hashes; it does not include the supplied absolute path. Symlinks,
oversized files and unbounded trees fail closed.

This is a provenance and capability inventory, not a build. TensorFold's own
build still chooses architecture flags, PyTorch/CUDA compatibility and linked
extensions.

## Probe a TensorFold endpoint

```sh
lab tensorfold probe \
  --endpoint http://127.0.0.1:8000/v1 \
  --model MODEL_ID \
  --compare-draft \
  --expect-text SPARK_TOOLS_OK \
  --out tensorfold-probe.json
```

The command checks `/v1/models`, sends a bounded deterministic chat request,
and, with `--compare-draft`, compares serial (`draft: false`) and drafted
(`draft: true`) output. Reports contain hashes, byte counts, token usage and
latency, but never generated text or the endpoint address. Authentication may
be supplied through `--token-env`; the token is never written.

One matching request does not establish general numerical equality. Use the
A/B/A-prime harness and a representative preregistered prompt panel for a
release decision.

## Verified boundary

The v0.2 implementation was exercised against a real TensorFold installation
on one DGX Spark. It found the runtime's CUDA, Triton, scheduler, server and
EXL3 source paths, and its deterministic draft-off/draft-on endpoint probe
matched. The EXL3 inspector also passed on the public
[`Honkware/Qwen3.5-0.8B-exl3-3.0bpw`](https://huggingface.co/Honkware/Qwen3.5-0.8B-exl3-3.0bpw)
checkpoint containing mixed 3/4/6-bit trellises. No TensorFold source, model
weight, response content or private endpoint is bundled. The EXLR8 inspector
also passed against the public
[`kindlingai/glm-5.3-exlr8-k2-k3-k4`](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4)
release using metadata and sampled header ranges only.
