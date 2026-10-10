# Credits

| Work used | Author / source | Licence and scope |
|---|---|---|
| Decode-cycle accounting, prefix-survival arithmetic, interval union, file/counter utilities and CPU replay labs | knapcio | Apache-2.0 where marked in the source; original MIT recipe notice also retained |
| Static compiler adapter, resource parser, instruction checks and CPU occupancy wrapper | knapcio | MIT; headers and licence retained |
| A/B/A' comparisons, bracket screens, native trajectory pair statistics, preregistration, first-divergence near-tie gate and null-calibrated quality rule | knapcio | Existing decision/HTTP logic with packaging adapters, Apache-2.0 |
| Runtime, scheduler and benchmark-output contracts used by the labs and versioned observation adapter | [vLLM contributors](https://github.com/vllm-project/vllm) | Apache-2.0; pinned format adapter, no runtime source bundled |
| Packed expert weights and FP8 projection layout used by the calibrated example | [Elias Frantar / IST-DASLab](https://github.com/IST-DASLab/marlin), Neural Magic and vLLM contributors | Apache-2.0; no kernel source bundled |
| Sparse attention metadata and static compilation context | [FlashInfer contributors](https://github.com/flashinfer-ai/flashinfer) | Apache-2.0; no source bundled |
| Ahead-of-time compilation API | [Triton contributors](https://github.com/triton-lang/triton) | MIT; optional external compiler |
| GB10 target, resource limits and CPU occupancy calculator API | [NVIDIA CUDA contributors](https://docs.nvidia.com/cuda/) | NVIDIA toolkit terms; installed headers are not bundled |
| Optional CPU Gloo rank transport | [PyTorch contributors](https://github.com/pytorch/pytorch) | BSD-3-Clause; optional dependency |
| Optional rank retention arrays | [NumPy contributors](https://github.com/numpy/numpy) | BSD-3-Clause; optional dependency |
| EXL3 trellis format, tensor naming and codebook markers | [Turboderp and ExLlamaV3 contributors](https://github.com/turboderp-org/exllamav3) | MIT; format-compatible header inspection, no runtime or kernels bundled; notice retained |
| EXLR8 v2 manifests, E4M3-rounded trellis contract and `gr8` loader fixture layout | [Kindling AI `gr8` contributors](https://github.com/kindlingai/gr8) | MIT; independent inspection/conformance fixtures, no runtime source or weights bundled |
| TensorFold source layout, EXL3 decoder tests, OpenAI endpoint, telemetry and draft request contract | [Ash Hart and TensorFold contributors](https://github.com/ashhart/TensorFold) | Apache-2.0; independent adapters and conformance fixtures, no TensorFold source bundled |
| Single-Spark Qwen Flash Next TensorFold recipe used for the exact-cell example | [MiaAI-Lab contributors](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) | Recipe terms apply separately; public configuration identities and sanitized measurements only, no recipe source or weights bundled |
| Public EXLR8 release used for metadata and bounded-header validation | [kindlingai/glm-5.3-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4) | GLM-5.3/model-card terms apply separately; no weight payload bundled |
| Public checkpoint used for header-only EXL3 compatibility validation | [Honkware/Qwen3.5-0.8B-exl3-3.0bpw](https://huggingface.co/Honkware/Qwen3.5-0.8B-exl3-3.0bpw), Qwen and ExLlamaV3 contributors | Model and checkpoint terms apply separately; validation only, no weights or headers bundled |
| GLM-5.3 shapes and native draft architecture | [Z.ai](https://huggingface.co/zai-org/GLM-5.3) | Model terms apply separately; dimensions and anonymised timing numbers only |
| Calibration recipe and mixed-precision context | [tonyd2wild / Tony](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark), [Tech2wild](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix) | Recipe and checkpoint terms apply separately; no source or weights bundled |
| Dynamic speculative lookahead context | Huang, Guo and Wang (SpecDec++); Mamou et al. (DISCO) | Research ideas only, no paper code copied |
| Speculative verification context | Leviathan, Kalman and Matias; Chen et al. | Research ideas only, no paper code copied |
| Bounded prefill scheduling context | Agrawal et al., Sarathi-Serve, OSDI 2024 | Research ideas only, no paper code copied |
| Sparse attention design context for the calibration | [CosmicRaisins](https://github.com/CosmicRaisins/glm-5.2-gb10); [Matt Mastracci](https://github.com/mmastrac/glm-5.3-flash-4x-gx10) | CosmicRaisins upstream Apache-2.0; discussion ideas only |
| Community investigation of native draft execution | keys / @u1tra_instinct; [Ash Hart and TensorFold contributors](https://github.com/ashhart/TensorFold); jayleaton | Ideas and measured context only; no code copied or relicensed |
| Collective transport context for the calibration | Luke Alonso / @lukealonso; Jason Cook / @original-el8; Local Inference Lab; rhys101; tonyd2wild | b12x lineage Apache-2.0; no transport source bundled |
| Benchmark interpretation context | sparkDash contributors; Alex Ellis and RigMark contributors | sparkDash MIT; benchmark observations only, no prompt bank or benchmark code bundled |

These community attributions are preserved from the source credits. Unrelated
loader, cache, memory-sharing and serving components were excluded and are not
dependencies of this project. Referencing a project does not relicense it.

The portable source repositories include an MIT recipe licence alongside
Apache-2.0 tool headers. The MIT notice is retained in
[LICENSES/MIT-portable-tools.txt](LICENSES/MIT-portable-tools.txt).
The original static-checker MIT notice is retained separately in
[LICENSES/MIT-static-checker.txt](LICENSES/MIT-static-checker.txt). The project
retains the ExLlamaV3 MIT notice in
[LICENSES/MIT-exllamav3.txt](LICENSES/MIT-exllamav3.txt). The project
and its packaging changes are Apache-2.0; external dependencies, models and
toolchains keep their own terms. No third-party runtime or kernel is bundled.

The scheduler fast policy and inert request fixture are knapcio CPU lab code
under Apache-2.0, informed by the vLLM scheduler contract credited above. The
rank fault matrices, bounded process cleanup and CPU retention probes are
knapcio code under Apache-2.0. The HTTP transport and hash-bound
preregistration pattern retain knapcio's existing panel logic (Apache-2.0).
Research ideas have no source-code licence dependency here: no paper code or
text is redistributed. Model, recipe and community context above is retained
from the source credit files; no weights or unrelated runtime code is copied.
