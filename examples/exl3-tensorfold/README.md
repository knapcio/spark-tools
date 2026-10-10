# EXL3, EXLR8 and TensorFold

Inspect any local EXL3 checkpoint without loading its weights:

```sh
lab exl3 inspect /path/to/model --out reports/exl3.json
```

Inspect an EXLR8 Hugging Face release without downloading weight payloads:

```sh
lab exlr8 inspect-hf kindlingai/glm-5.3-exlr8-k2-k3-k4 \
  --out reports/exlr8.json
```

Inventory an upstream TensorFold checkout or installed package:

```sh
lab tensorfold source-manifest /path/to/TensorFold --require-exl3 \
  --out reports/tensorfold-sources.json
```

Probe a locally running TensorFold server:

```sh
lab tensorfold probe --endpoint http://127.0.0.1:8000/v1 \
  --compare-draft --expect-text SPARK_TOOLS_OK \
  --out reports/tensorfold-probe.json
```

All commands are read-only with respect to the supplied checkpoint, remote
release, source tree and endpoint. The EXLR8 command needs network access but
range-reads headers instead of weight payloads. The probe sends one request per
selected arm. Generated text and the endpoint address are deliberately absent
from its report.
