# SPDX-License-Identifier: Apache-2.0
"""Run the existing synthetic decode fixture without retaining binary outputs."""
import json
from pathlib import Path
import tempfile

from moereplay.converters import convert_schema1
from moereplay.replay import replay
from moereplay.routes import Trace


HERE = Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory(prefix='moe-example-') as directory:
        root = Path(directory)
        for name in ('routes', 'training'):
            convert_schema1(HERE / (name + '.jsonl'), root / name,
                            model='example-model', status='assumed')
        with Trace(root / 'routes') as routes, Trace(root / 'training') as training:
            result = replay(routes, bootstrap_samples=200, train_trace=training)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
