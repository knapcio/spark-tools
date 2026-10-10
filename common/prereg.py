# SPDX-License-Identifier: Apache-2.0
"""Existing create-only, hash-bound preregistration pattern with caller configuration."""
import hashlib
from pathlib import Path
from common.config import load


def freeze(config, path, sources=()):
    import json
    value = {'schema': 1, 'configuration': config,
             'sources': {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}}
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with Path(path).open('xb') as stream:
        stream.write(data)
    with Path(path).with_suffix('.sha256').open('x') as stream:
        stream.write(hashlib.sha256(data).hexdigest() + '\n')
    return value


def verify(path, sources=()):
    path = Path(path)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if path.with_suffix('.sha256').read_text().strip() != sha:
        raise ValueError('preregistration digest drift')
    value = load(path)
    current = {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    if value.get('schema') != 1 or value.get('sources') != current:
        raise ValueError('source/preregistration drift')
    return value['configuration'], sha
