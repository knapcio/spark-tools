# SPDX-License-Identifier: Apache-2.0
"""Existing create-only, hash-bound preregistration pattern with caller configuration."""
import hashlib
from pathlib import Path
from common.atomic_publish import publish_bytes, validate_bytes
from common.config import load


def freeze(config, path, sources=()):
    import json
    value = {'schema': 1, 'configuration': config,
             'sources': {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}}
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    path = Path(path)
    digest_path = path.with_suffix('.sha256')
    if digest_path == path:
        raise ValueError('preregistration path must not end in .sha256')
    digest = (hashlib.sha256(data).hexdigest() + '\n').encode()

    # Check known conflicts before creating either record. Publication remains
    # race-safe below; this preflight merely avoids an unnecessary sidecar when
    # an incompatible main record already exists.
    for target, expected in ((path, data), (digest_path, digest)):
        try:
            validate_bytes(target, expected)
        except FileNotFoundError:
            pass

    # The digest is recoverable metadata. Publish it first and the validated
    # preregistration last, so the main record is the pair's commit point.
    publish_bytes(digest_path, digest, mode=0o666)
    publish_bytes(path, data, mode=0o666)
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
