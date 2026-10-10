# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 knapcio
"""One-time conversion; only convert_pt imports CPU torch."""
import gzip
import json
from pathlib import Path
from moereplay.routes import Writer, sha256, integer


def convert_pt(path, out, layer=3, prompt_rows=2048, model='GLM-5.3'):
    # Deliberately bounded to the existing small fixture; not a general tensor loader.
    if Path(path).stat().st_size > 512 * (1 << 20):
        raise ValueError('PT conversion capped at 512 MiB; extract route-only fixture first')
    import torch
    d = torch.load(path, map_location='cpu', weights_only=True)
    ids = d['ids']
    if ids.ndim != 2 or ids.shape[1] != 8 or ids.shape[0] > 65536:
        raise ValueError('expected bounded ids [rows,8]')
    if ids.dtype.is_floating_point or ids.dtype == torch.bool or ids.dtype.is_complex:
        raise ValueError('ids must have integer dtype')
    integer(prompt_rows, 'prompt_rows', 1)
    if ids.shape[0] % prompt_rows:
        raise ValueError('rows must divide prompt_rows')
    w = Writer(out, [layer], model, dict(status='measured', kind='prefill-fixture',
               file=Path(path).name, sha256=sha256(path), note='not decode; consecutive prompt rows'), 'prefill-proxy')
    try:
        for p in range(ids.shape[0] // prompt_rows):
            rows = [dict(request='prompt-%d' % p, position=i, padding=False, phase='prefill', k=0, concurrency=1)
                    for i in range(prompt_rows)]
            w.append(p, rows, {layer: ids[p*prompt_rows:(p+1)*prompt_rows].tolist()})
        return w.finish()
    except BaseException:
        w.abort()
        raise


def convert_schema1(path, out, model='GLM-5.3', status='measured'):
    opener = gzip.open if str(path).endswith('.gz') else open
    w = None
    try:
        with opener(path, 'rt') as f:
            # One step per line; no full-trace load, no partial-tail forgiveness.
            while True:
                line = f.readline(64*(1 << 20)+1)
                if not line:
                    break
                if len(line) > 64*(1 << 20):
                    raise ValueError('capture step exceeds 64 MiB; stream smaller step records')
                if not line.strip():
                    raise ValueError('blank record in capture')
                record = json.loads(line)
                if record.get('schema') != 1 or 'accepted' not in record and 'accept_flags' not in record:
                    raise ValueError('complete capture-hook schema 1 required')
                if w is None:
                    w = Writer(out, record['layers'], model, dict(status=status, kind='capture-hook-schema-1',
                               file=Path(path).name, sha256=sha256(path),
                               route_stage=record.get('route_stage','unknown')), 'decode-verify')
                if record.get('route_stage','unknown') != w.source['route_stage']:
                    raise ValueError('route capture stage changed within trace')
                if record['layers'] != w.layers or record['physical_rows'] != len(record['rows']):
                    raise ValueError('inconsistent layers/physical rows')
                if len(record['experts']) != len(w.layers):
                    raise ValueError('all layer routes required')
                rows = []
                for r in record['rows']:
                    if type(r.get('padding')) is not bool:
                        raise ValueError('schema 1 rows require explicit padding bool')
                    padding = r['padding']
                    rows.append(dict(request=None if padding else r['request_id'],
                                     position=None if padding else r['position'], padding=padding,
                                     phase='decode', k=record['k'], concurrency=record['concurrency']))
                k = integer(record['k'], 'k')
                requests = {r['request'] for r in rows if not r['padding']}
                accepted = record.get('accepted')
                flags = record.get('accept_flags')
                for values in (accepted, flags):
                    if values is not None and (not isinstance(values, dict) or set(values) != requests):
                        raise ValueError('acceptance metadata must cover every real request')
                if accepted is not None:
                    if any(type(v) is not int or not 0 <= v <= k for v in accepted.values()):
                        raise ValueError('accepted prefix must be in [0,K]')
                if flags is not None:
                    for request, values in flags.items():
                        if not isinstance(values, list) or len(values) != k or any(type(v) is not bool for v in values):
                            raise ValueError('accept_flags requires K bools per request')
                        prefix = 0
                        for flag in values:
                            if not flag:
                                break
                            prefix += 1
                        if accepted is not None and accepted[request] != prefix:
                            raise ValueError('accepted prefix disagrees with accept_flags')
                if accepted is None and flags is None:
                    raise ValueError('acceptance metadata missing')
                w.append(record['step'], rows, dict(zip(w.layers, record['experts'])))
        if w is None:
            raise ValueError('empty capture')
        return w.finish()
    except BaseException:
        if w is not None:
            w.abort()
        raise
