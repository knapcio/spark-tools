# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 knapcio
"""Versioned, hashed, streaming route interchange (no tensor dependencies)."""
import hashlib
import itertools
import json
import mmap
import os
from pathlib import Path
import shutil
import tempfile
import struct
from moereplay.geometry import LEGACY_GEOMETRY, load_geometry

SCHEMA = 'moe-routes v1'
SCHEMA_V2 = 'moe-routes v2'
MAX_STEP_ROWS = 65536


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def integer(x, label, minimum=0):
    if type(x) is not int or x < minimum:
        raise ValueError(label + ' must be an integer >= ' + str(minimum))
    return x


def valid_ids(row, geometry=None):
    if geometry is not None:
        geometry = load_geometry(geometry)
        if len(row) != geometry.top_k or any(type(e) is not int or not 0 <= e < geometry.num_experts for e in row) or len(set(row)) != geometry.top_k:
            raise ValueError('routes must contain top_k distinct logical expert IDs within geometry')
        return struct.pack('<%dH' % geometry.top_k, *row)
    if len(row) != 8 or any(type(e) is not int or not 0 <= e < 256 for e in row) or len(set(row)) != 8:
        raise ValueError('routes must contain eight distinct logical expert IDs in [0,256)')
    return bytes(row)


def valid_row(row, i):
    integer(row.get('row'), 'row')
    if row.get('row') != i:
        raise ValueError('row index must be contiguous from zero')
    integer(row.get('step'), 'step')
    if type(row.get('padding')) is not bool:
        raise ValueError('padding must be explicit bool')
    if row['padding']:
        if row.get('request') is not None or row.get('position') is not None:
            raise ValueError('padding request and position must be null')
    else:
        if not isinstance(row.get('request'), str) or not row['request']:
            raise ValueError('real row needs nonempty request id')
        integer(row.get('position'), 'position')
    if row.get('phase') not in ('decode', 'prefill'):
        raise ValueError('phase must be decode or prefill')
    integer(row.get('k'), 'k')
    integer(row.get('concurrency'), 'concurrency', 1)


class Writer:
    """Create-only atomic directory; holds at most one step and open layer files."""
    def __init__(self, out, layers, model, source, semantics, geometry=None):
        self.out = Path(out)
        if self.out.exists():
            raise ValueError('output directory already exists')
        if not layers or len(set(layers)) != len(layers):
            raise ValueError('layers must be nonempty and unique')
        for layer in layers:
            integer(layer, 'layer')
        if not model or not isinstance(source, dict) or source.get('status') not in ('measured', 'assumed'):
            raise ValueError('model/source with measured or assumed status required')
        if semantics not in ('decode-verify', 'prefill-proxy'):
            raise ValueError('unsupported step semantics')
        self.geometry = load_geometry(geometry) if geometry is not None else LEGACY_GEOMETRY
        self.schema = SCHEMA_V2 if geometry is not None else SCHEMA
        self.extension = 'u16le' if geometry is not None else 'u8'
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix='.moe-routes-', dir=str(self.out.parent)))
        self.layers = list(layers)
        self.model, self.source, self.semantics = model, source, semantics
        self.files = {l: (self.tmp / ('layer-%d.%s' % (l, self.extension))).open('wb') for l in layers}
        self.index = (self.tmp / 'rows.jsonl').open('w')
        self.count = 0
        self.last_step = -1

    def append(self, step, rows, experts):
        integer(step, 'step')
        if step <= self.last_step or not 0 < len(rows) <= MAX_STEP_ROWS:
            raise ValueError('steps must increase; nonempty bounded steps required')
        if set(experts) != set(self.layers):
            raise ValueError('all layers required on each step')
        normalized = []
        for offset, row in enumerate(rows):
            r = dict(row, row=self.count + offset, step=step)
            valid_row(r, self.count + offset)
            normalized.append(r)
        # Validate all data before emitting any of this step.
        buffers = {}
        for layer in self.layers:
            if len(experts[layer]) != len(rows):
                raise ValueError('layer shape mismatch')
            buffers[layer] = b''.join(valid_ids(r, self.geometry if self.schema == SCHEMA_V2 else None) for r in experts[layer])
        validate_step(normalized, self.semantics)
        for l, b in buffers.items():
            self.files[l].write(b)
        for r in normalized:
            self.index.write(json.dumps(r, separators=(',', ':')) + '\n')
        self.count += len(rows)
        self.last_step = step

    def finish(self):
        if not self.count:
            raise ValueError('empty trace')
        self.close()
        data = [{'layer': l, 'file': 'layer-%d.%s' % (l, self.extension),
                 'shape': [self.count, self.geometry.top_k],
                 'dtype': 'uint8' if self.schema == SCHEMA else 'uint16'} for l in self.layers]
        for item in data:
            item['sha256'] = sha256(self.tmp / item['file'])
        manifest = dict(schema=SCHEMA, model=self.model, layers=self.layers, num_experts=256,
                        top_k=8, rows=self.count, step_semantics=self.semantics,
                        row_index=dict(file='rows.jsonl', sha256=sha256(self.tmp / 'rows.jsonl')),
                        data=data, source=self.source)
        if self.schema == SCHEMA_V2:
            for item in data:
                item['byte_order'] = 'little'
            manifest.update(schema=SCHEMA_V2, geometry=self.geometry.to_dict(),
                            geometry_fingerprint=self.geometry.fingerprint,
                            num_experts=self.geometry.num_experts, top_k=self.geometry.top_k, data=data)
        (self.tmp / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        # create-only: refuse a destination created during conversion as well.
        if self.out.exists():
            raise ValueError('output appeared during conversion')
        os.rename(self.tmp, self.out)
        return manifest

    def close(self):
        self.index.close()
        for f in self.files.values():
            f.close()

    def abort(self):
        self.close()
        shutil.rmtree(self.tmp, ignore_errors=True)


def validate_step(rows, semantics):
    if len(rows) > MAX_STEP_ROWS:
        raise ValueError('step exceeds row bound')
    first = rows[0]
    for r in rows:
        if any(r[k] != first[k] for k in ('phase', 'k', 'concurrency', 'step')):
            raise ValueError('inconsistent step metadata')
    real = [r for r in rows if not r['padding']]
    if semantics == 'decode-verify':
        if first['phase'] != 'decode' or len(real) != first['concurrency'] * (first['k'] + 1):
            raise ValueError('decode real rows must equal c*(K+1)')
        groups = {}
        for r in real:
            groups.setdefault(r['request'], []).append(r['position'])
        if len(groups) != first['concurrency']:
            raise ValueError('decode request count != concurrency')
        for pos in groups.values():
            if len(pos) != first['k'] + 1 or pos != list(range(pos[0], pos[0] + len(pos))):
                raise ValueError('request positions must be consecutive anchor/proposals')
    elif first['phase'] != 'prefill' or len(real) != len(rows):
        raise ValueError('prefill proxy requires real prefill rows')


class Trace:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        m = self.manifest
        if m.get('schema') == SCHEMA_V2:
            self.geometry = load_geometry(m.get('geometry'))
            integer(m.get('num_experts'), 'num_experts', 1)
            integer(m.get('top_k'), 'top_k', 1)
            if m.get('geometry_fingerprint') != self.geometry.fingerprint or m.get('num_experts') != self.geometry.num_experts or m.get('top_k') != self.geometry.top_k:
                raise ValueError('route geometry fingerprint or dimensions mismatch')
            self.item_bytes = 2
        elif m.get('schema') == SCHEMA and m.get('num_experts') == 256 and m.get('top_k') == 8:
            self.geometry, self.item_bytes = LEGACY_GEOMETRY, 1
        else:
            raise ValueError('unsupported route schema or geometry')
        integer(m.get('rows'), 'rows', 1)
        if m.get('step_semantics') not in ('decode-verify', 'prefill-proxy'):
            raise ValueError('unknown step semantics')
        if not isinstance(m.get('model'), str) or not m['model']:
            raise ValueError('model required')
        if not isinstance(m.get('source'), dict) or m['source'].get('status') not in ('measured', 'assumed'):
            raise ValueError('source status required')
        layers = m.get('layers')
        if not isinstance(layers, list) or not layers or len(set(layers)) != len(layers):
            raise ValueError('invalid layers')
        for l in layers:
            integer(l, 'layer')
        if len(m.get('data', [])) != len(layers) or {d['layer'] for d in m['data']} != set(layers):
            raise ValueError('data must cover layers exactly')
        for d in [m['row_index']] + m['data']:
            p = self.path(d['file'])
            if sha256(p) != d.get('sha256'):
                raise ValueError('sha256 mismatch: ' + d['file'])
        self._maps, self._files = {}, []
        try:
            for d in m['data']:
                if d.get('dtype') != ('uint8' if self.item_bytes == 1 else 'uint16') or d.get('shape') != [m['rows'], self.geometry.top_k] or self.item_bytes == 2 and d.get('byte_order') != 'little':
                    raise ValueError('invalid binary shape/dtype')
                f = self.path(d['file']).open('rb')
                self._files.append(f)
                if os.fstat(f.fileno()).st_size != m['rows'] * self.geometry.top_k * self.item_bytes:
                    raise ValueError('binary length mismatch')
                self._maps[d['layer']] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            self.close()
            raise

    def path(self, name):
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError('data paths must be relative')
        p = (self.root / name).resolve()
        if self.root not in p.parents or '..' in Path(name).parts:
            raise ValueError('data path escapes trace')
        return p

    def row(self, layer, index):
        integer(index, 'row')
        if index >= self.manifest['rows']:
            raise ValueError('row outside trace')
        width = self.geometry.top_k * self.item_bytes
        data = self._maps[layer][index*width:(index+1)*width]
        return data if self.item_bytes == 1 else struct.unpack('<%dH' % self.geometry.top_k, data)

    def rows(self):
        count, previous = 0, -1
        with self.path(self.manifest['row_index']['file']).open() as f:
            while True:
                line = f.readline(16385)
                if not line:
                    break
                if len(line) > 16384:
                    raise ValueError('row-index line exceeds 16 KiB')
                r = json.loads(line)
                valid_row(r, count)
                if r['step'] < previous:
                    raise ValueError('steps out of order')
                previous = r['step']
                count += 1
                yield r
        if count != self.manifest['rows']:
            raise ValueError('row index length mismatch')

    def steps(self):
        for _, group in itertools.groupby(self.rows(), key=lambda r: r['step']):
            # Bound before materializing a potentially malformed giant step.
            rows = list(itertools.islice(group, MAX_STEP_ROWS + 1))
            validate_step(rows, self.manifest['step_semantics'])
            yield rows

    def check(self, require_layers=None, min_steps=0):
        if require_layers is not None and self.manifest['layers'] != list(require_layers):
            raise ValueError('required layers missing or out of order')
        steps, widths = 0, set()
        for rows in self.steps():
            steps += 1
            widths.add(len(rows))
            for r in rows:
                for l in self.manifest['layers']:
                    valid_ids(list(self.row(l, r['row'])), self.geometry if self.item_bytes == 2 else None)
        if steps < min_steps:
            raise ValueError('too few steps')
        result = dict(status='measured', structural_check='PASS', provenance='declared source, not independently certified',
                    rows=self.manifest['rows'], steps=steps, layers=self.manifest['layers'], physical_widths=sorted(widths),
                    step_semantics=self.manifest['step_semantics'])
        if self.item_bytes == 2:
            result.update(geometry=self.geometry.to_dict(), geometry_fingerprint=self.geometry.fingerprint)
        return result

    def close(self):
        for m in self._maps.values():
            m.close()
        for f in self._files:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
