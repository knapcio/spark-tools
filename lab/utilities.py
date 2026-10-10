# SPDX-License-Identifier: Apache-2.0
"""Small, bounded-memory I/O and arithmetic shared by the offline tools."""
import gzip
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, doc):
    Path(path).write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')


def stats(xs):
    xs = sorted(xs)
    if not xs:
        return None
    def q(p):
        i = (len(xs) - 1) * p
        a, b = int(i), math.ceil(i)
        return xs[a] + (xs[b] - xs[a]) * (i - a)
    return {'mean': statistics.fmean(xs), 'median': statistics.median(xs), 'p10': q(.1), 'p90': q(.9), 'n': len(xs)}


def union_us(intervals):
    edge, total = -math.inf, 0.0
    for a, b in sorted(intervals):
        total += max(0, b - max(a, edge))
        edge = max(edge, b)
    return total


class JsonStream:
    """Incremental JSON decoder; at most one event/value, bounded to 8 MiB."""
    def __init__(self, f):
        self.f, self.buf, self.pos, self.eof = f, '', 0, False
        self.decoder = json.JSONDecoder()

    def more(self):
        self.buf = self.buf[self.pos:]
        self.pos = 0
        if len(self.buf) > 8 << 20:
            raise ValueError('JSON value exceeds 8 MiB safety bound')
        chunk = self.f.read(65536)
        self.buf += chunk
        self.eof = not chunk
        return bool(chunk)

    def char(self):
        while True:
            while self.pos < len(self.buf) and self.buf[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buf):
                return self.buf[self.pos]
            if not self.more():
                return ''

    def take(self, ch):
        if self.char() != ch:
            raise ValueError('expected JSON ' + repr(ch))
        self.pos += 1

    def value(self):
        self.char()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buf, self.pos)
                # A scalar at a chunk boundary may be incomplete (e.g. 12|34).
                if end == len(self.buf) and not self.eof:
                    if self.more():
                        continue
                else:
                    self.pos = end
                    return value
            except json.JSONDecodeError:
                if self.eof:
                    raise ValueError('malformed or truncated JSON')
                self.more()


def trace_events(path, metadata):
    """Yield Chrome events, fully validating the gzip and JSON through EOF."""
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        s = JsonStream(f)
        bare = s.char() == '['
        if not bare:
            s.take('{')
        found = False
        while True:
            if bare:
                key = 'traceEvents'
            else:
                if s.char() == '}':
                    s.take('}')
                    break
                key = s.value()
                if not isinstance(key, str):
                    raise ValueError('non-string JSON key')
                s.take(':')
            if key == 'traceEvents':
                if found:
                    raise ValueError('duplicate traceEvents')
                found = True
                s.take('[')
                if s.char() != ']':
                    while True:
                        event = s.value()
                        if not isinstance(event, dict):
                            raise ValueError('trace event must be an object')
                        yield event
                        if s.char() != ',':
                            break
                        s.take(',')
                        if s.char() == ']':
                            raise ValueError('trailing event comma')
                s.take(']')
            else:
                metadata[key] = s.value()
            if bare or s.char() != ',':
                if not bare:
                    s.take('}')
                break
            s.take(',')
            if s.char() == '}':
                raise ValueError('trailing object comma')
        if s.char():
            raise ValueError('trailing JSON content')
        if not found:
            raise ValueError('missing traceEvents')


PROM = re.compile(r'^([\w:]+)(?:\{([^}]*)\})?\s+([-+\d.eE]+)(?:\s|$)')


def read_prom(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        m = PROM.match(line)
        if not m:
            continue
        name, raw, val = m.groups()
        if not name.startswith('vllm:spec_decode_') or not name.endswith('_total'):
            continue
        labels = tuple(sorted(re.findall(r'(\w+)="([^"]*)"', raw or '')))
        out[(name, labels)] = float(val)
    return out


def acceptance(before, after, k):
    a, b = read_prom(before), read_prom(after)
    if set(a) - set(b):
        raise ValueError('counter series disappeared after request')
    values, pos = {}, [0.0] * k
    seen_pos = set()
    for (name, labels), v in b.items():
        if (name, labels) not in a:
            raise ValueError('missing before counter series')
        d = v - a[(name, labels)]
        if d < 0 or not math.isfinite(d):
            raise ValueError('counter reset or non-finite counter')
        if 'per_pos' in name:
            p = int(dict(labels)['position'])
            if p < k:
                pos[p] += d
                seen_pos.add(p)
        else:
            values[name.split('spec_decode_', 1)[1]] = values.get(name.split('spec_decode_', 1)[1], 0) + d
    drafts = values.get('num_drafts_total', 0)
    if not {'num_drafts_total', 'num_draft_tokens_total', 'num_accepted_tokens_total'} <= set(values) or seen_pos != set(range(k)):
        raise ValueError('missing speculative total/per-position counters')
    if drafts <= 0:
        raise ValueError('no draft opportunities')
    survival = [p / drafts for p in pos]
    validate_survival(survival)
    if abs(sum(pos) - values.get('num_accepted_tokens_total', sum(pos))) > 1e-6:
        raise ValueError('accepted position sum disagrees with total')
    return {'survival': survival, 'committed': 1 + sum(survival), 'drafts': drafts,
            'nominal_draft_tokens_per_cycle': values.get('num_draft_tokens_total', 0) / drafts,
            'status': 'measured', 'passes_status': 'unknown (nominal tokens include kstop padding)'}


def validate_survival(ps):
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in ps):
        raise ValueError('prefix survival must be finite in [0,1]')
    if any(a + 1e-9 < b for a, b in zip(ps, ps[1:])):
        raise ValueError('prefix survival must be nonincreasing')
