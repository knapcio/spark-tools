# SPDX-License-Identifier: Apache-2.0
"""Content-free bounded JSONL and synthetic workload fixtures."""
from dataclasses import dataclass, field
import gzip
import hashlib
import re
import json
import math
from pathlib import Path
import random

FORMAT = 'sched-workload v1'
FIELDS = {'request_id', 'arrival_s', 'session_id', 'conversation_id', 'prompt_tokens',
          'cached_prefix_tokens', 'max_tokens', 'output_tokens', 'thinking', 'cancel_at_s',
          'kind', 'after_request_id', 'think_time_s', 'evidence', 'observed'}


LABELS = {'measured','estimated','assumed','unknown'}
OUTCOMES = {'completed','cancelled','disconnect','disconnected','upstream-error','error','timeout','unknown'}
TTFT_SOURCES = {'first-upstream-body-byte','first-token','unknown'}
SOURCES = {'synthetic','user-metadata','unknown'}


def identifier(value, name):
    if not isinstance(value,str) or not value or len(value)>256:
        raise ValueError('invalid '+name)
    # Idempotent opaque identifiers prevent free text passing through results.
    if re.fullmatch(r'id-[0-9a-f]{64}',value): return value
    return 'id-'+hashlib.sha256(value.encode()).hexdigest()


def label(value, choices, name):
    if not isinstance(value,str) or value not in choices:
        raise ValueError('invalid '+name+' enum')
    return value


def telemetry(row, keys):
    # Validate the entire retained projection before any SQL read/write. Never
    # serialize unvalidated allowlist values, including split/size-less events.
    result = {}
    for key in keys:
        value = row.get(key)
        if value is None: continue
        if key in ('conversation_id','root_session_id'):
            value = identifier(value,key)
        elif key in ('outcome','client_outcome'): label(value,OUTCOMES,key)
        elif key == 'ttft_source': label(value,TTFT_SOURCES,key)
        elif key == 'thinking':
            if type(value) is not bool: raise ValueError('thinking must be boolean')
        elif key in ('ttft_s','queue_wait_s'): number(value,key)
        else: integer(value,key)
        result[key] = value
    return result


def number(x, name, minimum=0):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < minimum:
        raise ValueError(name+' must be finite and >= '+str(minimum))
    return x


def integer(x, name, minimum=0):
    if type(x) is not int or x < minimum: raise ValueError(name+' must be an integer >= '+str(minimum))
    return x


@dataclass
class Work:
    request_id: str
    arrival_s: float
    session_id: str
    prompt_tokens: int
    cached_prefix_tokens: int
    max_tokens: int
    output_tokens: int
    thinking: bool = False
    cancel_at_s: float = None
    kind: str = 'prose'
    after_request_id: str = None
    think_time_s: float = 0
    evidence: dict = field(default_factory=dict)
    observed: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, row, index=0, rng=None):
        if not isinstance(row, dict) or set(row)-FIELDS:
            raise ValueError('unknown workload fields (content is forbidden)')
        rng = rng or random.Random(0)
        prompt = integer(row.get('prompt_tokens'), 'prompt_tokens', 1)
        hits = integer(row.get('cached_prefix_tokens', 0), 'cached_prefix_tokens')
        maximum = integer(row.get('max_tokens'), 'max_tokens', 1)
        if hits > prompt: raise ValueError('cached_prefix_tokens exceeds prompt')
        if prompt+maximum > 262144: raise ValueError('prompt + max_tokens exceeds 262144')
        output = row.get('output_tokens', maximum)
        if isinstance(output, dict):
            if set(output) == {'distribution', 'low', 'high'} and output['distribution'] == 'uniform':
                low, high = integer(output['low'], 'low', 1), integer(output['high'], 'high', 1)
                if low > high: raise ValueError('distribution low exceeds high')
                if high > maximum: raise ValueError('distribution high exceeds max_tokens')
                output = rng.randint(low, high)
            elif set(output) == {'distribution', 'values'} and output['distribution'] == 'choice':
                values = output['values']
                if not isinstance(values, list) or not values: raise ValueError('empty choice distribution')
                for value in values:
                    integer(value, 'choice value', 1)
                    if value>maximum: raise ValueError('choice value exceeds max_tokens')
                output = rng.choice(values)
            else: raise ValueError('output distribution must be uniform or choice')
        output = integer(output, 'output_tokens', 1)
        if output > maximum: raise ValueError('output_tokens exceeds max_tokens')
        arrival = number(row.get('arrival_s'), 'arrival_s')
        cancel = row.get('cancel_at_s')
        if cancel is not None and number(cancel, 'cancel_at_s') < arrival:
            raise ValueError('cancel_at_s precedes arrival')
        thinking = row.get('thinking', False)
        if type(thinking) is not bool: raise ValueError('thinking must be boolean')
        rid = row.get('request_id', 'r%d' % index)
        session = row.get('session_id', row.get('conversation_id', rid))
        rid = identifier(rid,'request_id')
        session = identifier(session,'session_id')
        kind = row.get('kind', 'prose')
        if kind not in ('prose', 'code'): raise ValueError('kind must be prose or code')
        dependency = row.get('after_request_id')
        if dependency is not None: dependency = identifier(dependency,'after_request_id')
        evidence, observed = row.get('evidence', {}), row.get('observed', {})
        if not isinstance(evidence, dict) or not isinstance(observed, dict):
            raise ValueError('evidence and observed must be metadata objects')
        # Explicit allowlists prevent arbitrary payload from being copied through.
        if set(evidence)-{'arrival', 'sizes', 'prefix', 'output', 'thinking', 'source'}:
            raise ValueError('unknown evidence fields')
        if set(observed)-{'ttft_s', 'queue_wait_s', 'kv_reservation', 'end_s', 'outcome', 'ttft_source'}:
            raise ValueError('unknown observed fields')
        for key,value in evidence.items():
            label(value,SOURCES if key == 'source' else LABELS,'evidence.'+key)
        for key in ('outcome','ttft_source'):
            if key in observed: label(observed[key],OUTCOMES if key == 'outcome' else TTFT_SOURCES,key)
        for key in ('ttft_s', 'queue_wait_s', 'kv_reservation', 'end_s'):
            if observed.get(key) is not None: number(observed[key], key)
        return cls(rid, arrival, session, prompt, hits, maximum, output, thinking, cancel,
                   kind, dependency, number(row.get('think_time_s', 0), 'think_time_s'), evidence, observed)

    def row(self):
        return {k:v for k,v in vars(self).items() if v is not None}


def lines(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        index = 0
        while True:
            line = stream.readline(1048577)
            if not line: break
            index += 1
            if len(line) > 1048576: raise ValueError('JSONL line exceeds 1 MiB at '+str(index))
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as error: raise ValueError('invalid JSONL at line '+str(index)) from error


def load(path, seed=0, max_records=50000):
    rows, meta, ids = [], {}, set()
    rng = random.Random(seed)
    for row in lines(path):
        if isinstance(row, dict) and 'format' in row:
            if rows or meta or row['format'] != FORMAT: raise ValueError('invalid workload header')
            if set(row)-{'format', 'evidence', 'source', 'notes'}: raise ValueError('unknown header fields')
            if 'evidence' in row: label(row['evidence'],LABELS,'header evidence')
            if 'source' in row: label(row['source'],SOURCES,'header source')
            if 'notes' in row: label(row['notes'],LABELS,'header notes')
            meta = row
            continue
        if len(rows) >= max_records: raise ValueError('workload exceeds bounded max-records')
        work = Work.parse(row, len(rows), rng)
        if work.request_id in ids: raise ValueError('duplicate request_id')
        if work.after_request_id is not None and work.after_request_id not in ids:
            raise ValueError('after_request_id must reference an earlier row')
        ids.add(work.request_id)
        rows.append(work)
    if not rows: raise ValueError('empty workload')
    return rows, meta or {'format': FORMAT, 'evidence': 'assumed'}


def write(path, rows, meta=None):
    rows = [Work.parse(row.row() if isinstance(row,Work) else row,i) for i,row in enumerate(rows)]
    meta = meta or {'format':FORMAT}
    if set(meta)-{'format','evidence','source','notes'} or meta.get('format') != FORMAT:
        raise ValueError('invalid workload header')
    for key in ('evidence','source','notes'):
        if key in meta: label(meta[key],SOURCES if key == 'source' else LABELS,'header '+key)
    with open(path, 'w', encoding='utf-8') as stream:
        stream.write(json.dumps(meta or {'format': FORMAT})+'\n')
        for row in rows: stream.write(json.dumps(row.row() if isinstance(row, Work) else row, sort_keys=True)+'\n')


def synthetic(name, seed=0, sessions=4, rounds=3, think_s=8):
    rng = random.Random(seed)
    rows = []
    if name == 'long-during-decode':
        # A is APC-warm; B arrives after A has begun decoding. Large A output
        # retains it throughout even the expensive N80 arm.
        rows = [dict(request_id='A', arrival_s=0, session_id='A', prompt_tokens=60043,
                     cached_prefix_tokens=60042, max_tokens=16000, output_tokens=16000),
                dict(request_id='B', arrival_s=8, session_id='B', prompt_tokens=100050,
                     cached_prefix_tokens=0, max_tokens=16, output_tokens=16)]
    elif name == 'newcomer':
        rows = [dict(request_id='A', arrival_s=0, prompt_tokens=128000, max_tokens=2, output_tokens=2),
                dict(request_id='B', arrival_s=10, prompt_tokens=300, max_tokens=16, output_tokens=16)]
    elif name == 'bursty':
        for i in range(sessions*rounds):
            rows.append(dict(request_id='r%d' % i, arrival_s=(i//sessions)*30+rng.random(),
                             session_id='s%d' % (i%sessions), prompt_tokens=rng.choice([1024,8192,65536,100000]),
                             cached_prefix_tokens=0, max_tokens=512, output_tokens=rng.randint(64,512), thinking=True))
    elif name == 'agents':
        for session in range(sessions):
            context, prefix, prior = 8192+session*2048, 0, None
            for turn in range(rounds):
                output = rng.randint(256,1024)
                rid = 's%d-t%d' % (session,turn)
                rows.append(dict(request_id=rid, arrival_s=session*.3, session_id='s%d' % session,
                                 prompt_tokens=context, cached_prefix_tokens=prefix, max_tokens=1024,
                                 output_tokens=output, thinking=True, after_request_id=prior, think_time_s=think_s,
                                 kind='code' if session%2 else 'prose'))
                prior, prefix = rid, context
                context += output+2048
    else: raise ValueError('synthetic name must be agents, bursty, long-during-decode, newcomer')
    works = [Work.parse({**r, 'evidence': {'arrival':'assumed', 'sizes':'assumed','prefix':'assumed', 'output':'assumed'}}, i, rng)
             for i,r in enumerate(rows)]
    return works, {'format': FORMAT, 'evidence':'assumed', 'source':'synthetic'}

