#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline Kineto/CUPTI decode-cycle ingestion. One trace in memory at a time.
Segmentation and interval unions adapted from the workspace gap_audit.py
(2026-10-01, Apache-2.0); no upstream full-document loader is used.
"""
import argparse
import bisect
import collections
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from cyclesim.utilities import acceptance, dump, sha256, stats, trace_events, union_us

HERE = Path(__file__).resolve().parent
COMM = re.compile(r'b12xcomm|roce|nccl|all_?reduce|all_?gather|reduce_?scatter|one_?shot|custom_ar', re.I)


def classify(name, cat='kernel'):
    n = name.lower()
    if cat in ('gpu_memcpy', 'gpu_memset'):
        return 'memcpy'
    if COMM.search(name):
        return 'allreduce/roce'
    if 'marlin_moe' in n or 'fused_moe' in n or 'moe_wna16' in n:
        return 'marlin_moe'
    # float4_e2m1f.id from the pinned vLLM ScalarType bit layout.
    if 'nvfp4' in n or '562949953487106' in n:
        return 'nvfp4_gemm'
    if 'marlin::marlin' in n or 'w8a16' in n or name == 'linear':
        return 'int8_gemm'
    if 'single_group_topk' in n or 'expert_tokens' in n or 'moe_align' in n:
        return 'router'
    if any(t in n for t in ('indexer', 'paged_mqa_logits', 'persistent_topk')):
        return 'indexer'
    if any(t in n for t in ('_mla', 'mla_', 'flash', 'concat_and_cache', 'sparse_attn')):
        return 'attention/mla'
    if any(t in n for t in ('gemm', 'gemv', 'cutlass', 'nvjet', 'xmma', 'cublas', 'splitkreduce')):
        return 'bf16_gemm'
    if any(t in n for t in ('sampl', 'rejection', 'argmax', 'softmax', 'top_k', 'top_p', 'sort', 'gumbel', 'cumsum', 'multinomial')):
        return 'sampling'
    if any(t in n for t in ('norm', 'elementwise', 'act_and_mul', 'rotary', 'moe_sum')):
        return 'rmsnorm/elementwise'
    return 'other'


def load_acts(path):
    acts, meta, names = [], {}, {}
    for e in trace_events(path, meta):
        if e.get('ph') != 'X' or e.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        a = e.get('args', {})
        ts, dur = float(e['ts']), float(e.get('dur', 0))
        if not math.isfinite(ts) or not math.isfinite(dur) or dur < 0:
            raise ValueError('invalid activity duration/timestamp')
        name = names.setdefault(e.get('name', ''), e.get('name', ''))
        acts.append((ts, ts + dur, name, classify(name, e['cat']), a.get('graph id', 0),
                     a.get('correlation'), a.get('graph node id'), a.get('bytes'), tuple(a.get('grid') or ()), a.get('projection'), a.get('quantization'), a.get('working_set_bytes')))
    acts.sort(key=lambda a: a[0])
    rank = (meta.get('distributedInfo') or {}).get('rank')
    if rank is None:
        m = re.search(r'rank(\d+)', str(path))
        rank = int(m[1]) if m else None
    if rank is None:
        raise ValueError('rank unknown; use rankN directories or distributedInfo.rank')
    return acts, rank, meta


def category_for(cl, phase, head=False):
    if cl == 'allreduce/roce':
        return 'collectives'
    if phase.startswith('draft') or phase == 'draft/eager':
        return 'draft_graph'
    if head:
        return 'vocab_head'
    return {'marlin_moe': 'moe_routed', 'nvfp4_gemm': 'attention_gemms',
            'int8_gemm': 'attention_gemms', 'bf16_gemm': 'shared_dense',
            'attention/mla': 'mla', 'indexer': 'indexer', 'router': 'router'}.get(cl, 'host_sampling')


def bytes_for(ledger, cl, phase, calls, head, c):
    if head:
        quant = 'nvfp4' if cl == 'nvfp4_gemm' else 'bf16'
        which = 'draft' if phase.startswith('draft') else 'target'
        value = ledger.get('head_bytes', {}).get(which, {}).get(quant)
        if value is None and quant == 'bf16':
            value = ledger['vocab_head_bytes']
        if value is None:
            return None, None, 'unknown', 'head phase/quantization unbound'
        return value * calls, value, 'derived', 'vocab-shard dimensions'
    budgets = ledger.get('cycle_budgets', {}).get(str(c), {})
    value = budgets.get(phase, {}).get(cl)
    if value is None:
        return None, None, 'unknown', 'no shape/traffic binding'
    return value, None, 'estimated', 'class ledger apportioned by calls; expert union assumed; working set unknown'


def analyze(path, ledger, c=1, k=3, target_norms=None, target_descriptors=None):
    acts, rank, meta = load_acts(path)
    groups = collections.defaultdict(list)
    for a in acts:
        if a[4]:
            groups[(a[4], a[5])].append(a)
    if not groups:
        raise ValueError('no resolved CUDA graph nodes')
    replays = []
    for key, xs in groups.items():
        norms = len({a[6] for a in xs if 'fused_add_rms_norm' in a[2]})
        replays.append((min(a[0] for a in xs), max(a[1] for a in xs), key, norms))
    replays.sort()
    want = target_norms if target_norms is not None else max(r[3] for r in replays)
    if want < 16 and target_norms is None:
        raise ValueError('no target-sized graph (at least 16 residual norms)')
    descriptor_norms = {d['graph_id']:d['norms'] for d in target_descriptors or []}
    if descriptor_norms:
        for replay in replays:
            if replay[2][0] in descriptor_norms and replay[3] != descriptor_norms[replay[2][0]]:
                raise ValueError('trace target residual norms contradict descriptor evidence')
        targets = [r for r in replays if r[2][0] in descriptor_norms]
    else:
        targets = [r for r in replays if r[3] == want]
    target_keys = {r[2] for r in targets}
    if len(targets) < 4:
        raise ValueError('fewer than 4 target replays (need 2 complete kept cycles)')
    times = [a[0] for a in acts]
    # Union the complete trace before clipping busy time to each cycle.
    # Calls and graph labels still belong to the cycle in which they start.
    busy_intervals = []
    for a in acts:
        if busy_intervals and a[0] <= busy_intervals[-1][1]:
            busy_intervals[-1][1] = max(busy_intervals[-1][1], a[1])
        else:
            busy_intervals.append([a[0], a[1]])
    busy_starts = [a for a, b in busy_intervals]
    busy_ends = [b for a, b in busy_intervals]
    rep_times = [r[0] for r in replays]
    cycles = []
    for index in range(1, len(targets) - 1):
        start, end = targets[index][0], targets[index + 1][0]
        xs = acts[bisect.bisect_left(times, start):bisect.bisect_left(times, end)]
        rs = replays[bisect.bisect_left(rep_times, start):bisect.bisect_left(rep_times, end)]
        labels, di = {}, 0
        for r in rs:
            if r[2] in target_keys:
                labels[r[2]] = 'target'
            else:
                di += 1
                labels[r[2]] = 'draft/' + str(di)
        # An eager pass after the last graph is detected by routed/residual kernels.
        eager_start = min((a[0] for a in xs if not a[4] and a[0] >= rs[-1][1]
                           and ('fused_add_rms_norm' in a[2] or a[3] == 'marlin_moe')), default=end)
        data, comm = collections.defaultdict(lambda: [0, 0.0]), []
        signature = []
        for a in xs:
            phase = labels.get((a[4], a[5]), 'draft/eager' if a[0] >= eager_start else 'post_target')
            if not a[4] and phase != 'draft/eager' and di and a[0] > rs[1][0]:
                phase = 'tail'
            matches = [b for b in ledger.get('kernel_bindings', []) if
                       b.get('graph_id') == a[4] and b.get('node_id') == a[6] and
                       b.get('phase') == ('draft' if phase.startswith('draft') else phase)]
            if len(matches) > 1:
                raise ValueError('ambiguous projection binding')
            binding = matches[0] if matches else {}
            key = (a[2], a[3], phase, a[8], binding.get('projection', a[9]),
                   binding.get('quantization', a[10]), binding.get('working_set_bytes', a[11]),
                   binding.get('bytes_per_call'))
            data[key][0] += 1
            data[key][1] += a[1] - a[0]
            if a[3] == 'allreduce/roce':
                comm.append((a[2], a[1] - a[0]))
            if phase == 'target':
                signature.append((a[6], a[2]))
        sig = hashlib.sha256(repr(sorted(signature, key=repr)).encode()).hexdigest()
        overlaps = busy_intervals[bisect.bisect_right(busy_ends, start):
                                  bisect.bisect_left(busy_starts, end)]
        busy = union_us((max(start, a), min(end, b)) for a, b in overlaps)
        cycles.append({'index': index, 'cycle_ms': (end - start) / 1000, 'kernel_union_ms': busy / 1000,
                       'idle_ms': (end - start - busy) / 1000, 'device_activity_count': len(xs),
                       'kernel_count': sum(a[3] != 'memcpy' for a in xs), 'data': data,
                       'comm': comm, 'signature': (sig, di), 'executed_passes': di + int(eager_start < end), 'target_graph': rs[0][2][0]})
    # Keep variable draft counts: kstop changes these within a legitimate cell.
    # Every target-sized descriptor is valid: q2/q3/q4 graphs can differ in
    # node IDs and widths. Report their frequencies, never modal-filter them.
    kept = cycles
    keys = sorted({key for cy in kept for key in cy['data']}, key=repr)
    # Head fingerprint: longest BF16/INT8 projection in post-target or a draft pass.
    heads = set()
    for phase in {key[2] for key in keys}:
        candidates = [key for key in keys if key[2] == phase and key[1] in ('bf16_gemm', 'int8_gemm', 'nvfp4_gemm')]
        explicit = [key for key in candidates if key[4] == 'head']
        if explicit:
            heads.update(explicit)
        elif ledger['stack'].startswith('historical') and phase != 'target' and any(q[1]=='bf16_gemm' for q in candidates):
            # Historical BF16 trace fingerprint only; never infer a current DH
            # head from duration (the smaller head need not be longest).
            heads.add(max([q for q in candidates if q[1]=='bf16_gemm'], key=lambda key: stats([cy['data'].get(key, [0, 0])[1] for cy in kept])['mean'] /
                               max(1, stats([cy['data'].get(key, [0, 0])[0] for cy in kept])['mean'])))
    phase_calls = collections.Counter()
    for key in keys:
        if key not in heads:
            phase_calls[(key[2], key[1])] += stats([cy['data'].get(key, [0, 0])[0] for cy in kept])['mean']
    rows = []
    category = collections.defaultdict(float)
    for number, key in enumerate(keys):
        name, cl, phase, grid, projection, quantization, captured_working, bound_bytes = key
        calls = stats([cy['data'].get(key, [0, 0])[0] for cy in kept])
        duration = stats([cy['data'].get(key, [0, 0])[1] for cy in kept])
        head = key in heads
        bp = 'draft' if phase.startswith('draft') else phase
        nbytes, working, status, basis = bytes_for(ledger, cl, bp, calls['mean'] if head else phase_calls[(phase, cl)], head, c)
        if not ledger['stack'].startswith('historical') and cl in ('bf16_gemm','nvfp4_gemm','int8_gemm') and not projection:
            nbytes, working, status, basis = None, None, 'unknown', 'current projection/quantization/phase binding missing'
        if nbytes is not None and not head:
            duty = sum(any(q[2] == phase and q[1] == cl for q in cy['data']) for cy in kept) / len(kept)
            nbytes *= duty * calls['mean'] / max(1e-9, phase_calls[(phase, cl)])
        if bound_bytes is not None:
            if head and nbytes is not None and not math.isclose(bound_bytes*calls['mean'], nbytes, abs_tol=1e-6):
                raise ValueError('head binding bytes contradict phase/quantization ledger')
            if not isinstance(bound_bytes, (int, float)) or not math.isfinite(bound_bytes) or bound_bytes < 0:
                raise ValueError('invalid bound per-call traffic')
            nbytes, status, basis = bound_bytes*calls['mean'], 'derived', 'projection/quant/phase binding'
        if captured_working is not None:
            working = float(captured_working)
            if not math.isfinite(working) or working < 0:
                raise ValueError('invalid captured working set')
        if head and bp=='target' and cl=='nvfp4_gemm':
            raise ValueError('target head must remain BF16')
        if head and not projection:
            status = 'estimated historical head fingerprint; weight bytes derived'
        if head and quantization and quantization != ('nvfp4' if cl == 'nvfp4_gemm' else 'bf16'):
            raise ValueError('head quantization disagrees with kernel class')
        gbs = nbytes / (duration['mean'] * 1000) if nbytes is not None and duration['mean'] else None
        cat = category_for(cl, phase, head)
        category[cat] += duration['mean'] / 1000
        rows.append({'id': 'r%d-k%d' % (rank, number), 'rank': rank, 'name': name, 'grid': list(grid), 'phase': phase,
                     'projection': 'head' if head else projection, 'quantization': quantization or ('nvfp4' if cl == 'nvfp4_gemm' else 'bf16' if head else 'unknown'),
                     'class': ('draft/' + cl) if phase.startswith('draft') else cl, 'category': cat,
                     'calls_per_cycle': calls, 'total_us_per_cycle': duration,
                     'time_status': 'measured', 'share_of_cycle': duration['mean'] / (stats([cy['cycle_ms'] for cy in kept])['mean'] * 1000),
                     'estimated_bytes_read': nbytes, 'bytes_status': status, 'bytes_basis': basis,
                     'working_set_bytes': working, 'achieved_gbs': gbs, 'gbs_status': 'estimated' if gbs is not None else 'unknown',
                     'dram_ceiling_fraction': [gbs / 270, gbs / 250] if gbs is not None else None,
                     'l2_resident': (working < 24e6) if working is not None else None,
                     'l2_status': 'assumed heuristic; not a cache measurement'})
    summary = {field: stats([cy[field] for cy in kept]) for field in
               ('cycle_ms', 'kernel_union_ms', 'idle_ms', 'kernel_count', 'device_activity_count')}
    category['host_sampling'] += summary['idle_ms']['mean']
    summary.update({'rank': rank, 'cycles_total': len(cycles), 'cycles_kept': len(kept),
                    'target_descriptor_frequencies': dict(collections.Counter(cy['signature'][0] for cy in kept)),
                    'target_graph_frequencies': dict(collections.Counter(str(cy['target_graph']) for cy in kept)),
                    'pass_count_frequencies': dict(collections.Counter(str(cy['executed_passes']) for cy in kept)),
                    'executed_passes': stats([cy['executed_passes'] for cy in kept]),
                    'draft_graph_passes': stats([cy['signature'][1] for cy in kept]),
                    'representative_cycle_index': min(kept, key=lambda cy: abs(cy['cycle_ms'] - summary['cycle_ms']['median']))['index'],
                    'target_norms': want, 'status': 'measured', 'host_gap_scope': 'GPU idle upper bound; includes CUPTI launch inflation',
                    'category_ms': dict(category), 'vllm_version': meta.get('vllm_version')})
    compact = [{'signature': cy['signature'][0], 'comm': cy['comm']} for cy in kept]
    return summary, rows, compact


def compare_ranks(compact):
    ranks = sorted(compact)
    joined = []
    for i in range(min(map(len, compact.values()))):
        cs = [compact[r][i] for r in ranks]
        if len({cy['signature'] for cy in cs}) != 1 or any([x[0] for x in cy['comm']] != [x[0] for x in cs[0]['comm']] for cy in cs):
            continue
        floor = [min(cy['comm'][j][1] for cy in cs) for j in range(len(cs[0]['comm']))]
        joined.append([sum(max(0, d - floor[j]) for j, (_, d) in enumerate(cy['comm'])) / 1000 for cy in cs])
    return {'status': 'estimated' if joined and len(ranks) > 1 else 'unknown', 'joined_cycles': len(joined),
            'collective_excess_ms': {str(r): stats([v[i] for v in joined]) for i, r in enumerate(ranks)} if len(ranks) > 1 else {},
            'scope': 'ordinal join with identical target/collective signatures; cross-rank lower-envelope wait proxy, never clock subtraction'}


def check_capture(root, strict=True):
    p = root / 'manifest.json'
    if not p.exists():
        if strict:
            raise ValueError('missing capture manifest.json')
        return None
    manifest = json.loads(p.read_text())
    if manifest.get('schema') != 1 or manifest.get('mode') not in ('profile', 'companions-only'):
        raise ValueError('invalid capture manifest schema/mode')
    c, k = manifest.get('c'), manifest.get('k')
    if c not in (1, 2, 4) or k != (3 if c == 1 else 2) or manifest.get('domain') not in ('prose', 'code'):
        raise ValueError('capture requires boot K3 c1 or uniform K2 c2/c4, prose/code')
    state_keys = ('c', 'k', 'domain', 'head', 'policy', 'capture_layout', 'pad_hygiene')
    if manifest.get('head') not in ('bf16', 'nvfp4') or manifest.get('policy') != ('kstop' if c == 1 else 'uniform_k2'):
        raise ValueError('missing/invalid head or policy')
    if manifest.get('capture_layout') != 'reuse' or manifest.get('pad_hygiene') is not True:
        raise ValueError('current capture requires reuse and pad hygiene')
    files = manifest.get('files', [])
    by_role = {}
    for row in files:
        original = root / row['path']
        path = original.resolve()
        if root.resolve() not in path.parents or not path.is_file() or original.is_symlink():
            raise ValueError('invalid manifest path')
        if sha256(path) != row['sha256']:
            raise ValueError('sha256 mismatch: ' + row['path'])
        if row['role'] != 'trace' and row['role'] in by_role:
            raise ValueError('duplicate companion role')
        by_role[row['role']] = path
    needed = {'prom_before', 'prom_after', 'client_timings', 'env', 'graph_sizes', 'a1', 'a2', 'prompts'}
    if manifest['mode'] == 'profile':
        needed.add('profile_client')
    if not needed <= set(by_role):
        raise ValueError('missing companion roles: ' + ','.join(sorted(needed - set(by_role))))
    if 'routing_capture' not in manifest:
        raise ValueError('missing reserved routing_capture slot')
    counters = acceptance(by_role['prom_before'], by_role['prom_after'], k)
    if counters['drafts'] < 16 * c:
        raise ValueError('counter observation coverage below 16 per stream')
    def compatible(cell):
        if any(cell.get(key) != manifest[key] for key in state_keys):
            raise ValueError('incompatible domain/policy/head/graph state')
        if cell.get('boot') != manifest.get('boot'):
            raise ValueError('incompatible boot')
    brackets = []
    for role in ('a1', 'a2') + (('profile_client',) if manifest['mode'] == 'profile' else ()):
        cell = json.loads(by_role[role].read_text())
        value = cell.get('client_cycle_ms')
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(role + ': positive client_cycle_ms required')
        if cell.get('instrument') != 'client_decode_wall_per_draft':
            raise ValueError(role + ': exact instrument client_decode_wall_per_draft required')
        compatible(cell)
        if cell.get('observations', 0) < 16 or cell.get('errors') != 0 or cell.get('preemptions') != 0:
            raise ValueError(role + ': insufficient observations or errors/preemptions')
        if role == 'profile_client':
            if cell.get('record_shapes') is not False or not 16 <= cell.get('iterations', 0) <= 24:
                raise ValueError('P requires 16-24 iterations and record_shapes=false')
            for field in ('peak_memory_debit_gib', 'retained_memory_debit_gib'):
                v = cell.get(field)
                if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                    raise ValueError('P missing separate peak/retained memory measurements')
            memory = cell.get('memory', {})
            if set(memory) != set(map(str, range(4))):
                raise ValueError('P needs memory observations on all ranks')
            for rank in memory.values():
                for field in ('before_gib','minimum_gib','after_gib','peak_debit_gib','retained_debit_gib'):
                    v = rank.get(field)
                    if not isinstance(v,(int,float)) or not math.isfinite(v) or v < 0:
                        raise ValueError('invalid per-rank memory evidence')
                if rank['before_gib'] < 7.8 or rank['minimum_gib'] < 6.0:
                    raise ValueError('profile capture violates unchanged 7.8 preflight/6.0 capture floors')
                if not math.isclose(rank['peak_debit_gib'], max(0,rank['before_gib']-rank['minimum_gib']), abs_tol=.02):
                    raise ValueError('peak debit inconsistent with memory minimum')
                if not math.isclose(rank['retained_debit_gib'], max(0,rank['before_gib']-rank['after_gib']), abs_tol=.02):
                    raise ValueError('retained debit inconsistent with after memory')
        else:
            brackets.append(value)
    mean = sum(brackets) / 2
    anchor = manifest.get('unprofiled_cycle_ms')
    if not isinstance(anchor, (int, float)) or not math.isfinite(anchor) or not math.isclose(anchor, mean, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError('unprofiled_cycle_ms must equal compatible A1/A2 arithmetic mean')
    timings = [json.loads(line) for line in by_role['client_timings'].read_text().splitlines() if line.strip()]
    arms = {'A1', 'P', 'A2'} if manifest['mode'] == 'profile' else {'A1', 'A2'}
    streams = collections.defaultdict(list)
    for row in timings:
        if (not isinstance(row.get('arm'), str) or row['arm'] not in arms or
                not isinstance(row.get('request_id'), str) or not row['request_id'] or
                any(not isinstance(row.get(key), int) or isinstance(row[key], bool) or row[key] < 0
                    for key in ('monotonic_ns', 'stream', 'chunk_index'))):
            raise ValueError('invalid client timing row')
        streams[(row['arm'], row['stream'], row['request_id'])].append(row)
    if {key[:2] for key in streams} != {(arm, s) for arm in arms for s in range(c)} or len(streams) != len(arms) * c:
        raise ValueError('client timings need one request per arm and stream')
    for rows in streams.values():
        if len(rows) < 17 or any(b['monotonic_ns'] <= a['monotonic_ns'] or b['chunk_index'] != a['chunk_index'] + 1
                                 for a, b in zip(rows, rows[1:])):
            raise ValueError('need 16 consecutive positive client intervals per arm and stream')
    env = json.loads(by_role['env'].read_text())
    compatible(env)
    if not env.get('boot') or not env.get('image') or not env.get('cmd') or not isinstance(env.get('env'), dict):
        raise ValueError('env needs boot/image/cmd/env fields')
    ev = env['env']
    if (ev.get('MTP_KSTOP') != '1' or ev.get('MTP_UNIFORM_BATCH') != 'k2' or
            ev.get('MTP_CAPTURE_LAYOUT') != 'reuse' or ev.get('PAD_HYGIENE') != '1' or
            ev.get('DRAFT_HEAD') != ('nvfp4' if manifest['head'] == 'nvfp4' else '0')):
        raise ValueError('environment contradicts execution state')
    graphs = json.loads(by_role['graph_sizes'].read_text())
    compatible(graphs)
    if graphs.get('configured') != [1, 4, 12, 16] or set(graphs.get('ranks', {})) != {'0', '1', '2', '3'}:
        raise ValueError('graph_sizes needs configured widths and all four rank observations')
    passes = []
    profiled = manifest['mode'] == 'profile'
    for rank in graphs['ranks'].values():
        compatible(dict(rank, boot=manifest['boot']))
        descriptors = rank.get('descriptors', [])
        if not isinstance(rank.get('evidence'), str) or not rank['evidence'] or not isinstance(descriptors, list) or not descriptors:
            raise ValueError('empty/invalid target descriptor-size evidence')
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                raise ValueError('invalid target descriptor evidence')
            for field, minimum in (('graph_id', 1), ('norms', 16), ('rows', 1)):
                value = descriptor.get(field)
                # Boot logging records descriptor sizes. IDs/norm counts need
                # profiler evidence; omitted/null values remain unknown here.
                if value is None and not profiled and field != 'rows':
                    continue
                if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                    raise ValueError('empty/invalid resolved target descriptor evidence')
        if {d.get('rows') for d in descriptors} != ({2, 3, 4} if c == 1 else {3*c}):
            raise ValueError('target descriptors disagree with K-specific boot widths')
        known_ids = [d['graph_id'] for d in descriptors if d.get('graph_id') is not None]
        if len(set(known_ids)) != len(known_ids):
            raise ValueError('duplicate target graph ID')
        count = rank.get('executed_passes')
        if not profiled and count is None:
            passes.append(None)
            continue
        if not isinstance(count, (int, float)) or not math.isfinite(count) or not 1 <= count <= k or (c > 1 and count != 2):
            raise ValueError('missing/invalid executed pass evidence')
        passes.append(count)
    if any(p is None for p in passes) and not all(p is None for p in passes):
        raise ValueError('rank executed-pass availability mismatch')
    if all(p is not None for p in passes) and max(passes) - min(passes) > 1e-9:
        raise ValueError('rank execution pass mismatch')
    manifest['executed_passes'] = passes[0]
    prompts = json.loads(by_role['prompts'].read_text()).get('prompts', {})
    if any(not isinstance(prompts.get(domain), list) or len(prompts[domain]) < 4 or
           any(not isinstance(text, str) or not text for text in prompts[domain]) for domain in ('prose', 'code')):
        raise ValueError('prompts needs four prose and four code texts')
    if manifest['mode'] == 'profile':
        traces = [row for row in files if row['role'] == 'trace']
        if len(traces) != 4 or {row.get('rank') for row in traces} != {0, 1, 2, 3}:
            raise ValueError('profile needs exactly one trace rank0..3')

    return manifest


def ingest(root, ledger, c, k, target_norms=None, strict=False):
    manifest = check_capture(root, strict)
    if strict and (manifest['c'] != c or manifest['k'] != k):
        raise ValueError('CLI c/k differs from capture manifest')
    if manifest and manifest['mode'] == 'companions-only':
        paths = {row['role']: root / row['path'] for row in manifest['files']}
        return {'schema': 1, 'status': 'usable-companions-only', 'kernels': [], 'ranks': [],
                'c': manifest['c'], 'k': manifest['k'], 'domain': manifest['domain'],
                'executed_passes': manifest['executed_passes'],
                'acceptance': acceptance(paths['prom_before'], paths['prom_after'], manifest['k']),
                'per_kernel_capture': 'unknown: profiler-disabled companions-only capture', 'manifest_sha256': sha256(root / 'manifest.json')}
    binding_files = [row for row in (manifest or {}).get('files', []) if row['role']=='kernel_bindings']
    if binding_files:
        ledger = dict(ledger, kernel_bindings=json.loads((root/binding_files[0]['path']).read_text())['bindings'])
    traces = sorted(set(root.rglob('*.pt.trace.json')) | set(root.rglob('*.pt.trace.json.gz')))
    if strict:
        listed = {row['path'] for row in manifest['files'] if row['role'] == 'trace'}
        if {path.relative_to(root).as_posix() for path in traces} != listed:
            raise ValueError('trace files differ from hashed manifest')
    if not traces:
        raise ValueError('no *.pt.trace.json[.gz] traces')
    summaries, rows, compact, sources = [], [], {}, []
    for path in traces:
        descriptors = None
        if strict:
            expected = next(row['rank'] for row in manifest['files'] if row['role']=='trace' and row['path']==path.relative_to(root).as_posix())
            graphs = json.loads(next((root/r['path']).read_text() for r in manifest['files'] if r['role']=='graph_sizes'))
            descriptors = graphs['ranks'][str(expected)]['descriptors']
        s, rs, cs = analyze(path, ledger, c, k, target_norms, descriptors)
        if strict:
            expected = next(row['rank'] for row in manifest['files'] if row['role'] == 'trace' and row['path'] == path.relative_to(root).as_posix())
            if s['rank'] != expected:
                raise ValueError('trace metadata rank differs from manifest rank')
        if strict and s['cycles_kept'] > 22:
            raise ValueError('more than 22 complete cycles: capture exceeds 24-iteration contract')
        if strict and s['cycles_kept'] < 14:
            raise ValueError('fewer than 14 complete decode cycles (16-24 iterations required)')
        if s['rank'] in compact:
            raise ValueError('multiple traces for one rank; pass ONE cell directory')
        if strict:
            graph_doc = json.loads(next((root/r['path']).read_text() for r in manifest['files'] if r['role']=='graph_sizes'))
            descriptors = graph_doc['ranks'][str(s['rank'])]['descriptors']
            expected = {str(d['graph_id']) for d in descriptors if 'graph_id' in d}
            if not expected or set(s['target_graph_frequencies']) != expected:
                raise ValueError('trace lacks explicit target graph coverage or has unexpected graph IDs')
            if not math.isclose(s['executed_passes']['mean'], manifest['executed_passes'], rel_tol=1e-6):
                raise ValueError('executed-pass evidence differs from resolved trace distribution')
            if any(int(count) < 1 or int(count) > k for count in s['pass_count_frequencies']):
                raise ValueError('trace contains executed pass count outside capture K')
            if c > 1 and set(s['pass_count_frequencies']) != {'2'}:
                raise ValueError('uniform K2 requires exactly two executed passes in every cycle')
            for row in rs:
                if (row['projection'] == 'head' and row['phase'].startswith('draft') and
                        row['quantization'] != manifest['head']):
                    raise ValueError('bound draft-head kernel contradicts capture head evidence')
        summaries.append(s)
        rows.extend(rs)
        compact[s['rank']] = cs
        sources.append({'path': path.relative_to(root).as_posix(), 'sha256': sha256(path), 'status': 'measured'})
    lead = min(summaries, key=lambda s:s['rank'])
    categories = dict(lead['category_ms'])
    peer_categories = {cat: stats([s['category_ms'].get(cat, 0) for s in summaries])
                       for cat in sorted({cat for s in summaries for cat in s['category_ms']})}
    reference = manifest.get('unprofiled_cycle_ms') if manifest else None
    return {'schema': 1, 'status': 'measured-time/estimated-traffic', 'c': c, 'k': k,
            'domain': manifest.get('domain', 'unknown') if manifest else 'unknown',
            'stack': ledger['stack'], 'ledger_status': 'derived/assumed', 'sources': sources,
            'ranks': summaries, 'kernels': rows, 'category_ms': categories,
            'reference_rank':lead['rank'], 'category_peer_stats':peer_categories,
            'unprofiled_cycle_ms': reference, 'rank_skew': compare_ranks(compact),
            'head': manifest['head'] if manifest else ('bf16' if ledger['stack'].startswith('historical') else 'unknown'),
            'policy': manifest['policy'] if manifest else ('fixed' if ledger['stack'].startswith('historical') else 'unknown'),
            'executed_passes': manifest.get('executed_passes') if manifest else None,
            'pass_count_frequencies': lead['pass_count_frequencies'],
            'acceptance': acceptance(next(root/r['path'] for r in manifest['files'] if r['role']=='prom_before'),
                                     next(root/r['path'] for r in manifest['files'] if r['role']=='prom_after'), k) if manifest else None,
            'normalization_status': 'measured companions' if manifest else 'unknown; explicit capture state required before Model use',
            'notes': ['Kernel sums overlap across streams; shares may sum above one.',
                      'Category sum and cycle union differ; cyclesim uses a signed residual closure.',
                      'Expert traffic needs captured topk_ids; ledger union is assumed.']}


def write_outputs(out, doc):
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'kernels.json', doc)
    fields = ['id', 'rank', 'name', 'grid', 'phase', 'class', 'category', 'calls_per_cycle', 'total_us_per_cycle', 'projection', 'quantization',
              'share_of_cycle', 'estimated_bytes_read', 'bytes_status', 'bytes_basis', 'working_set_bytes',
              'achieved_gbs', 'gbs_status', 'dram_ceiling_fraction', 'l2_resident', 'l2_status']
    with (out / 'kernels.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for row in doc['kernels']:
            w.writerow({k: json.dumps(row[k]) if isinstance(row[k], (dict, list)) else row[k] for k in fields})
    lines = ['# Per-kernel decode profile', '', 'Times measured; bytes and GB/s estimated, unknown where unbound.', '',
             '| rank | kernel id / name / class | calls mean / median | us mean / median [p10,p90] | cycle share | bytes | GB/s | L2 heuristic |',
             '|---:|---|---:|---|---:|---:|---:|---|']
    for row in doc['kernels']:
        t = row['total_us_per_cycle']
        lines.append('| %s | %s / %s | %.1f / %.1f | %.2f / %.2f [%.2f,%.2f] | %.3f | %s | %s | %s |' %
                     (row['rank'], row['id'] + ' ' + row['name'][:140].replace('|', '/'), row['class'], row['calls_per_cycle']['mean'], row['calls_per_cycle']['median'], t['mean'], t['median'], t['p10'], t['p90'],
                      row['share_of_cycle'], row['estimated_bytes_read'] if row['estimated_bytes_read'] is not None else 'unknown',
                      '%.2f' % row['achieved_gbs'] if row['achieved_gbs'] is not None else 'unknown', row['l2_resident']))
    lines += ['', '| rank | cycles kept | cycle ms [p10,p90] | kernel union ms | GPU idle / host gap ms | kernels/cycle |',
              '|---:|---:|---|---:|---:|---:|']
    for s in doc['ranks']:
        t = s['cycle_ms']
        lines.append('| %d | %d | %.3f [%.3f,%.3f] | %.3f | %.3f | %.0f |' %
                     (s['rank'], s['cycles_kept'], t['median'], t['p10'], t['p90'], s['kernel_union_ms']['median'],
                      s['idle_ms']['median'], s['kernel_count']['median']))
    lines += ['', 'Full kernel names and all per-rank cycle/idle/count quantiles: kernels.json and kernels.csv.', '',
              'Rank skew: ' + json.dumps(doc.get('rank_skew', {})), '']
    (out / 'kernels.md').write_text('\n'.join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trace_dir', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--c', type=int, choices=(1, 2, 4), default=1)
    ap.add_argument('--k', type=int, choices=(1, 2, 3, 4), default=3)
    ap.add_argument('--ledger', type=Path, required=True)
    ap.add_argument('--target-norms', type=int)
    ap.add_argument('--check', action='store_true', help='strict manifest, companion, hash and decode-cycle validation')
    a = ap.parse_args(argv)
    try:
        doc = ingest(a.trace_dir, json.loads(a.ledger.read_text()), a.c, a.k, a.target_norms, a.check)
        if a.out:
            write_outputs(a.out, doc)
        elif not a.check:
            ap.error('--out required unless --check')
        print(json.dumps({'status': doc['status'], 'ranks': len(doc['ranks']), 'kernel_rows': len(doc['kernels'])}))
    except (ValueError, KeyError, OSError, EOFError) as e:
        ap.exit(2, 'capture rejected: %s\n' % e)


if __name__ == '__main__':
    main()
