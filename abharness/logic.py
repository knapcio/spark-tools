# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 knapcio
"""Existing paired bracket and first-divergence gate; offline decisions only."""
import json
import math
import random
import re
import statistics

T0_POLICY = {"draft": "draft_only", "target": "target_changing", "neutral": "target_neutral"}

def byte_output(output):
    # Preserve all returned token/logprob bits, channel text, length and finish reason.
    if not output or output.get('basis') == 'text_characters':
        raise ValueError('native token representation unavailable: HOLD')
    if output.get('basis') == 'token_ids' and not output.get('token_ids'):
        raise ValueError('native token IDs unavailable: HOLD')
    generated={k:output.get(k) for k in ('basis','sequence','token_ids','tokens','text','reasoning_text','finish_reason')}
    generated['completion_tokens']=(output.get('usage') or {}).get('completion_tokens')
    return json.dumps(generated, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def native_ids(output):
    """vLLM exposes native IDs in logprob token strings, even with a byte basis label."""
    if not output: raise ValueError('native token IDs unavailable: HOLD')
    ids = output.get('token_ids')
    if ids is not None:
        if not isinstance(ids,list) or not ids or any(type(i) is not int or i<0 for i in ids):
            raise ValueError('invalid native token IDs: HOLD')
        return ids
    tokens=output.get('tokens')
    if not isinstance(tokens,list) or not tokens:raise ValueError('native token IDs unavailable: HOLD')
    result=[]
    for token in tokens:
        match=re.fullmatch(r'token_id:(0|[1-9][0-9]*)',token.get('token',''))
        if not match:raise ValueError('native token IDs unavailable: HOLD')
        result.append(int(match[1]))
    return result


def id_diff(a,b):
    indices=[i for i in range(max(len(a),len(b))) if i>=len(a) or i>=len(b) or a[i]!=b[i]]
    return dict(count=len(indices),first_index=indices[0] if indices else None)


def _compare_t0_exact(a, b, *, repeats=3):
    if repeats not in (1,3):raise ValueError('T0 repeats must be 1 or 3')
    for key in ('panel_sha256', 'model', 'configuration', 'apc_mode'):
        if a.get(key) != b.get(key): raise ValueError('unmatched T0 ' + key)
    if a.get('status') != 'COMPLETE' or b.get('status') != 'COMPLETE' or a.get('apc_mode') != 'cold':
        raise ValueError('complete cold T0 required')
    if len(a.get('rows', [])) != 30 or len(b.get('rows', [])) != 30: raise ValueError('all 30 T0 rows required')
    if [r.get('row') for r in a['rows']]!=list(range(1,31)):raise ValueError('unique ordered T0 rows required')
    equal, diverging,details = [], [], []
    for left, right in zip(a['rows'], b['rows']):
        for key in ('row', 'category', 'prompt', 'nonce'):
            if left.get(key) != right.get(key): raise ValueError('T0 input mismatch ' + key)
        sequences=[]
        for row in (left, right):
            if row.get('gate_eligible') is not True or len(row.get('repeats', [])) != repeats:
                raise ValueError('T0 incomplete/ineligible row')
            ids=[]
            for r in row['repeats']:
                if not r.get('attempts') or any(x.get('prefix_cache_hits', {}).get('delta') != 0 or
                    x.get('reset', {}).get('http_status') != 200 for x in r['attempts']):
                    raise ValueError('T0 reset/hits evidence missing')
                ids.append(native_ids(r.get('output')))
            sequences.append(ids)
        differences=[]; byte_equal=[]
        for i,(lr,rr) in enumerate(zip(left['repeats'],right['repeats'])):
            if lr.get('request_body') != rr.get('request_body'):raise ValueError('T0 request bytes differ')
            differences.append(dict(repeat=i+1,**id_diff(sequences[0][i],sequences[1][i])))
            byte_equal.append(byte_output(lr['output'])==byte_output(rr['output']))
        unstable=[id_diff(seq[0],other) for seq in sequences for other in seq[1:]]
        mismatches=[x for x in differences+unstable if x['count']]
        first=min((x['first_index'] for x in mismatches),default=None)
        details.append(dict(row=left['row'],category=left['category'],count=max((x['count'] for x in mismatches),default=0),
            first_index=first,repeats=differences,repeats_stable=not any(x['count'] for x in unstable),
            output_bytes_equal=all(byte_equal)))
        (diverging if mismatches else equal).append(left['row'])
    reasons=['native token ID divergence; diagnose padded-batch near ties'] if diverging else []
    return dict(equal_rows=equal,diverging_rows=diverging,diverging_count=len(diverging),rows=details,
        index_base=0,native_ids=True,reasons=reasons,decision='HOLD' if reasons else 'PASS')


def compare_t0(a, b, variant=None, reference=None):
    result = _compare_t0_exact(a, b)
    if variant is None:
        return result  # Baseline self-validation and same-configuration MLA comparisons.
    policy = T0_POLICY.get(variant)
    if policy == 'target_neutral':
        return result
    if policy not in ('draft_only', 'target_changing'):
        raise ValueError('undeclared T0 policy for variant ' + variant)
    failures = []
    refs = {}
    for row in (reference or {}).get('rows', []):
        refs.setdefault(row.get('row'), []).append(row)
    for left, right, detail in zip(a['rows'], b['rows'], result['rows']):
        aid = native_ids(left['repeats'][0]['output'])
        bid = native_ids(right['repeats'][0]['output'])
        detail.update(b_repeats_stable=all(native_ids(r['output']) == bid for r in right['repeats']),
                      margin_nats=None, b_in_a_top5=None, first_token_equal=aid[0] == bid[0])
        issues = []
        if not detail['repeats_stable']:
            issues.append('within-arm native IDs unstable')
        if not detail['first_token_equal']:
            issues.append('first token differs')
        entries = refs.get(left['row'], [])
        # Both frozen reference repeats must reproduce the complete saved A sequence.
        if len(entries) != 2 or {r.get('repeat') for r in entries} != {1, 2} or any(
                [x.get('tok') for x in r.get('seq', [])] != aid for r in entries):
            issues.append('missing/mismatched A top5 reference')
        elif aid != bid:
            k = id_diff(aid, bid)['first_index']
            detail['first_index'] = k
            margins = []
            for entry in entries:
                try:
                    top = entry['seq'][k]['top']
                    if len(top) != 5 or len({x[0] for x in top}) != 5:
                        raise ValueError('invalid top5')
                    probs = dict(top)
                    if any(type(v) not in (int, float) or not math.isfinite(v) for v in probs.values()):
                        raise ValueError('invalid logprob')
                    if top[0][0] != aid[k] or top[0][1] != max(probs.values()):
                        raise ValueError('A top1 mismatch')
                    margins.append(top[0][1] - probs[bid[k]])
                except (KeyError, IndexError, TypeError, ValueError):
                    issues.append('B token absent from valid A top5 at first divergence')
            detail['b_in_a_top5'] = len(margins) == 2
            if margins:
                detail['margin_nats'] = max(margins)
                if max(margins) > 0.5:
                    issues.append('A margin exceeds 0.5 nats')
        detail['reasons'] = sorted(set(issues))
        if issues:
            failures.append(dict(row=left['row'], k=detail['first_index'],
                                 margin_nats=detail['margin_nats'], reasons=detail['reasons']))
    result.update(policy=policy, variant=variant, hold_rows=failures,
                  reasons=['T0 stability/reference/near-tie gate failed'] if failures else [],
                  decision='HOLD' if failures else ('PASS_NEAR_TIE' if result['diverging_count'] else 'PASS'))
    return result


def cycle_pairs(a, b):
    def index(report):
        rows = report.get('rows', [])
        result = {(r['kind'], r['i']):r for r in rows}
        if len(rows) != 20 or len(result) != 20 or set(result) != {(k,i) for k in ('prose','code') for i in range(10)}:
            raise ValueError('incomplete cycle panel: HOLD')
        return result
    left, right = index(a), index(b); pairs = []
    for key in sorted(left):
        x,y = left[key],right[key]
        for r in (x,y):
            if any(type(r.get(k)) not in (float,int) or not math.isfinite(r[k]) or r[k] <= 0
                   for k in ('cycle_ms','committed','tps')) or not r.get('text_sha256'):
                raise ValueError('invalid cycle observation: HOLD')
        pairs.append(dict(kind=key[0], i=key[1], a_cycle_ms=x['cycle_ms'], b_cycle_ms=y['cycle_ms'],
            saving_ms=x['cycle_ms']-y['cycle_ms'], a_committed=x['committed'],b_committed=y['committed'], committed_ratio=y['committed']/x['committed'],
            throughput_ratio=y['tps']/x['tps'], text_equal=x['text_sha256']==y['text_sha256']))
    return pairs


def bracket_pairs(a, aprime, b):
    """Pair each prompt before averaging: B - mean(A,Aprime), never panel medians."""
    left=cycle_pairs(a,b);right=cycle_pairs(aprime,b);result=[]
    for x,y in zip(left,right):
        reference=(x['a_cycle_ms']+y['a_cycle_ms'])/2
        result.append(dict(kind=x['kind'],i=x['i'],a_cycle_ms=x['a_cycle_ms'],aprime_cycle_ms=y['a_cycle_ms'],
            b_cycle_ms=x['b_cycle_ms'],reference_cycle_ms=reference,delta_ms=x['b_cycle_ms']-reference,
            saving_ms=reference-x['b_cycle_ms'],cycle_regression=x['b_cycle_ms']/reference-1,
            reference_committed=(x['a_committed']+y['a_committed'])/2,b_committed=x['b_committed'],
            committed_ratio=2/(1/x['committed_ratio']+1/y['committed_ratio']),
            text_equal=x['text_equal'] and y['text_equal'],
            committed_equal=x['committed_ratio']==1 and y['committed_ratio']==1))
    return result


def decision(a, aprime, b):
    """Three 20-prompt panels per configuration, separate prose/code crossed paired bootstrap."""
    if any(not isinstance(boot,list) or len(boot)!=3 for boot in (a,aprime,b)):
        raise ValueError('exactly three complete panels per configuration required: HOLD')
    blocks=[bracket_pairs(x,y,z) for x,y,z in zip(a,aprime,b)]
    groups={};rng=random.Random(1729)
    for kind in ('prose','code'):
        grouped=[[p for p in block if p['kind']==kind] for block in blocks]
        pairs=[p for block in grouped for p in block]
        samples=[]
        # Preserve repeated prompt identity across panels while resampling both factors.
        for _ in range(4000):
            panels=rng.choices(grouped,k=3);prompts=rng.choices(range(10),k=10)
            samples.append(statistics.mean(panel[i]['saving_ms'] for panel in panels for i in prompts))
        samples.sort()
        groups[kind]=dict(n=len(pairs),cycle_saving_ms=statistics.mean(p['saving_ms'] for p in pairs),
            delta_ms=statistics.mean(p['delta_ms'] for p in pairs),ci95_ms=[samples[100],samples[3899]],
            cycle_regression=statistics.mean(p['delta_ms'] for p in pairs)/statistics.mean(p['reference_cycle_ms'] for p in pairs),
            committed_ratio=statistics.mean(p['b_committed'] for p in pairs)/statistics.mean(p['reference_committed'] for p in pairs),
            text_equal=sum(p['text_equal'] for p in pairs),committed_equal=sum(p['committed_equal'] for p in pairs))
    pairs=[dict(panel=i+1,**p) for i,block in enumerate(blocks) for p in block]
    committed=statistics.mean(p['b_committed'] for p in pairs)/statistics.mean(p['reference_committed'] for p in pairs)
    prose=groups['prose'];code=groups['code']
    gates=dict(prose_saving=prose['cycle_saving_ms']>=.2-1e-12,prose_ci=prose['ci95_ms'][0]>0,
        committed=committed>=.995-1e-12,code_cycle=code['cycle_regression']<=.005+1e-12)
    return dict(decision='PASS' if all(gates.values()) else 'KILL',reasons=[k for k,v in gates.items() if not v],
        gates=gates,groups=groups,pairs=pairs,committed_ratio=committed,
        cycle_saving_ms=prose['cycle_saving_ms'],ci95_ms=prose['ci95_ms'],code_regression=code['cycle_regression'])

