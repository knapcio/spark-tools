# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 knapcio
"""Existing offline bracket screens and native trajectory pair statistics.

These screens compare the candidate with each surrounding baseline separately.
They do not imply release qualification or alter any runtime configuration.
"""
import math
import random
import statistics
from abharness.logic import _compare_t0_exact


PROFILES = frozenset(('glue', 'async', 'trim', 'tau', 'k4'))


def _validate_tokens(row):
    ids = row.get('token_ids')
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('invalid native token IDs')


def _validate_observation(row):
    _validate_tokens(row)
    for field in ('cycles', 'committed_tokens', 'seconds', 'committed', 'cycle_ms', 'tps'):
        value = row.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('invalid screening observation: ' + field)
        if field in ('cycles', 'committed_tokens') and type(value) is not int:
            raise ValueError('invalid screening observation: ' + field)


def exact_t0(a, b, *, repeats=3):
    result = _compare_t0_exact(a, b, repeats=repeats)
    return result


def hard_equal(a, b):
    if set(a) != set(b) or not a:
        raise ValueError('incomplete token panel')
    for row in (*a.values(), *b.values()):
        _validate_tokens(row)
    for key in a:
        if a[key]['metadata'] != b[key]['metadata']:
            raise ValueError('prompt identity drift')
        if a[key]['token_ids'] != b[key]['token_ids']:
            return False
    return True


def bracket(before, after, candidate, scorer, *, exact=False, saving_ms=.15, profile='glue'):
    if profile not in PROFILES:
        raise ValueError('unsupported screening profile')
    for panel in (before, after, candidate):
        for row in panel.values():
            _validate_observation(row)
    if not hard_equal(before, after):
        raise ValueError('baseline controls must have HARD token-ID equality')
    if set(before) != set(candidate):
        raise ValueError('candidate panel incomplete')
    groups={}; reasons=[]
    kinds={r['metadata']['kind'] for r in before.values()}
    if profile=='tau' and any(r['metadata'].get('thinking') is True for r in before.values()):kinds.add('thinking')
    for kind in kinds:
        keys=sorted(k for k in before if (before[k]['metadata'].get('thinking') is True if kind=='thinking' else before[k]['metadata']['kind']==kind))
        groups[kind]=[scorer(base,candidate,keys) for base in (before,after)]
        for result in groups[kind]:
            if exact and any(not p['identical'] for p in result['pairs']):reasons.append('HARD token-ID divergence')
            if not exact and any(not p['near_tie'] for p in result['pairs']):reasons.append('non-near-tie divergence')
    if profile in ('glue','async','trim'):
        for kind,results in groups.items():
            for result in results:
                pairs=result['pairs'];savings=[p['A_cycle_ms']-p['B_cycle_ms'] for p in pairs]
                rnd=random.Random(1729)
                ci=sorted(statistics.mean(rnd.choices(savings,k=len(savings))) for _ in range(4000))
                result['saving_ms']=statistics.mean(savings);result['saving_ci95_ms']=[ci[100],ci[3899]]
                if kind=='prose' and profile!='trim' and (result['saving_ms']<saving_ms or ci[100]<=0):reasons.append('prose saving/CI')
                if kind=='code' or profile=='trim':
                    if statistics.mean(p['B_cycle_ms']/p['A_cycle_ms']-1 for p in pairs)>.005:reasons.append('cycle regression >0.5%')
                if statistics.mean(p['B_committed']/p['A_committed'] for p in pairs)<.995:reasons.append('committed <99.5%')
    else:
        for kind,results in groups.items():
            for result in results:
                if profile=='tau':
                    if kind=='prose' and (result['ci95'][0]<=1):reasons.append('tau prose CI')
                    if kind in ('code','thinking') and result['ratio']<1:reasons.append('tau code regression')
                elif profile=='k4':
                    if kind in ('code','structured') and (result['ratio']<1.01 or result['ci95'][0]<=1):reasons.append('K4 code/structured gain')
                    if kind=='prose' and (result['ratio']<.997 or result['ci95'][0]<.995):reasons.append('K4 prose regression')
    return dict(decision='PASS' if not reasons else 'HOLD',reasons=sorted(set(reasons)),groups=groups,
        token_policy='HARD_ID_EQUALITY' if exact else 'FIRST_DIVERGENCE_NEAR_TIE',profile=profile,
        brackets=['before','after'],release_gate=False)


NEAR_TIE=dict(max_margin_nats=.5,top_k=5,position='first divergence')

def det_flip(a,b):
    _validate_tokens(a)
    _validate_tokens(b)
    x,y=a['token_ids'],b['token_ids'];n=min(len(x),len(y))
    i=next((i for i in range(n) if x[i]!=y[i]),None)
    if x==y:return dict(near_tie=True,margin_nats=None,B_in_A_top5=None)
    if i is None:return dict(near_tie=False,margin_nats=None,B_in_A_top5=False)
    top=a.get('top_logprobs',[])
    if len(top)!=len(x):return dict(near_tie=False,margin_nats=None,B_in_A_top5=False)
    row=top[i]
    try:
        if (len(row) != 5 or any(type(v['token_id']) is not int or v['token_id'] < 0
                                or type(v['logprob']) not in (int, float)
                                or not math.isfinite(v['logprob']) for v in row)):
            raise ValueError('invalid top-5 row')
        probs = {v['token_id']: v['logprob'] for v in row}
        if (len(probs) != 5 or row[0]['token_id'] != x[i]
                or row[0]['logprob'] != max(probs.values())):
            raise ValueError('baseline top-1 mismatch')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('invalid frozen top-5 evidence') from exc
    present = y[i] in probs
    margin = probs[x[i]] - probs[y[i]] if present else None
    return dict(near_tie=present and margin<=NEAR_TIE['max_margin_nats'],margin_nats=margin,B_in_A_top5=present)

def det_pair_stats(a,b,keys):
    import random
    if not keys:raise ValueError('empty paired subgroup')
    for k in keys:
        _validate_observation(a[k])
        _validate_observation(b[k])
        if a[k]['metadata']!=b[k]['metadata'] or a[k]['time_source']!=b[k]['time_source']:raise ValueError('paired prompt identity drift')
    def ratio(draw):return (sum(b[k]['committed_tokens'] for k in draw)/sum(b[k]['seconds'] for k in draw))/(sum(a[k]['committed_tokens'] for k in draw)/sum(a[k]['seconds'] for k in draw))
    rnd=random.Random(20261009);draws=sorted(ratio([rnd.choice(keys) for _ in keys]) for _ in range(4000))
    pairs=[]
    for k in keys:
        x,y=a[k],b[k];tx,ty=x['token_ids'],y['token_ids'];n=min(len(tx),len(ty))
        pairs.append(dict(**det_flip(x,y),pair_key=k,identical=tx==ty,first_divergence=next((i for i in range(n) if tx[i]!=ty[i]),n if len(tx)!=len(ty) else None),token_differences=sum(x!=y for x,y in zip(tx,ty))+abs(len(tx)-len(ty)),A_committed=x['committed'],B_committed=y['committed'],A_cycle_ms=x['cycle_ms'],B_cycle_ms=y['cycle_ms'],A_tps=x['tps'],B_tps=y['tps']))
    return dict(ratio=ratio(keys),ci95=[draws[100],draws[3899]],pairs=pairs,divergent_prompts=sum(not p['identical'] for p in pairs),n=len(keys))
