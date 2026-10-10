# SPDX-License-Identifier: Apache-2.0
"""Explicit conditional scope backed by anonymised mixed-gap calibration."""
from lab.schedreplay.comparisons import compare
from lab.schedreplay.policy import Policy
from lab.schedreplay.replay import Replay

CAPTURE_REQUESTS = [
    {'regime':'isolated-2048-long-context',
     'request':'Isolated 2048-token chunk anchors above 16154 tokens (60K/100K/128K/250K), including launch overhead.'},
    {'regime':'mixed-512-1024',
     'request':'Matched 512/1024 mixed step timings and acceptance at one 60K decoder plus one 100K cold prefill.'},
    {'regime':'APC-startup',
     'request':'APC-warm startup and retained-prefix marginal timings with actual cached-token counts.'},
    {'regime':'decode-overlap-intervals',
     'request':'Decode-during-prefill overlap intervals: arrivals, prefill start/end, decoder output events and acceptance; include newcomer traffic.'},
    {'regime':'joint-prefills',
     'request':'Joint isolated/mixed prefill timings for concurrent newcomers with per-request context and chunk counts.'},
    {'regime':'concurrency-context-transfer',
     'request':'Same-configuration cycle/mixed timings and acceptance for c3/c4 at 60K/128K, with effective K and capture layout.'},
    {'regime':'c6',
     'request':'c6 allocator/graph/admission qualification and matched timing/acceptance contracts before ranking.'},
]

EXCLUSIONS = [
    'APC startup: A is conditioned to be already decoding at t=0; no startup TTFT/cost claim',
    'A startup, whole-run TTFT/throughput/queue/failure objectives',
    'decode throughput: excluded from primary objectives; including it can produce a tradeoff',
    'isolated prefill lower bound: this scope uses the measured mixed event-gap fit directly',
    'cross-chunk 2048/512/1024 extrapolation and joint prefills',
    'APC-dependent newcomer timings, other concurrency/context/domain and c6 qualification',
    'admission OFF, external KV floors and starvation-fix variants',
]


def supported_subset(works, metadata, timing_factory, mode='fast', floor=0):
    scope = {'name':'conditional-4096-one-decoder-cold-prefill',
             'members':'fair:4096:N, N in {0,10,20,40,80}; versus N40',
             'settings':{'mode':mode,'admission':True,'max_num_seqs':4,'configured_k':3,
                         'effective_decode_k':3,'queue_depth':2,'variant':'none','engine_floor_tokens':0},
             'objectives':['B_TTFT_s','A_event_gap_stall_s'],
             'condition':'one already-decoding 50K-75K prose stream plus one 90K-105K cold prefill',
             'timing':'estimated mixed event-gap fit, fixed c1 K3; serial two-deep queue',
             'exclusions':EXCLUSIONS,'capture_requests':CAPTURE_REQUESTS}
    reasons = []
    if len(works) != 2:
        reasons.append('scope requires exactly one decoder and one cold prefill')
    else:
        decoders = [w for w in works if w.arrival_s == 0 and w.kind == 'prose'
                    and 50000 <= w.prompt_tokens <= 75000
                    and w.cached_prefix_tokens >= w.prompt_tokens-1]
        prefills = [w for w in works if w.arrival_s > 0 and w.kind == 'prose'
                    and 90000 <= w.prompt_tokens <= 105000 and w.cached_prefix_tokens == 0]
        if len(decoders) != 1:
            reasons.append('A is not an already-decoding ~60K prose scenario input')
        if len(prefills) != 1:
            reasons.append('B is not a cold ~100K prose prefill arriving during A decode')
        if any(w.after_request_id or w.cancel_at_s is not None for w in works):
            reasons.append('conditional scope lacks dependency/cancellation timings')
    if floor: reasons.append('external engine KV floor is outside declared supported settings')
    if reasons:
        return {'ranking':[],**scope,'status':'indeterminate','blocking_reasons':reasons}
    a,b = decoders[0],prefills[0]
    results = []
    for n in (0,10,20,40,80):
        timing = timing_factory()
        timing.matched_mixed = True
        replay = Replay(Policy(chunk=4096,decode_steps=n),mode=mode,admission=True,timing=timing)
        result = replay.run(works,metadata,already_decoding=(a.request_id,))
        b_row = next(r for r in result['requests'] if r['request_id']==b.request_id)
        intervals = [w for w in result['decode_during_prefills']
                   if w['decode_request']==a.request_id and w['prefill_request']==b.request_id]
        blockers = list(result['unsupported_timing_regimes'])
        if (not intervals or b_row['first_s'] is None or
                replay.states[a.request_id].end <= b_row['first_s']):
            blockers.append('A output does not span B prefill: full decode-during-prefill overlap interval missing')
        result['scoped_metrics'] = {'B_TTFT_s':b_row['ttft_s'],
                                   'A_event_gap_stall_s':intervals[0]['max_stall_s'] if intervals else None}
        result['metric_support'] = {name:sorted(set(blockers)) for name in scope['objectives']}
        results.append(result)
    baseline = results[3]
    ranking = compare(results,baseline,scope['objectives'],scope['name'])
    # Per-objective comparisons remain valid even if another objective is missing.
    for row,result in zip(ranking,results):
        row['metric_comparisons'] = {name:compare([result],baseline,[name],scope['name'])[0]
                                     for name in scope['objectives']}
        row['decode_throughput_excluded_tok_s'] = result['decode_during_prefills_tok_s']
        row['estimate_basis'] = 'interpolated N10/extrapolated N80, unmeasured arms' if result['policy'].endswith((':10',':80')) else 'in-sample mixed-gap reconstruction; not validation'
    ranking.sort(key=lambda r:(r['pareto'] is not True,r['values']['B_TTFT_s']))
    return {'ranking':ranking,**scope,'status':'estimated-supported' if all(r['pareto'] is not None for r in ranking) else 'indeterminate',
            'blocking_reasons':sorted({reason for r in ranking for reason in r['ranking_withheld_reasons']})}
