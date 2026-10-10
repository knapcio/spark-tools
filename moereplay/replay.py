# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 knapcio
"""Observed steps or explicitly labeled prefill proxies, streamed by layer."""
from collections import Counter
import itertools
import json
import random
from moereplay.model import Costs, LAYOUTS, BYTES_TP4, placement, rank_work, bootstrap, MIN_BOOTSTRAP

PLACEMENTS = ('contiguous', 'round-robin', 'greedy')
DEFAULT_ROWS_PER_REQUEST = {1: 4, 2: 3, 4: 3}


def proxy_widths(overrides=None):
    widths = DEFAULT_ROWS_PER_REQUEST.copy()
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError('rows_per_request must map concurrency to row count')
        for c, width in overrides.items():
            if type(c) is not int or c not in widths or type(width) is not int or not 2 <= width <= 16:
                raise ValueError('rows_per_request requires c in 1,2,4 and rows in [2,16]')
            widths[c] = width
    return widths


def parse_proxy_widths(value):
    widths = {}
    try:
        for item in value.split(','):
            c, width = map(int, item.split('='))
            if c in widths:
                raise ValueError('duplicate concurrency')
            widths[c] = width
        return proxy_widths(widths)
    except (ValueError, TypeError) as e:
        raise ValueError('rows-per-request must be comma-separated c=rows (c:1,2,4; rows:2..16)') from e


def train_frequency(trace, layer, train_steps=None, train_requests=None):
    freq = Counter()
    for rows in trace.steps():
        for r in rows:
            if r['padding'] or train_steps is not None and r['step'] not in train_steps:
                continue
            if train_requests is not None and r['request'] not in train_requests:
                continue
            freq.update(trace.row(layer,r['row']))
    if not freq:
        raise ValueError('no held-out training rows')
    return freq


def decode_events(trace, layer, excluded_steps):
    for rows in trace.steps():
        if rows[0]['step'] in excluded_steps:
            continue
        real = [r for r in rows if not r['padding']]
        yield dict(step=rows[0]['step'], c=rows[0]['concurrency'], M=len(rows), k=rows[0]['k'],
                   requests=sorted(set(r['request'] for r in real)),
                   routes=[trace.row(layer, r['row']) for r in rows],
                   status=trace.manifest['source']['status'])


def prefill_events(trace, layer, test_requests, samples, seed, rows_per_request=None):
    widths = proxy_widths(rows_per_request)
    counts = {width: Counter(p for p,_ in iter_segments(trace,test_requests,width))
              for width in set(widths.values())}
    for width, per_prompt in counts.items():
        if any(not per_prompt[p] for p in test_requests):
            raise ValueError('prefill prompt needs an eligible consecutive %d-row segment within a step' % width)
    rng = random.Random(seed)
    plans, needed = [],set()
    for c in (1,2,4):
        width = widths[c]
        for step in range(samples):
            # Replacement is explicit; these are not independent decode requests.
            prompts = (rng.sample(test_requests,c) if c <= len(test_requests)
                       else [rng.choice(test_requests) for _ in range(c)])
            keys = [(width,p,rng.randrange(counts[width][p])) for p in prompts]
            needed.update(keys)
            plans.append((c,step,prompts,keys))
    # Stream each distinct width; retain sampled indices only.
    indices = {}
    for width in sorted(counts):
        seen = Counter()
        for p,rows in iter_segments(trace,test_requests,width):
            key = (width,p,seen[p]);seen[p] += 1
            if key in needed:
                indices[key] = rows
    for c,step,prompts,keys in plans:
        routes = [trace.row(layer,i) for key in keys for i in indices[key]]
        yield dict(step=step,c=c,M=widths[c]*c,k=widths[c]-1,
                   requests=sorted(set(prompts)),routes=routes,status='assumed')


def iter_segments(trace, requests, width=4):
    """All consecutive real segments, including the last; no request/step crossing."""
    requests = set(requests)
    for step in trace.steps():
        groups = {}
        for r in step:
            if not r['padding'] and r['request'] in requests:
                groups.setdefault(r['request'],[]).append(r)
        for request, rows in groups.items():
            for start in range(len(rows)-width+1):
                chunk = rows[start:start+width]
                if all(r['position']==chunk[0]['position']+i for i,r in enumerate(chunk)):
                    yield request,tuple(r['row'] for r in chunk)


def eligible_segments(trace, requests, width=4):
    """Small-fixture inspection helper; production sampling uses iter_segments."""
    segments = {}
    for request,rows in iter_segments(trace,requests,width):
        segments.setdefault(request,[]).append(rows)
    return segments


def union_growth(trace, layer, sizes=(1,2,3,4,5,6,8,12,16), limit=None):
    # Real consecutive positions within a request AND step. Never crosses rejected
    # verify boundaries or concatenates different requests into a c1 growth curve.
    sums, counts = Counter(), Counter()
    for step in trace.steps():
        groups = {}
        for r in step:
            if not r['padding']:
                groups.setdefault(r['request'], []).append(r)
        for rows in groups.values():
            for start in range(len(rows)):
                seen = set()
                for offset in range(min(max(sizes),len(rows)-start)):
                    if rows[start+offset]['position'] != rows[start]['position']+offset:
                        break
                    seen.update(trace.row(layer, rows[start+offset]['row']))
                    m = offset + 1
                    if m in sizes and (limit is None or counts[m] < limit):
                        sums[m] += len(seen)
                        counts[m] += 1
    return {str(m):dict(U=sums[m]/counts[m] if counts[m] else 'unknown', segments=counts[m],
                        status=trace.manifest['source']['status'],
                        coverage='all eligible segments' if limit is None else 'first eligible segments (bounded)',
                        scope='consecutive real rows within request and step') for m in sizes}


def replay(trace, cost_fit=None, samples=1000, bootstrap_samples=200, seed=1234,
           replicate_h=0, details=None, extrapolate_layers=None, train_trace=None, rows_per_request=None):
    widths = proxy_widths(rows_per_request)
    if type(samples) is not int or samples < 1 or type(bootstrap_samples) is not int or bootstrap_samples != 0 and bootstrap_samples < MIN_BOOTSTRAP:
        raise ValueError('samples >=1; bootstrap 0 or >=200 required')
    if type(replicate_h) is not int or not 0 <= replicate_h <= 256:
        raise ValueError('replicate_h must be in [0,256]')
    trace.check()
    if train_trace is not None:
        train_trace.check()
        if train_trace.manifest['model'] != trace.manifest['model'] or train_trace.manifest['layers'] != trace.manifest['layers']:
            raise ValueError('training trace model/layers differ')
        if train_trace.root == trace.root or train_trace.manifest['data'] == trace.manifest['data']:
            raise ValueError('training trace must differ from test data')
        test_ids = {r['request'] for r in trace.rows() if not r['padding']}
        train_ids = {r['request'] for r in train_trace.rows() if not r['padding']}
        if test_ids & train_ids:
            raise ValueError('external training requires disjoint request IDs')
    costs = Costs(cost_fit)
    steps = [rows[0]['step'] for rows in trace.steps()]
    proxy = trace.manifest['step_semantics'] == 'prefill-proxy'
    if proxy:
        prompts = sorted(set(r['request'] for r in trace.rows()))
        if len(prompts) < 2 and train_trace is None:
            raise ValueError('prefill proxy requires >=2 prompts for holdout')
        split = max(1,len(prompts)//2)
        train_requests = prompts[:split] if train_trace is None else None
        test_requests = prompts[split:] if train_trace is None else prompts
        train_steps = set()
        holdout = dict(status='assumed', unit='prompt', train=train_requests or 'external', test=test_requests,
                       c4_limitation='replacement sampling when fewer than four test prompts')
    else:
        if len(steps) < 2 and train_trace is None:
            raise ValueError('decode replay needs >=2 steps or external train trace')
        train_steps = set(steps[:max(1,len(steps)//2)]) if train_trace is None else set()
        train_requests = None
        holdout = dict(status='assumed',unit='chronological steps',training_steps=len(train_steps),
                       test_steps=len(steps)-len(train_steps),prompt_overlap='possible; use external train trace for prompt holdout')
    layers = trace.manifest['layers']
    scale = 1.
    if extrapolate_layers is not None:
        if extrapolate_layers < len(layers):
            raise ValueError('extrapolation count smaller than captured layers')
        scale = extrapolate_layers/len(layers)
    # Store scalar aggregates only; raw routes never accumulate over layers/steps.
    totals, clusters, bank_totals = {}, {}, {}
    layer_summaries, growth, fit_records, placement_fits = {}, {}, {}, {}
    stream = open(details,'x') if details else None
    try:
        for layer in layers:
            training = train_trace or trace
            freq = train_frequency(training,layer, None if train_trace else train_steps if not proxy else None,
                                   None if train_trace else train_requests)
            hottest = sorted(range(256),key=lambda e: (-freq[e],e))[:replicate_h]
            owners = {(lay,p): placement(n,p,freq) for lay,(n,_) in LAYOUTS.items() for p in PLACEMENTS}
            placement_fits[str(layer)] = dict(status=training.manifest['source']['status'],
                training_rows=sum(freq.values())//8,frequencies=[freq[e] for e in range(256)],
                hottest=hottest,owners={'/'.join(key):value for key,value in owners.items()})
            layer_banks = {}
            for layout,(n,tp) in LAYOUTS.items():
                for name in list(PLACEMENTS) + (['replicate-hottest'] if replicate_h else []):
                    owner = owners[layout,'greedy' if name=='replicate-hottest' else name]
                    counts = Counter(owner)
                    hot_counts = Counter(owner[e] for e in hottest) if name=='replicate-hottest' else Counter()
                    banks = [(counts[g]-hot_counts[g]+(replicate_h if name=='replicate-hottest' else 0))*BYTES_TP4*n
                             for g in range(n) for _ in range(tp)]
                    layer_banks[layout,name] = banks
                    total = bank_totals.setdefault((layout,name),[0.]*4)
                    for rank in range(4):
                        total[rank] += banks[rank]*scale
            events = (prefill_events(trace,layer,test_requests,samples,seed,widths) if proxy else decode_events(trace,layer,train_steps))
            layer_values = {}
            rng = random.Random(seed+layer)
            for ev in events:
                c, m = ev['c'],ev['M']
                uniform = [bytes(rng.sample(range(256),8)) for _ in range(m)]
                for control,routes in (('observed',ev['routes']),('uniform-random',uniform)):
                    baseline_work = rank_work(routes,'TP4',owners['TP4','contiguous'])
                    baseline_time = costs.evaluate('TP4',m,baseline_work['D_r'],baseline_work)
                    base = baseline_time['layer_us']/1000
                    for layout in LAYOUTS:
                        names = list(PLACEMENTS) + (['replicate-hottest'] if replicate_h else [])
                        for name in names:
                            p = 'greedy' if name == 'replicate-hottest' else name
                            work = rank_work(routes,layout,owners[layout,p],hottest if name == 'replicate-hottest' else ())
                            time = costs.evaluate(layout,m,work['D_r'],work)
                            eligible = time['calibration_eligible'] and baseline_time['calibration_eligible']
                            ms, delta = time['layer_us']/1000,(time['layer_us']/1000-base)
                            cell = 'c%d_M%d_K%d' % (c,m,ev['k'])
                            key = (control,cell,layout,name)
                            event_id = ev['step']
                            totals.setdefault(key,{}).setdefault(event_id,[0.,0.,True,set(),True,True])
                            totals[key][event_id][0] += ms*scale
                            totals[key][event_id][1] += delta*scale
                            totals[key][event_id][2] &= eligible
                            totals[key][event_id][4] &= time['calibration_eligible']
                            totals[key][event_id][5] &= baseline_time['calibration_eligible']
                            totals[key][event_id][3].update('candidate: '+r for r in time['calibration_reasons'])
                            totals[key][event_id][3].update('TP4 baseline: '+r for r in baseline_time['calibration_reasons'])
                            clusters[(cell,event_id)] = ev['requests']
                            vals = layer_values.setdefault(key,dict(D_union=[],maxD=[],D_r=[[] for _ in range(4)],bytes_r=[[] for _ in range(4)],slots_r=[[] for _ in range(4)],ms=[],extrapolated=[],unknown_range=[],eligible=[],baseline_eligible=[],delta_eligible=[]))
                            vals['D_union'].append(work['D_union']); vals['maxD'].append(work['max_rank_D']); vals['ms'].append(ms); vals['extrapolated'].append(time['extrapolation'] is True); vals['unknown_range'].append(time['extrapolation']=='unknown')
                            vals['eligible'].append(time['calibration_eligible']); vals['baseline_eligible'].append(baseline_time['calibration_eligible']); vals['delta_eligible'].append(eligible)
                            for rank in range(4):
                                vals['D_r'][rank].append(work['D_r'][rank]); vals['bytes_r'][rank].append(work['bytes_r'][rank])
                                vals['slots_r'][rank].append(work['routed_slots_r'][rank])
                            fit_records['%s_M%d' % (layout,m)] = time['fit']
                            if stream:
                                stream.write(json.dumps(dict(layer=layer,control=control,cell=cell,step=ev['step'],layout=layout,
                                      placement=name,route_status=ev['status'] if control=='observed' else 'assumed',
                                      work=work,time=time,TP4_baseline_calibration_eligible=baseline_time['calibration_eligible'],
                                      TP4_baseline_calibration_reasons=baseline_time['calibration_reasons'],
                                      delta_calibration_eligible=eligible,delta_vs_TP4_ms=delta),separators=(',', ':'))+'\n')
            import statistics
            layer_summaries[str(layer)] = {'/'.join(key): dict(D_union=statistics.fmean(v['D_union']),
                 max_rank_D=statistics.fmean(v['maxD']),D_r=[statistics.fmean(x) for x in v['D_r']],
                 bytes_r=[statistics.fmean(x) for x in v['bytes_r']],bytes_status='estimated',
                 routed_slots_r=[statistics.fmean(x) for x in v['slots_r']],
                 calibration_eligible_fraction=statistics.fmean(v['eligible']),
                 TP4_baseline_eligible_fraction=statistics.fmean(v['baseline_eligible']),
                 delta_calibration_eligible_fraction=statistics.fmean(v['delta_eligible']),
                 layer_ms=statistics.fmean(v['ms']),time_status='estimated',
                 fit_extrapolation_fraction=sum(v['extrapolated'])/(len(v['extrapolated'])-sum(v['unknown_range']))
                     if sum(v['unknown_range'])<len(v['unknown_range']) else 'unknown',
                 unknown_calibration_range_fraction=statistics.fmean(v['unknown_range']),
                 bank_bytes_r=layer_banks[key[2],key[3]],bank_bytes_status='estimated',
                 routes_status=trace.manifest['source']['status'] if not proxy and key[0]=='observed' else 'assumed')
                 for key,v in layer_values.items()}
            growth[str(layer)] = union_growth(trace,layer)
        result = {}
        whatifs = []
        for (control,cell,layout,name),events in totals.items():
            event_ids = sorted(events)
            groups = [clusters[cell,i] for i in event_ids]
            times = bootstrap([events[i][0] for i in event_ids],groups,bootstrap_samples,seed)
            deltas = bootstrap([events[i][1] for i in event_ids],groups,bootstrap_samples,seed)
            eligible = all(events[i][2] for i in event_ids) and scale==1 and not proxy
            reasons = sorted(set().union(*(events[i][3] for i in event_ids)))
            if scale!=1:
                reasons.append('layer extrapolation')
            if proxy:
                reasons.append('prefill proxy is not decode evidence')
            elif trace.manifest['source'].get('route_stage')!='post-kstop-remap':
                eligible = False
                reasons.append('post-kstop-remap route provenance unknown')
            if control!='observed':
                eligible = False
                reasons.append('uniform random control is a sensitivity')
            if trace.manifest['source']['status']!='measured':
                eligible = False
                reasons.append('route source is assumed')
            deltas.update(calibration_eligible=eligible,calibration_reasons=reasons,
                          eligible_steps=sum(events[i][2] for i in event_ids))
            times.update(calibration_eligible=all(events[i][4] for i in event_ids),
                         scope='routed-kernel calibration only; layer/route assumptions remain separate')
            n,tp = LAYOUTS[layout]
            key = '/'.join((control,cell,layout,name))
            banks = bank_totals[layout,name]
            result[key] = dict(cycle_kernel_ms=times,delta_vs_TP4_ms=deltas,bank_bytes_r=banks,
                               TP4_baseline_calibration_eligible=all(events[i][5] for i in event_ids),
                               bank_bytes_status='estimated', bank_scope='sum over modeled routed layers',
                               supported_in_pinned_vllm=None,
                               serving_compatible=None,
                               serving_limitation='unknown: no runtime pin or compatibility check; placement and collective behavior are assumptions',
                               placement_status='assumed', replicate_h=replicate_h if name=='replicate-hottest' else 0)
            parts=cell.split('_')
            shape=dict(concurrency=int(parts[0][1:]),physical_rows=int(parts[1][1:]),K=int(parts[2][1:]))
            whatifs.append(dict(control=control,cell=cell,layout=layout,placement=name,status='estimated',
                                calibration_eligible=eligible, calibration_reasons=reasons,
                                required_shape=shape,
                                modeled_layer_count=extrapolate_layers or len(layers),
                                compatibility='caller must match cyclesim concurrency, physical width, K and capture layout; proxy defaults c1/M4/K3, c2/M6/K2, c4/M12/K2 (reuse)',
                                kernel_ms={'moe_routed':deltas['mean']} if eligible else {},
                                sensitivity_kernel_ms={'moe_routed':deltas['mean']},
                                argv=['--c',str(shape['concurrency']),'--k',str(shape['K']),
                                      '--kernel-ms','moe_routed=%.9g' % deltas['mean']] if eligible else [],
                                sensitivity_only=not eligible,
                                scope='target verify routed-kernel delta; drafting and collectives unchanged'))
        return dict(schema='moe-replay v1',status='estimated', source=trace.manifest['source'],
                    route_semantics=trace.manifest['step_semantics'],holdout=holdout,
                    proxy_rows_per_request={str(c):w for c,w in widths.items()} if proxy else None,
                    layers=layers,layer_scale=dict(status='assumed',factor=scale,
                       modeled_layers=extrapolate_layers or len(layers)),
                    collective=dict(status='assumed',hidden_states='replicated; DP1, PCP1, SP off',
                       per_layer='one TP4 all-reduce; no all2all',delta_ms=0,
                       source='caller must validate collective behavior for the target runtime',
                       EP2xTP2='hypothetical; same four-rank reduction assumed'),
                    replica_policy=dict(status='assumed',dispatch='hot occurrence to least occupied EP group; one execution',
                       availability='requires runtime changes; no throughput claim'),
                    capture_requests=['real decode routes for layers 3..77; c1 prose/code and c4 mixed, >=2000 steps per panel',
                                      'per-layout,M EP benchmark fit; out-of-D-range timing calibration',
                                      'EP4 overlay compatibility and collective-inclusive qualification'],
                    placement_fits=placement_fits,cost_fits=fit_records,cost_fit_adapter=getattr(costs,'adapter','unknown'),cycles=result,per_layer=layer_summaries,union_growth=growth,
                    cyclesim_what_if=whatifs,
                    limits=['prefill routes are not decode evidence','bootstrap excludes model/calibration uncertainty',
                            'bytes are weight footprint, not measured DRAM traffic','no router/shared/draft/communication gain modeled'])
    finally:
        if stream:
            stream.close()
