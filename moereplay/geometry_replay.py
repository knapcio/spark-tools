# SPDX-License-Identifier: Apache-2.0
"""Geometry-aware offline analysis. Missing timing and weight data stay null."""
from collections import Counter
import json
import random
import statistics

from moereplay.geometry import resolve_layouts
from moereplay.model import Costs, placement, rank_work, bootstrap, MIN_BOOTSTRAP


def replay_geometry(trace, cost_fit=None, samples=1000, bootstrap_samples=200, seed=1234,
                    replicate_h=0, details=None, extrapolate_layers=None, train_trace=None,
                    rows_per_request=None, layouts=None):
    from moereplay.replay import (proxy_widths, train_frequency, prefill_events,
                                 decode_events, union_growth, PLACEMENTS)
    geometry = trace.geometry
    selected = resolve_layouts(geometry, layouts)
    baseline = selected[0]
    widths = proxy_widths(rows_per_request)
    if type(samples) is not int or samples < 1 or type(bootstrap_samples) is not int or bootstrap_samples != 0 and bootstrap_samples < MIN_BOOTSTRAP:
        raise ValueError('samples >=1; bootstrap 0 or >=200 required')
    if type(replicate_h) is not int or not 0 <= replicate_h <= geometry.num_experts:
        raise ValueError('replicate_h must be in [0,num_experts]')
    trace.check()
    if train_trace is not None:
        train_trace.check()
        if train_trace.geometry != geometry:
            raise ValueError('training trace geometry differs')
        if train_trace.manifest['model'] != trace.manifest['model'] or train_trace.manifest['layers'] != trace.manifest['layers']:
            raise ValueError('training trace model/layers differ')
        if train_trace.root == trace.root or train_trace.manifest['data'] == trace.manifest['data']:
            raise ValueError('training trace must differ from test data')
        test_ids = {r['request'] for r in trace.rows() if not r['padding']}
        train_ids = {r['request'] for r in train_trace.rows() if not r['padding']}
        if test_ids & train_ids:
            raise ValueError('external training requires disjoint request IDs')
    costs = Costs(cost_fit, geometry, selected)
    steps = [rows[0]['step'] for rows in trace.steps()]
    proxy = trace.manifest['step_semantics'] == 'prefill-proxy'
    if proxy:
        prompts = sorted({r['request'] for r in trace.rows()})
        if len(prompts) < 2 and train_trace is None:
            raise ValueError('prefill proxy requires >=2 prompts for holdout')
        split = max(1, len(prompts)//2)
        train_requests = prompts[:split] if train_trace is None else None
        test_requests = prompts[split:] if train_trace is None else prompts
        train_steps = set()
        holdout = dict(status='assumed', unit='prompt', train=train_requests or 'external', test=test_requests,
                       c4_limitation='replacement sampling when fewer than four test prompts')
    else:
        if len(steps) < 2 and train_trace is None:
            raise ValueError('decode replay needs >=2 steps or external train trace')
        train_steps = set(steps[:max(1, len(steps)//2)]) if train_trace is None else set()
        train_requests = None
        holdout = dict(status='assumed', unit='chronological steps', training_steps=len(train_steps),
                       test_steps=len(steps)-len(train_steps), prompt_overlap='possible; use external train trace for prompt holdout')
    layers = trace.manifest['layers']
    if extrapolate_layers is not None and (type(extrapolate_layers) is not int or extrapolate_layers < len(layers)):
        raise ValueError('extrapolation count must be an integer >=captured layer count')
    scale = extrapolate_layers/len(layers) if extrapolate_layers else 1
    totals, clusters, banks, placement_fits, per_layer, growth, fit_records = {}, {}, {}, {}, {}, {}, {}
    names = list(PLACEMENTS) + (['replicate-hottest'] if replicate_h else [])
    stream = open(details, 'x') if details else None

    def average(values):
        return statistics.fmean(values) if all(v is not None for v in values) else None

    def distribution(values, groups):
        if any(v is None for v in values):
            return dict(mean=None, steps_ci95=None, prompts_ci95=None, status='unavailable', steps=len(values))
        return bootstrap(values, groups, bootstrap_samples, seed)

    try:
        for layer in layers:
            training = train_trace or trace
            frequency = train_frequency(training, layer, None if train_trace else train_steps if not proxy else None,
                                        None if train_trace else train_requests)
            hottest = sorted(range(geometry.num_experts), key=lambda e: (-frequency[e], e))[:replicate_h]
            owners = {(layout.name, name):placement(layout.ep, name, frequency, geometry)
                      for layout in selected for name in PLACEMENTS}
            placement_fits[str(layer)] = dict(status=training.manifest['source']['status'],
                training_rows=sum(frequency.values())//geometry.top_k,
                frequencies=[frequency[e] for e in range(geometry.num_experts)], hottest=hottest,
                owners={'/'.join(key):value for key, value in owners.items()})
            layer_banks = {}
            for layout in selected:
                for name in names:
                    owner = owners[layout.name, 'greedy' if name == 'replicate-hottest' else name]
                    counts = Counter(owner)
                    hot_counts = Counter(owner[e] for e in hottest) if name == 'replicate-hottest' else Counter()
                    bank = [(counts[g]-hot_counts[g]+(replicate_h if name == 'replicate-hottest' else 0))*geometry.expert_bytes/layout.tp
                            if geometry.expert_bytes is not None else None
                            for g in range(layout.ep) for _ in range(layout.tp)]
                    layer_banks[layout.name, name] = bank
                    total = banks.setdefault((layout.name, name), [0. if geometry.expert_bytes is not None else None]*geometry.world_size)
                    for rank, value in enumerate(bank):
                        if value is not None:
                            total[rank] += value*scale
            events = prefill_events(trace, layer, test_requests, samples, seed, widths) if proxy else decode_events(trace, layer, train_steps)
            layer_values = {}
            rng = random.Random(seed+layer)
            for event in events:
                m, c = event['M'], event['c']
                uniform = [tuple(rng.sample(range(geometry.num_experts), geometry.top_k)) for _ in range(m)]
                for control, routes in (('observed', event['routes']), ('uniform-random', uniform)):
                    baseline_work = rank_work(routes, baseline, owners[baseline.name, 'contiguous'], geometry=geometry)
                    baseline_time = costs.evaluate(baseline.name, m, baseline_work['D_r'], baseline_work)
                    for layout in selected:
                        for name in names:
                            work = rank_work(routes, layout, owners[layout.name, 'greedy' if name == 'replicate-hottest' else name],
                                             hottest if name == 'replicate-hottest' else (), geometry)
                            timing = costs.evaluate(layout.name, m, work['D_r'], work)
                            ms = timing['layer_us']/1000 if timing['layer_us'] is not None else None
                            delta = ms-baseline_time['layer_us']/1000 if ms is not None and baseline_time['layer_us'] is not None else None
                            eligible = timing['calibration_eligible'] and baseline_time['calibration_eligible']
                            cell = 'c%d_M%d_K%d' % (c, m, event['k'])
                            key = (control, cell, layout.name, name)
                            record = totals.setdefault(key, {}).setdefault(event['step'], dict(ms=0., delta=0., eligible=True, reasons=set()))
                            record['ms'] = record['ms']+ms*scale if record['ms'] is not None and ms is not None else None
                            record['delta'] = record['delta']+delta*scale if record['delta'] is not None and delta is not None else None
                            record['eligible'] &= eligible
                            record['reasons'].update(timing['calibration_reasons'])
                            record['reasons'].update('baseline: '+r for r in baseline_time['calibration_reasons'])
                            clusters[cell, event['step']] = event['requests']
                            layer_values.setdefault(key, []).append(dict(work=work, ms=ms, eligible=eligible))
                            fit_records['%s_M%d' % (layout.name, m)] = timing['fit']
                            if stream:
                                stream.write(json.dumps(dict(layer=layer, control=control, cell=cell, step=event['step'], layout=layout.name,
                                    placement=name, route_status=event['status'] if control == 'observed' else 'assumed',
                                    geometry_fingerprint=geometry.fingerprint, work=work, time=timing,
                                    baseline_layout=baseline.name, delta_vs_baseline_ms=delta), separators=(',', ':'), allow_nan=False)+'\n')
            per_layer[str(layer)] = {'/'.join(key):dict(
                D_union=average([v['work']['D_union'] for v in values]),
                max_rank_D=average([v['work']['max_rank_D'] for v in values]),
                D_r=[average([v['work']['D_r'][rank] for v in values]) for rank in range(geometry.world_size)],
                bytes_r=[average([v['work']['bytes_r'][rank] for v in values]) for rank in range(geometry.world_size)],
                routed_slots_r=[average([v['work']['routed_slots_r'][rank] for v in values]) for rank in range(geometry.world_size)],
                bytes_status='estimated' if geometry.expert_bytes is not None else 'unavailable',
                bank_bytes_r=layer_banks[key[2], key[3]], layer_ms=average([v['ms'] for v in values]),
                time_status='estimated' if all(v['ms'] is not None for v in values) else 'unavailable',
                calibration_eligible_fraction=average([v['eligible'] for v in values])) for key, values in layer_values.items()}
            growth[str(layer)] = union_growth(trace, layer)
        cycles = {}
        for key, events in totals.items():
            ids = sorted(events)
            groups = [clusters[key[1], i] for i in ids]
            reasons = set().union(*(events[i]['reasons'] for i in ids))
            if scale != 1:
                reasons.add('layer extrapolation')
            if proxy:
                reasons.add('prefill proxy is not decode evidence')
            if key[0] != 'observed':
                reasons.add('uniform random control is a sensitivity')
            if trace.manifest['source'].get('route_stage') != 'post-kstop-remap':
                reasons.add('post-kstop-remap route provenance unknown')
            if trace.manifest['source']['status'] != 'measured':
                reasons.add('route source is assumed')
            times = distribution([events[i]['ms'] for i in ids], groups)
            deltas = distribution([events[i]['delta'] for i in ids], groups)
            deltas.update(calibration_eligible=all(events[i]['eligible'] for i in ids) and not reasons,
                          calibration_reasons=sorted(reasons))
            cycles['/'.join(key)] = dict(cycle_kernel_ms=times, delta_vs_baseline_ms=deltas,
                bank_bytes_r=banks[key[2], key[3]], bank_bytes_status='estimated' if geometry.expert_bytes is not None else 'unavailable',
                serving_compatible=None, supported_in_pinned_vllm=None,
                serving_limitation='unknown: no runtime pin or compatibility check', placement_status='assumed')
        return dict(schema='moe-replay v2', status='estimated', geometry=geometry.to_dict(), geometry_fingerprint=geometry.fingerprint,
            layouts=[l.to_dict() for l in selected], baseline_layout=baseline.name, source=trace.manifest['source'],
            route_semantics=trace.manifest['step_semantics'], holdout=holdout, layers=layers,
            layer_scale=dict(status='assumed', factor=scale, modeled_layers=extrapolate_layers or len(layers)),
            proxy_rows_per_request={str(c):w for c, w in widths.items()} if proxy else None,
            placement_fits=placement_fits, cost_fits=fit_records, cost_fit_adapter=getattr(costs, 'adapter', 'unknown'),
            cycles=cycles, per_layer=per_layer, union_growth=growth, cyclesim_what_if=[],
            collective=dict(status='unavailable', delta_ms=None, source='geometry does not establish collective behavior'),
            limits=['No timing prior is inferred from geometry; geometry-bound layout/M fits required.',
                    'Geometry presets do not establish model or runtime support.',
                    'Weight bytes are unavailable unless explicitly declared; not measured DRAM traffic.',
                    'Generic geometry does not establish cyclesim runtime compatibility.'])
    finally:
        if stream:
            stream.close()
