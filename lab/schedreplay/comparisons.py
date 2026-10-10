# SPDX-License-Identifier: Apache-2.0
"""Metric-scoped comparisons. Missing observations never become zero."""
import math


METRICS = {
    'ttft_p90_s': (lambda r:r['ttft_s']['p90'],1),
    'max_decode_stall_s': (lambda r:r['max_decode_stall_s'],1),
    'queue_wait_p90_s': (lambda r:r['queue_wait_s']['p90'],1),
    'failures': (lambda r:r['total_requests']-r['completed'],1),
    'decode_during_prefills_tok_s': (lambda r:r['decode_during_prefills_tok_s'],-1),
    'prefill_throughput_tok_s': (lambda r:r['prefill_throughput_tok_s'],-1),
    'B_TTFT_s': (lambda r:r['scoped_metrics']['B_TTFT_s'],1),
    'A_event_gap_stall_s': (lambda r:r['scoped_metrics']['A_event_gap_stall_s'],1),
}


def projection(result, names):
    values, reasons = [],[]
    if result['max_num_seqs'] == 6:
        reasons.append('c6 allocator/graph/admission qualification missing')
    for name in names:
        getter, direction = METRICS[name]
        value = getter(result)
        if value is None or type(value) not in (int,float) or not math.isfinite(value):
            reasons.append('missing objective: '+name)
            values.append(None)
        else: values.append(direction*value)
        support = result.get('metric_support',{}).get(name)
        if support is not None:
            reasons.extend(name+': '+reason for reason in support)
        elif not result.get('timing_supported',False):
            reasons.extend(name+': '+reason for reason in
                           result.get('unsupported_timing_regimes',[]) or ['timing support unknown'])
    return values,sorted(set(reasons))


def dominates(a,b):
    tolerances = [1e-9*max(1,abs(x),abs(y)) for x,y in zip(a,b)]
    return (all(x <= y+eps for x,y,eps in zip(a,b,tolerances)) and
            any(x < y-eps for x,y,eps in zip(a,b,tolerances)))


def compare(results, baseline, objectives, scope):
    """Frontier only over supported members of the explicitly named scope."""
    base, base_reasons = projection(baseline,objectives)
    projections = [projection(r,objectives) for r in results]
    supported = [v for v,reasons in projections if not reasons]
    table = []
    for result,(values,reasons) in zip(results,projections):
        blockers = sorted(set(reasons+base_reasons))
        table.append({'policy':result['policy'],'admission':result['admission'],
                      'max_num_seqs':result['max_num_seqs'],'objectives':list(objectives),
                      'values':{name:METRICS[name][0](result) for name in objectives},
                      'vs_current':('indeterminate' if blockers else
                                    'dominates' if dominates(values,base) else
                                    'dominated' if dominates(base,values) else
                                    'equal' if all(math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-9)
                                                   for x,y in zip(values,base)) else 'tradeoff'),
                      'pareto':None if reasons else not any(dominates(other,values) for other in supported),
                      'ranking_withheld_reasons':blockers,'scope':scope,
                      'frontier_supported_members':len(supported),
                      'frontier_excluded_members':len(results)-len(supported),
                      'evidence':'estimated conditional comparison; no hardware qualification'})
    return table
