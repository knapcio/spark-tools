#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fit explicitly named evidence subsets and report disjoint holdout errors."""
import argparse
import collections
import json
import statistics
from pathlib import Path
from cyclesim.utilities import dump, sha256
from cyclesim.model import Model


def _report_float(value):
    """Keep human-facing calibration reports stable across Python builds."""
    return round(float(value), 12)


def split_prompt_roles(observations):
    """Global hash membership, including all fitting instruments and arms.
    Unknown hashes cannot be claimed to be a prompt-disjoint holdout.
    """
    fit_hashes = {h for x in observations if x.get('role') in ('fit', 'transfer-fit')
                  for h in x.get('prompt_hashes', [])}
    for x in observations:
        if x.get('role') in ('fit', 'transfer-fit'):
            continue
        hashes = x.get('prompt_hashes', [])
        if not hashes:
            x['role'] = 'unknown-prompt-transfer'
        elif fit_hashes.intersection(hashes):
            x['role'] = 'repeated-prompt-transfer'
        else:
            x['role'] = 'holdout'
    return sorted(fit_hashes)


def fit(observations, profile):
    cur, arms = observations['current'], observations['arms']
    domains = {}
    # Historical fixed K2 graph cost is only a structural prior, not a
    # DH-derived pass estimate. Current c1 pass distributions remain unknown.
    pass_ms = profile['category_ms']['draft_graph'] / 2
    empirical = {}
    for domain in ('prose', 'code', 'structured'):
        off = arms[domain+'/OFF/fit']
        empirical[domain] = {}
        for arm in ('DH', 'K4', 'BOTH'):
            x = arms[domain+'/'+arm+'/fit']
            if x['prompt_hashes'] != off['prompt_hashes']:
                raise ValueError('empirical arms need matched prompt hashes')
            empirical[domain][arm] = {'cycle_delta_ms': x['cycle_ms']-off['cycle_ms'],
                'survival_delta':[x['survival'][i]-off['survival'][i] for i in range(4)],
                'status':'measured paired prompt arm; transfer to thinking/default sampling estimated',
                'fit_prompt_hashes':x['prompt_hashes'], 'instrument':x['instrument']}
    for domain in ('prose', 'code'):
        xs = [x for x in cur if x['domain'] == domain and x['role'] == 'fit']
        off, dh = arms[domain+'/OFF/fit'], arms[domain+'/DH/fit']
        ka = arms[domain+'/K4/fit']
        domains[domain] = {'cycle_fit_ms': statistics.median(x['cycle_ms'] for x in xs),
                           'survival': [statistics.fmean(x['survival'][i] for x in xs) for i in range(3)],
                           'draft_passes': 2 if domain=='prose' else 3,
                           'k4_fires_per_cycle': ka['fires_per_cycle'],
                           'k4_survival_delta': [ka['survival'][i] - off['survival'][i] for i in range(4)],
                           'domain_union_extra_ms': 0, 'status': 'estimated from measured anchors',
                           'acceptance_status': 'measured fit-counter prefix survival',
                           'pass_fit_status': 'assumed structural prior only; current executed-pass distribution unknown'}
    structured = arms['structured/OFF/fit']
    sk = arms['structured/K4/fit']
    for domain in ('structured', 'json'):
        domains[domain] = dict(domains['code'])
        domains[domain].update({'survival': structured['survival'][:3], 'draft_passes': 3,
            'k4_fires_per_cycle': sk['fires_per_cycle'],
            'k4_survival_delta': [sk['survival'][i] - structured['survival'][i] for i in range(4)],
            'acceptance_status': 'estimated transfer of T=0 structured counter survival' if domain == 'structured' else
                                 'assumed structured survival; JSON counter capture missing',
            'cycle_fit_ms': domains['code']['cycle_fit_ms']})
    cats = dict(profile['category_ms'])
    cats.pop('draft_graph', None)
    # The measured current transformation is -9% cycle. Apply that budget to
    # formerly INT8 categories proportionally; FP4x has equal-cycle observations.
    int8_old = cats.pop('attention_gemms')
    cycle_ratio = observations['transforms']['nvfp4_cycle_ratio']
    if cycle_ratio is None:
        raise ValueError('NVFP4 paired cycle ratio missing')
    delta = (1 - cycle_ratio) * profile['ranks'][0]['cycle_ms']['median']
    ratio = max(0, (int8_old - delta) / int8_old)
    cats['attention_gemms'] = 18.973 * ratio
    cats['shared_dense'] += (int8_old - 18.973) * ratio
    total = sum(cats.values()) + domains['prose']['draft_passes'] * pass_ms
    closure = domains['prose']['cycle_fit_ms'] - total
    domains['code']['domain_union_extra_ms'] = (domains['code']['cycle_fit_ms'] - domains['prose']['cycle_fit_ms'] -
                                 (domains['code']['draft_passes'] - domains['prose']['draft_passes']) * pass_ms)
    for domain in ('structured', 'json'):
        domains[domain]['domain_union_extra_ms'] = domains['code']['domain_union_extra_ms']
        domains[domain]['cycle_fit_ms'] = sum(cats.values()) + closure + domains[domain]['domain_union_extra_ms'] + domains[domain]['draft_passes']*pass_ms
    external = {}
    for domain in ('prose', 'code'):
        rates = [x['tps'] for x in observations['observations'] if x['id'].startswith('dash/') and x['domain'] == domain and x['c'] == 1]
        baseline = (1+sum(domains[domain]['survival'])) / domains[domain]['cycle_fit_ms'] * 1000
        # First two of five Dash runs fit; the remaining three are holdout.
        external[domain] = statistics.median(rates[:2]) / baseline
    external['structured'] = external['code']
    external['json'] = external['code']
    rigmark = {}
    for domain, evidence in observations.get('rigmark_fit',{}).items():
        baseline = (1+sum(domains[domain]['survival'])) / domains[domain]['cycle_fit_ms'] * 1000
        rigmark[domain] = {'factor':statistics.median(evidence['fit_rates'])/baseline,
                           'fit_runs':[1,2,3,4], 'held_out_run':5, 'settings':evidence['settings'],
                           'status':evidence['status']}
    calib = {'schema': 5, 'stack': 'calibrated-nvfp4-k3', 'baseline_configuration': 'example',
             'domains': domains, 'category_prior_ms': cats,
             'category_status': {name: 'estimated rescaled historical profile; current per-kernel capture missing' for name in cats},
             'category_bytes': {'moe_routed': 75*3*6144*512*(.5+2/128)*17.6,
                                'vocab_head': 475791360, 'attention_gemms': 77*(2048*6144+576*6144+4096*2048+6144*4096)*9/16},
             'expert_union_c1': 17.6, 'expert_union_status': 'assumed historical prior',
             'draft_pass_ms': pass_ms, 'draft_pass_c_increment_ms': .65, 'draft_cost_status': 'historical fixed K2 expected graph cost divided by two; current price unknown',
             'empirical_arms': empirical, 'execution':observations['execution'],
             'fit_prompt_hashes':observations['fit_prompt_hashes'], 'release_policy_sha256':observations['release_policy_sha256'], 'closure_ms': closure, 'padded_row_ms': 7.5,
             'dram_ceiling_gbs': 270, 'collective_scale': {'1': 1, '2': 1.8, '4': 2.28},
             'external_factors': external, 'rigmark_factors':rigmark, 'external_c_factors': {'1': 1, '2': 1, '4': 1},
             'errors': {'c1': {'max_abs_pct': 0}, 'c2_c4': {'max_abs_pct': 0}},
             'source_sha256': observations['source_sha256'], 'capture_requests': observations['capture_requests'] +
                 ['current resolved per-kernel traces', 'JSON and c2/c4 per-position acceptance counters',
                  'confidence histogram for tau != .90', 'topk_ids routing capture for expert union and bytes'],
             'fit_protocol': 'Global prompt-hash split: current+screen 0..4 fit; all fit hashes excluded from true holdout across boots/arms. First two c1 Dash runs fit instrument transfer (prompt hashes unknown). Later Dash/RigMark are unknown-prompt transfer; repeated qualification hashes are repeated-prompt transfer. No concurrency fit. RigMark independently fits runs1..4, holds out run5; individual hashes unknown, instrument-repeat validation only, not prompt generalization. The original sparkDash-to-RigMark miss remains a separate unfitted transfer test.',
             'transforms': [{'name': 'NVFP4 attention/shared/dense/MTP', 'cycle_ratio': cycle_ratio,
                              'status': 'measured aggregate prior; all category attribution estimated'},
                            {'name': 'FP4x KV', 'cycle_ratio': 1.0, 'acceptance_ratio_range': [.95, .97],
                             'status': 'measured aggregate prior; current fit survival already includes effect'}]}
    return calib


def validate(calib, observations):
    model, rows = Model(calib), []
    groups = collections.defaultdict(list)
    def identified(hashes):
        return (isinstance(hashes, list) and bool(hashes) and
                all(isinstance(h, str) and bool(h.strip()) for h in hashes))

    fit_hashes = calib.get('fit_prompt_hashes')
    for x in observations['observations']:
        role = x.get('role')
        if role in ('fit', 'transfer-fit'):
            continue
        if role == 'holdout':
            hashes = x.get('prompt_hashes')
            if not identified(hashes) or not identified(fit_hashes):
                role = 'unknown-prompt-transfer'
            elif set(hashes).intersection(fit_hashes):
                raise ValueError('holdout prompt leakage')
        arm = x.get('arm', 'OFF')
        pred = model.predict(x['c'], x['domain'], k=4 if arm in ('K4', 'BOTH') else 3,
                             head='nvfp4' if arm in ('DH', 'BOTH') else 'bf16')
        anchor = x['id']
        if anchor.startswith('dash/'):
            anchor = 'sparkDash/%s/c%d' % (x['domain'], x['c'])
        elif anchor.startswith('current/'):
            anchor = 'current/%s/c1' % x['domain']
        instrument = x.get('instrument', 'unknown')
        metrics = {}
        if x.get('cycle_ms'):
            metrics['cycle_ms'] = (x['cycle_ms'], pred['cycle_bench_hard_panel']['cycle_ms'])
        if 'committed' in x:
            metrics['committed_per_cycle'] = (x['committed'], pred['cycle_bench_hard_panel']['committed_per_cycle'])
        if x.get('tps'):
            column = x.get('prediction_column') or ('sparkdash_calibrated' if instrument.startswith(('sparkDash', 'RigMark')) else 'cycle_bench_hard_panel')
            metrics['tok_s_per_stream'] = (x['tps'], pred[column]['tok_s_per_stream'])
        for metric, values in metrics.items():
            groups[(anchor, metric, x['c'], role, instrument)].append(values)
    for (anchor, metric, c, role, instrument), pairs in groups.items():
        m, p = statistics.median(x[0] for x in pairs), statistics.median(x[1] for x in pairs)
        rows.append({'anchor':anchor, 'metric':metric, 'c':c, 'n':len(pairs),
                     'measured':_report_float(m), 'predicted':_report_float(p),
                     'error_pct':_report_float(100*(p/m-1)),
                     'measurement_status':'measured', 'prediction_status':'estimated',
                     'role':role, 'instrument':instrument})
    def summary(xs):
        errors = [abs(x['error_pct']) for x in xs]
        return {'median_abs_pct':_report_float(statistics.median(errors)) if errors else None,
                'max_abs_pct':_report_float(max(errors)) if errors else None, 'n_anchors':len(errors),
                'status':'measured error of estimated predictions; cohort/instrument-specific'}
    calib['metric_errors'] = []
    for cohort in sorted({x['role'] for x in rows}):
        for concurrency, cs in (('c1', (1,)), ('c2_c4', (2,4))):
            for metric in ('cycle_ms', 'committed_per_cycle', 'tok_s_per_stream'):
                for instrument in sorted({x['instrument'] for x in rows if x['role']==cohort and x['metric']==metric and x['c'] in cs}):
                    xs = [x for x in rows if x['role']==cohort and x['metric']==metric and x['c'] in cs and x['instrument']==instrument]
                    calib['metric_errors'].append(dict(summary(xs), cohort=cohort, concurrency=concurrency,
                                                       metric=metric, instrument=instrument))
    for name, cs in (('c1',(1,)),('c2_c4',(2,4))):
        # Bands use throughput transfer error only. Never mix cycle and
        # acceptance percentages into a throughput calibration headline.
        calib['errors'][name] = summary([x for x in rows if x['c'] in cs and x['metric']=='tok_s_per_stream'])
    return rows


def render(calib, rows):
    lines = ['# Calibration v5 - fix round 1', '',
             'Structural holdout is globally prompt-hash-disjoint across all prompt-based fits. Unknown-hash instrument factors do not enter structural holdout metrics; instrument-repeat and repeated/unknown-prompt transfers are separate cohorts.', '',
             '| cohort | concurrency | metric | instrument | median abs % | max abs % | anchors |',
             '|---|---|---|---|---:|---:|---:|']
    for v in calib['metric_errors']:
        lines.append('| %s | %s | %s | %s | %.2f | %.2f | %d |' %
            (v['cohort'],v['concurrency'],v['metric'],v['instrument'],v['median_abs_pct'],v['max_abs_pct'],v['n_anchors']))
    lines += ['', '| anchor | cohort | metric | instrument | measured | predicted (estimated) | error % | n |',
              '|---|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| %s | %s | %s | %s | %.4f | %.4f | %+.2f | %d |' %
            (row['anchor'],row['role'],row['metric'],row['instrument'],row['measured'],row['predicted'],row['error_pct'],row['n']))
    lines += ['', 'Fit protocol: '+calib['fit_protocol'], '',
              'c2/c4: uniform K2, reuse, pad hygiene, steady M6/M12; K2 acceptance is a truncated c1 prefix transfer, not measured concurrent survival.',
              'Executed c1 passes and cost remain assumed historical priors. Reduced-K/new-policy structural scenarios are refused.',
              'DH/K4/BOTH use separate paired empirical deltas; no head-roofline or padded-row pass-count fit.',
              'The original sparkDash to RigMark test remains unfitted (prose miss retained). A separate RigMark panel factor fits four repeats and tests run5; it is T=0 low-effort, not identified thinking-on prose. Individual repeat prompt hashes are unknown, so this is instrument-repeat transfer, not prompt generalization.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observations', type=Path, required=True)
    p.add_argument('--profile-prior', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--report-dir', type=Path, required=True)
    a = p.parse_args()
    observations = json.loads(a.observations.read_text())
    calib = fit(observations, json.loads(a.profile_prior.read_text()))
    calib['input_sha256'] = {'observations': sha256(a.observations), 'profile_prior': sha256(a.profile_prior)}
    rows = validate(calib, observations)
    a.report_dir.mkdir(parents=True, exist_ok=True)
    dump(a.out, calib)
    dump(a.report_dir/'calibration-validation.json', {'errors': calib['errors'], 'metric_errors':calib['metric_errors'], 'rows': rows})
    (a.report_dir/'CALIBRATION.md').write_text(render(calib, rows))
    print(json.dumps(calib['errors']))


if __name__ == '__main__':
    main()
