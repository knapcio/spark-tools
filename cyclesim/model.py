#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode-cycle cost model: calibrated anchors, prefix survival and bounded what-ifs."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from cyclesim.utilities import validate_survival

HERE = Path(__file__).resolve().parent
DOMAINS = ('prose', 'code', 'structured', 'json')


def document_sha256(value):
    """Hash parsed JSON canonically, independent of whitespace and key order."""
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return 'sha256:' + hashlib.sha256(encoded).hexdigest()


def finite(x, low=-math.inf, high=math.inf):
    x = float(x)
    if not math.isfinite(x) or not low <= x <= high:
        raise ValueError('finite value outside allowed range')
    return x


def union_experts(rows, base=17.6, experts=256):
    """Occupancy union with correlated per-sequence rows folded into c1 prior.
    Inspired by v1-v4 workspace simulators; this is an assumed routing model.
    """
    return experts * (1 - (1 - base / experts) ** (rows / 4))


def edits(values):
    out = {}
    for value in values or []:
        # A C++ kernel name contains commas. Repeat the flag for such names;
        # comma lists are only decoded when they contain multiple assignments.
        for part in ([value] if value.count('=') == 1 else value.split(',')):
            name, sep, raw = part.rpartition('=')
            if not sep or not name:
                raise ValueError('edits require NAME=VALUE')
            out[name] = out.get(name, 0) + finite(raw)
    return out


class Model:
    def __init__(self, calibration=None, kernel_table=None):
        if calibration is None: raise ValueError('explicit calibration configuration required')
        self.data = calibration
        if self.data.get('schema') != 5:
            raise ValueError('calibration schema 5 required')
        label = self.data.get('calibration_label', self.data.get('stack', 'schema-5'))
        if (not isinstance(label, str) or not label.strip() or len(label) > 160 or
                any(ord(char) < 32 or ord(char) == 127 for char in label)):
            raise ValueError('calibration_label must be a nonempty printable string of at most 160 characters')
        self.calibration_label = label.strip()
        self.calibration_sha256 = document_sha256(self.data)
        self.table = json.loads(Path(kernel_table).read_text()) if kernel_table else None
        self.kernel_table_sha256 = document_sha256(self.table) if self.table else None
        if self.table and (self.table.get('schema') != 1 or not self.table.get('kernels')):
            raise ValueError('kernel table must contain an ingested profile')
        if self.table:
            t = self.table
            if (t.get('domain') not in DOMAINS or t.get('head') not in ('bf16', 'nvfp4') or
                    t.get('policy') not in ('kstop', 'uniform_k2') or
                    t.get('k') not in (2, 3) or not t.get('executed_passes') or
                    not t.get('unprofiled_cycle_ms')):
                raise ValueError('table requires capture domain/K/policy/head/executed passes and bracket anchor')
            finite(t['executed_passes'], 1, t['k'])
            if t['c']==1 and (t['k'] != 3 or t['policy'] != 'kstop'):
                raise ValueError('unidentified reduced-K capture policy')
            if t['c'] > 1 and (t['head'] != 'bf16' or t['k'] != 2 or t['policy'] != 'uniform_k2'):
                raise ValueError('unidentified concurrent profile head/policy')
            if any('mean' not in r.get('total_us_per_cycle', {}) for r in t['kernels']):
                raise ValueError('table requires mean expected-work costs; re-ingest legacy table')
            for value in self.table['category_ms'].values():
                finite(value, 0)
            if self.table.get('unprofiled_cycle_ms') is not None:
                finite(self.table['unprofiled_cycle_ms'], 1e-9)

    def provenance(self):
        """Portable prediction identity without leaking a local input path."""
        return {'calibration_label': self.calibration_label,
                'calibration_schema': self.data['schema'],
                'calibration_sha256': self.calibration_sha256,
                'kernel_table_sha256': self.kernel_table_sha256}

    def baseline_parts(self, c, domain):
        if c not in (1, 2, 4) or domain not in DOMAINS:
            raise ValueError('supported c=1,2,4 and prose/code/structured/json')
        cats, pass_ms, rows, union = self._structural_parts(c, domain)
        closure = self.data['closure_ms']
        note = None
        if self.table:
            t = self.table
            reference, capture_domain = t['c'], t['domain']
            ref, _, _, _ = self._structural_parts(reference, capture_domain, t['k'], t['executed_passes'], t['head'])
            cats = {cat: value * cats.get(cat, 0) / max(1e-9, ref.get(cat, value))
                    for cat, value in t['category_ms'].items()}
            # Anchor the capture itself, then transfer category work. Never
            # normalize a prose capture against the requested code domain.
            closure = t['unprofiled_cycle_ms'] - sum(t['category_ms'].values())
            passes = self._passes(c, domain)
            pass_ms = cats.get('draft_graph', 0) / passes
            note = {'status': 'estimated transfer from measured capture', 'reference_c': reference,
                    'capture_domain': capture_domain, 'capture_k': t['k'], 'capture_head': t['head'],
                    'capture_policy': t['policy'], 'capture_executed_passes': t['executed_passes'],
                    'closure_reference_ms': t['unprofiled_cycle_ms'], 'closure_reference_c': reference,
                    'stack': t.get('stack', 'unknown')}
        return cats, closure, pass_ms, rows, union, note

    def _passes(self, c, domain):
        if self.table and c==self.table['c'] and domain==self.table['domain']:
            return self.table['executed_passes']
        return 2. if c > 1 else self.data['domains'][domain]['draft_passes']

    def _structural_parts(self, c, domain, k=None, passes=None, head='bf16'):
        data, d = self.data, self.data['domains'][domain]
        k = k if k is not None else (3 if c == 1 else 2)
        rows = self.data['execution'][str(c)]['verify_rows'] if k == self.data['execution'][str(c)]['k'] else c * (k+1)
        union = union_experts(rows, data['expert_union_c1'])
        cats = dict(data['category_prior_ms'])
        cats['moe_routed'] *= union / data['expert_union_c1']
        cats['moe_routed'] += d['domain_union_extra_ms'] * c ** .5
        cats['attention_gemms'] *= 1 + .01 * (rows - 4)
        cats['shared_dense'] *= 1 + .003 * (rows - 4)
        cats['collectives'] *= data['collective_scale'][str(c)]
        cats['mla'] *= 1 + .10 * (c - 1)
        cats['indexer'] *= c
        cats['router'] *= 1 + .1 * (c - 1)
        cats['host_sampling'] *= 1 + .05 * (c - 1)
        pass_ms = data['draft_pass_ms'] + data['draft_pass_c_increment_ms'] * (c - 1)
        passes = self._passes(c, domain) if passes is None else passes
        cats['draft_graph'] = passes * pass_ms
        if head == 'nvfp4':
            # Only an empirical paired cost delta is available. Attribution to
            # draft_graph for normalization is estimated, not a pass-price fit.
            delta = self._arm(domain, 'DH')['cycle_delta_ms']
            cats['draft_graph'] += delta
            if cats['draft_graph'] <= 0:
                raise ValueError('DH capture normalization exceeds draft budget')
        return cats, pass_ms, rows, union

    def _prior_parts(self, c, domain):
        return self._structural_parts(c, domain)[0]

    def _arm(self, domain, arm):
        if domain == 'json' or arm not in self.data.get('empirical_arms', {}).get(domain, {}):
            raise ValueError('unidentified empirical arm; capture required')
        return self.data['empirical_arms'][domain][arm]

    def predict(self, c=1, domain='prose', k=3, head='bf16', tau=.90, policy='kstop',
                kernel_ms=None, kernel_gbs=None, byte_edits=None, accept_pos=None,
                accept_scale=None, draft_pass_delta=0):
        """Stable schedreplay API. Default K3 request resolves to uniform K2 at c>1.
        Reduced c1 K and new confidence policies require identified observations.
        """
        if c not in (1, 2, 4) or domain not in DOMAINS:
            raise ValueError('supported c=1,2,4 and prose/code/structured/json')
        if head not in ('bf16', 'nvfp4') or finite(tau, 0, 1) != .90:
            raise ValueError('only measured heads and tau=.90 are supported')
        requested_k = k
        if c > 1:
            if k not in (2, 3) or policy not in ('kstop', 'fixed', 'uniform_k2') or head != 'bf16':
                raise ValueError('only calibrated uniform K2 BF16 identified at c2/c4')
            k, policy = 2, 'uniform_k2'
        elif k not in (3, 4) or policy != 'kstop':
            raise ValueError('unidentified reduced-K widths/pass distribution or fixed policy; capture required')
        data, d = self.data, self.data['domains'][domain]
        cats, closure, pass_ms, rows, union, table_note = self.baseline_parts(c, domain)
        base_cats = dict(cats)
        survival = list(d['survival'][:k])
        baseline_survival = list(d['survival'][:(3 if c == 1 else 2)])
        passes = self._passes(c, domain)
        matching_capture_survival = None
        if self.table and c==self.table['c'] and domain==self.table['domain'] and self.table.get('acceptance'):
            baseline_survival = list(self.table['acceptance']['survival'])
            if self.table['head']=='nvfp4':
                if head=='nvfp4':
                    matching_capture_survival = list(baseline_survival)
                else:
                    baseline_survival = [p-v for p,v in zip(baseline_survival,self._arm(domain,'DH')['survival_delta'])]
            validate_survival(baseline_survival)
            survival = list(baseline_survival)
        scenario, basis = [], 'structural_prior'
        empirical_delta = 0.
        if self.table and c == self.table['c'] and domain == self.table['domain']:
            passes = self.table['executed_passes']
            # baseline_parts normalized to DH-OFF; reverse exactly once when
            # the requested head matches a capture already using DH.
        if k == 4 or head == 'nvfp4':
            arm_name = 'BOTH' if k == 4 and head == 'nvfp4' else ('K4' if k == 4 else 'DH')
            arm = self._arm(domain, arm_name)
            if self.table and k == 4:
                raise ValueError('K4 empirical arm cannot be attributed to a replacement kernel table')
            if self.table and self.table['head'] == 'nvfp4' and head == 'nvfp4':
                ref, _, _, _ = self._structural_parts(self.table['c'], self.table['domain'],
                    self.table['k'], self.table['executed_passes'], 'nvfp4')
                desired, _, _, _ = self._structural_parts(c, domain, head='nvfp4')
                for cat, value in self.table['category_ms'].items():
                    cats[cat] = value * desired.get(cat, 0) / max(1e-9, ref.get(cat, value))
            else:
                # Paired cycle effects are separate empirical terms. Do not
                # assign K4's confounded extra work to routed MoE or infer a
                # draft-pass price from DH.
                empirical_delta = arm['cycle_delta_ms']
            survival = [p + v for p, v in zip(survival + ([0.] if k == 4 else []), arm['survival_delta'][:k])]
            if matching_capture_survival is not None:
                survival = matching_capture_survival
            passes, pass_ms, basis = None, None, 'empirical_paired_arm'
            if self.table and self.table['head']=='nvfp4' and head=='nvfp4':
                passes = self._passes(c, domain)
                pass_ms = cats.get('draft_graph', 0) / passes
                basis = 'measured_profile' if c==self.table['c'] and domain==self.table['domain'] else 'estimated_profile_transfer'
            rows = None if k == 4 else rows
            scenario.append('Measured T=0 thinking-off paired arm; transfer to default thinking/sampling estimated. No pass/row sensitivity identified.')
        ps = accept_pos or {}
        if any(int(i) < 0 or int(i) >= k for i in ps):
            raise ValueError('acceptance position outside K')
        scale = (accept_scale or {}).get(domain, 1.)
        if set(accept_scale or {}) - set(DOMAINS):
            raise ValueError('unknown acceptance domain')
        finite(scale, 0, 10)
        survival = [p * scale + ps.get(str(i), 0) for i, p in enumerate(survival)]
        validate_survival(survival)
        committed = 1 + sum(survival)
        if draft_pass_delta:
            if not self.table or passes is None or not self.table.get('pass_count_frequencies'):
                raise ValueError('draft-pass sensitivity unidentified; executed pass distribution required')
            cats['draft_graph'] += passes * finite(draft_pass_delta)
            if cats['draft_graph'] < 0:
                raise ValueError('draft-pass edit exceeds budget')
        # One atom ledger backs category, kernel, name and class aliases. Every
        # edit touches the same time/traffic balances; final floors run last.
        atoms = []
        if self.table:
            lead = min(s['rank'] for s in self.table['ranks'])
            for r in self.table['kernels']:
                if r['rank'] != lead:
                    continue
                cat = r['category']
                factor = cats.get(cat, 0) / max(1e-9, self.table['category_ms'].get(cat, 0))
                t = r['total_us_per_cycle']['mean'] / 1000 * factor
                traffic_factor = factor
                if cat == 'draft_graph':
                    traffic_factor = self._passes(c, domain) / self.table['executed_passes']
                    if r.get('projection') == 'head':
                        head_bytes = {'bf16':475791360, 'nvfp4':133816324}
                        traffic_factor *= head_bytes[head] / head_bytes[self.table['head']]
                atoms.append({'id':r['id'], 'category':cat, 'name':r.get('name'), 'class':r.get('class'),
                              'time':t, 'bytes':None if r['estimated_bytes_read'] is None else r['estimated_bytes_read']*traffic_factor,
                              'changed':bool(draft_pass_delta and cat=='draft_graph')})
            for cat, value in cats.items():
                remainder = value - sum(a['time'] for a in atoms if a['category']==cat)
                if remainder > 1e-9:
                    atoms.append({'id':'unbound/'+cat, 'category':cat, 'time':remainder, 'bytes':None, 'changed':False})
                elif remainder < -1e-6:
                    raise ValueError('table category mean smaller than kernel mean sum')
        else:
            for cat, t in cats.items():
                traffic = data['category_bytes'].get(cat)
                if traffic is not None:
                    traffic *= t / max(1e-9, data['category_prior_ms'].get(cat, t))
                atoms.append({'id':cat, 'category':cat, 'time':t, 'bytes':traffic, 'changed':False})
        bindings = {}
        for a in atoms:
            for key in set(filter(None, (a['id'], a['category'], a.get('name'), a.get('class')))):
                bindings.setdefault(key, []).append(a)
        km, kg, be = kernel_ms or {}, kernel_gbs or {}, byte_edits or {}
        unknown = (set(km) | set(kg) | set(be)) - set(bindings)
        if unknown:
            raise ValueError('unknown category/kernel: ' + ','.join(sorted(unknown)))
        traffic_keys = set(kg) | set(be)
        touched = set()
        for key in sorted(traffic_keys):
            ids = {a['id'] for a in bindings[key]}
            if touched.intersection(ids):
                raise ValueError('overlapping traffic edits require one category or disjoint kernels')
            touched.update(ids)
        # Apply traffic/bandwidth first, all time savings last. CLI spelling or
        # alias sort order cannot silently erase a saving or evade the floor.
        for stage, keys in (('traffic', traffic_keys), ('time', set(km))):
            for key in sorted(keys):
                bound = bindings[key]
                original = sum(a['time'] for a in bound)
                traffic = sum(a['bytes'] for a in bound) if all(a['bytes'] is not None for a in bound) else None
                replacement = original
                if stage == 'traffic':
                    if traffic is None and key not in be:
                        raise ValueError('unknown bytes; supply explicit --bytes')
                    new_traffic = finite(be[key], 0) if key in be else traffic
                    bandwidth = kg.get(key)
                    if bandwidth is None:
                        if not traffic or original <= 0:
                            raise ValueError('bytes-only edit requires original traffic/time')
                        bandwidth = traffic / (original * 1e6)
                    replacement = new_traffic / (min(finite(bandwidth, 1e-9), data['dram_ceiling_gbs']) * 1e6)
                    for a in bound:
                        fraction = a['bytes']/traffic if traffic else a['time']/max(original,1e-9)
                        a['bytes'] = new_traffic * fraction
                replacement += finite(km.get(key, 0)) if stage == 'time' else 0
                if replacement < -1e-9:
                    raise ValueError('combined edits exceed kernel/category budget')
                for a in bound:
                    fraction = a['bytes']/new_traffic if stage=='traffic' and new_traffic else (a['time']/original if original else 1/len(bound))
                    a['time'] = max(0, replacement) * fraction
                    a['changed'] = True
        for a in atoms:
            if a['time'] < -1e-9 or (a['changed'] and a['bytes'] is not None and
                    a['time'] + 1e-9 < a['bytes'] / (data['dram_ceiling_gbs'] * 1e6)):
                raise ValueError('final edited kernel time violates traffic/DRAM floor: ' + a['id'])
        cats = {cat:sum(a['time'] for a in atoms if a['category']==cat) for cat in cats}
        cycle = sum(cats.values()) + closure + empirical_delta
        floor = sum(a['bytes'] or 0 for a in atoms) / (data['dram_ceiling_gbs']*1e6)
        if (km or kg or be or draft_pass_delta) and cycle + 1e-9 < floor:
            raise ValueError('final cycle violates aggregate known-traffic DRAM floor')
        if cycle <= 0:
            raise ValueError('nonpositive cycle after edits')
        hard = committed / cycle * 1000
        factor = data['external_factors'][domain] * data['external_c_factors'][str(c)]
        external = hard * factor
        baseline_cycle = sum(base_cats.values()) + closure
        baseline_committed = 1 + sum(baseline_survival)
        baseline_hard = baseline_committed / baseline_cycle * 1000
        error = data['errors']['c1' if c == 1 else 'c2_c4']['max_abs_pct'] / 100
        column = {'cycle_ms':cycle/factor, 'committed_per_cycle':committed, 'tok_s_per_stream':external,
                  'aggregate_tok_s':c*external, 'status':'estimated sparkDash-only instrument transfer; RigMark unfitted'}
        rig_fit = data.get('rigmark_factors',{}).get(domain) if c==1 else None
        rig_column = {'status':'unknown; no matched instrument/concurrency fit'}
        if rig_fit:
            rig_column = {'tok_s_per_stream':hard*rig_fit['factor'], 'cycle_ms':cycle/rig_fit['factor'],
                          'committed_per_cycle':committed, 'aggregate_tok_s':hard*rig_fit['factor'],
                          'status':'estimated independent RigMark T=0 low-effort panel fit; run5 held out; prompt generalization unknown'}
        return {**self.provenance(),
                'c':c, 'domain':domain, 'k':k, 'requested_k':requested_k, 'head':head, 'policy':policy,
                'tau':tau, 'status':'estimated', 'scenario_basis':basis,
                'cycle_bench_hard_panel': {'cycle_ms':cycle, 'committed_per_cycle':committed,
                    'tok_s_per_stream':hard, 'aggregate_tok_s':c*hard, 'status':'estimated'},
                'sparkdash_calibrated': column, 'rigmark_calibrated':rig_column,
                'sparkdash_rigmark_calibrated': dict(column, deprecated_alias_for='sparkdash_calibrated'),
                'delta_vs_baseline': {'cycle_ms':cycle-baseline_cycle, 'committed_per_cycle':committed-baseline_committed,
                    'hard_tok_s':hard-baseline_hard, 'calibrated_tok_s':external-baseline_hard*factor},
                'uncertainty': {'calibrated_tok_s_band':[external/(1+error),external/max(.01,1-error)],
                    'hard_tok_s_band':[hard/(1+error),hard/max(.01,1-error)], 'calibration_max_abs_pct':100*error,
                    'status':'estimated throughput transfer max-error envelope; not a confidence interval'},
                'survival':survival, 'acceptance_status':'measured matching capture or estimated calibrated prefix/arm transfer', 'draft_passes_per_cycle':passes, 'draft_pass_ms':pass_ms,
                'passes_status': 'unknown for empirical paired arm' if passes is None else ('measured capture' if c==self.table['c'] and domain==self.table['domain'] else 'estimated capture transfer') if self.table else ('derived uniform K2' if c>1 else 'assumed prior; executed c1 distribution unknown'),
                'graph_rows':rows, 'expert_union':union, 'stale_row_penalty_ms':0,
                'routing_status':'assumed occupancy; reuse/pad hygiene prevent extra stale routes',
                'category_ms':cats, 'category_bytes': {cat:sum(a['bytes'] for a in atoms if a['category']==cat)
                    if all(a['bytes'] is not None for a in atoms if a['category']==cat) else None for cat in cats},
                'category_status':'estimated prior or measured expected-work table transfer; empirical arm delta separate',
                'empirical_cycle_delta_ms':empirical_delta,
                'closure_ms':closure, 'closure_status':'signed capture residual; includes overlap and profiler distortion',
                'kernel_table':table_note, 'notes':scenario+['c*per-stream assumes synchronous equal streams; asynchronous aggregate can differ.']}

    def sensitivity(self, c, domain):
        base = self.predict(c, domain)
        cycle = base['cycle_bench_hard_panel']['cycle_ms']
        committed = base['cycle_bench_hard_panel']['committed_per_cycle']
        factor = self.data['external_factors'][domain] * self.data['external_c_factors'][str(c)]
        return {**self.provenance(), 'c': c, 'domain': domain, 'status': 'estimated local derivative',
                'category_per_ms_saved': {cat: {'hard_tok_s': 1000*committed/cycle**2,
                                               'calibrated_tok_s': factor*1000*committed/cycle**2,
                                               'budget_ms': budget} for cat, budget in base['category_ms'].items()},
                'acceptance_per_position_plus_1pp': {str(i): {'hard_tok_s': 10/cycle, 'calibrated_tok_s': factor*10/cycle,
                    'feasible': p + .01 <= (1 if i == 0 else base['survival'][i-1])}
                    for i, p in enumerate(base['survival'])},
                'scope': 'one percentage point of unconditional prefix survival; downstream conditional probabilities held unspecified'}

    def targets(self):
        p = self.predict(1, 'prose')
        hard = p['cycle_bench_hard_panel']
        factor = self.data['external_factors']['prose']
        return [{**self.provenance(), 'target_tok_s': rate, 'column': col, 'status': 'estimated',
                 'cycle_ms_at_current_acceptance': 1000*hard['committed_per_cycle']*f/rate,
                 'committed_at_current_cycle': rate*hard['cycle_ms']/1000/f,
                 'required_prefix_sum': rate*hard['cycle_ms']/1000/f - 1,
                 'feasible_acceptance_only_K3': rate*hard['cycle_ms']/1000/f <= 4}
                for rate in (40, 50) for col, f in (('hard_panel', 1), ('calibrated', factor))]


def run_cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('predict', 'whatif', 'sensitivity', 'targets'):
        a = sub.add_parser(name)
        a.add_argument('--calibration', '--config', type=Path, required=True)
        a.add_argument('--kernel-table', type=Path)
        a.add_argument('--c', default='1')
        a.add_argument('--domain', default='prose')
        if name in ('predict', 'whatif'):
            a.add_argument('--k', type=int, default=3)
            a.add_argument('--head', choices=('bf16', 'nvfp4'), default='bf16')
            a.add_argument('--policy', choices=('kstop', 'fixed', 'uniform_k2'), default='kstop')
            a.add_argument('--tau', type=float, default=.90)
            for opt in ('kernel-ms', 'kernel-gbs', 'bytes', 'accept-pos', 'accept-scale'):
                a.add_argument('--'+opt, action='append', default=[])
            a.add_argument('--draft-pass-ms', type=float, default=0)
    a = p.parse_args(argv)
    try:
        model = Model(json.loads(a.calibration.read_text()), a.kernel_table)
        if a.command == 'targets':
            result = model.targets()
        else:
            result = []
            for c in map(int, a.c.split(',')):
                for domain in a.domain.split(','):
                    if a.command == 'sensitivity':
                        result.append(model.sensitivity(c, domain))
                    else:
                        result.append(model.predict(c, domain, a.k, a.head, a.tau, a.policy,
                            edits(a.kernel_ms), edits(a.kernel_gbs), edits(a.bytes), edits(a.accept_pos),
                            edits(a.accept_scale), a.draft_pass_ms))
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError) as e:
        p.exit(2, 'scenario rejected: %s\n' % e)


