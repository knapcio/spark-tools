# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 knapcio
"""Rank-shape cost model and held-out placement; arithmetic is estimated."""
from collections import Counter
import json
import math
import random
import statistics
import hashlib
from moereplay.geometry import LEGACY_GEOMETRY, Layout, load_geometry, resolve_layouts

LAYOUTS = {'TP4': (1, 4), 'EP2xTP2': (2, 2), 'EP4': (4, 1)}
STATUSES = ('measured', 'estimated', 'assumed')
BYTES_TP4 = 4866048  # 3*6144*512/2 packed INT4 + BF16 g128 scales


def placement(n, name, freq=None, geometry=None):
    # Adapted from knapcio's existing expert-placement analysis.
    # Stable descending frequency; capacity E/n; minimum load ties choose first rank.
    size = 256 if geometry is None else load_geometry(geometry).num_experts
    if type(n) is not int or not 1 <= n <= size:
        raise ValueError('EP group count must be in [1,num_experts]')
    if name == 'contiguous':
        return [e*n//size for e in range(size)]
    if name == 'round-robin':
        return [e % n for e in range(size)]
    if name != 'greedy' or freq is None:
        raise ValueError('unknown placement or missing held-out frequency')
    capacities = Counter(e*n//size for e in range(size))
    loads, counts, result = [0]*n, [0]*n, [0]*size
    for e in sorted(range(size), key=lambda e: -freq[e]):
        r = min((r for r in range(n) if counts[r] < capacities[r]), key=lambda r: loads[r])
        result[e] = r
        loads[r] += freq[e]
        counts[r] += 1
    return result


def rank_work(routes, layout, owners, hottest=(), geometry=None):
    if geometry is None:
        n, tp = LAYOUTS[layout]
        world, expert_bytes = 4, BYTES_TP4*n
    else:
        geometry = load_geometry(geometry)
        selected = resolve_layouts(geometry, [layout])[0]
        n, tp, world = selected.ep, selected.tp, geometry.world_size
        expert_bytes = None if geometry.expert_bytes is None else geometry.expert_bytes//tp
        if len(owners) != geometry.num_experts or any(type(r) is not int or not 0 <= r < n for r in owners):
            raise ValueError('owners must assign every expert to a valid EP group')
        if any(type(e) is not int or not 0 <= e < geometry.num_experts for e in hottest) or len(set(hottest)) != len(hottest):
            raise ValueError('invalid hottest experts')
        from moereplay.routes import valid_ids
        for row in routes:
            valid_ids(row, geometry)
        if not routes:
            raise ValueError('nonempty routes required')
    seen, slots = [set() for _ in range(n)], [0]*n
    multiplicities = [Counter() for _ in range(n)]
    per_row = [[] for _ in range(n)]
    patterns = [[] for _ in range(n)]
    local_indices = [{} for _ in range(n)]
    hot = set(hottest)
    for row in routes:
        row_slots = [0]*n
        row_patterns = [[-1]*len(row) for _ in range(n)]
        for slot,e in enumerate(row):
            # Replicas execute each occurrence exactly once. Assign hot occurrences
            # to least occupied group, preserving one sum over logical outputs.
            r = min(range(n), key=lambda g: (slots[g], g)) if e in hot else owners[e]
            seen[r].add(e)
            slots[r] += 1
            multiplicities[r][e] += 1
            row_slots[r] += 1
            row_patterns[r][slot] = local_indices[r].setdefault(e,len(local_indices[r]))
        for r in range(n):
            per_row[r].append(row_slots[r])
            patterns[r].append(row_patterns[r])
    distinct = [len(s) for s in seen]
    # Group members share exactly the same distinct experts and TP-shard bytes.
    physical_d = [d for d in distinct for _ in range(tp)]
    return dict(D_union=len(set(e for row in routes for e in row)), D_r=physical_d,
                max_rank_D=max(physical_d), mean_rank_D=sum(physical_d)/world,
                max_over_mean=max(physical_d)/(sum(physical_d)/world), bytes_r=[d*expert_bytes if expert_bytes is not None else None for d in physical_d],
                routed_slots_r=[s for s in slots for _ in range(tp)],
                local_slots_per_row_r=[s for s in per_row for _ in range(tp)],
                expert_multiplicities_r=[sorted(s.values()) for s in multiplicities for _ in range(tp)],
                local_route_sha256_r=[hashlib.sha256(json.dumps(p,separators=(',',':')).encode()).hexdigest() for p in patterns for _ in range(tp)],
                duplicate_rows=sum(len(set(row)) != len(row) for row in routes),
                bytes_status='estimated' if expert_bytes is not None else 'unavailable',
                bytes_scope='packed weights + scales touched once per distinct expert; not DRAM counters')


class Costs:
    def __init__(self, fit=None, geometry=None, layouts=None):
        self.fits = {}
        self.source = None
        self.geometry = load_geometry(geometry) if geometry is not None else LEGACY_GEOMETRY
        self.explicit_geometry = geometry is not None
        self.layouts = {l.name: l for l in resolve_layouts(self.geometry, layouts)}
        from pathlib import Path
        self.prior = json.loads(Path(__file__).with_name('prior.json').read_text())
        if fit is not None:
            from moereplay.routes import sha256
            data = json.loads(Path(fit).read_text())
            if 'geometry' in data:
                if load_geometry(data['geometry']) != self.geometry or data.get('geometry_fingerprint') != self.geometry.fingerprint:
                    raise ValueError('cost fit geometry or fingerprint mismatch')
                if data.get('schema') != 'moe-cost-fit v2':
                    raise ValueError('geometry-bound fit requires moe-cost-fit v2')
                bound = resolve_layouts(self.geometry, data.get('layouts', []))
                for l in bound:
                    if l.name in self.layouts and self.layouts[l.name] != l:
                        raise ValueError('cost fit layout mismatch')
            elif self.explicit_geometry:
                raise ValueError('explicit geometry requires a geometry-bound cost fit')
            elif data.get('schema') == 'moe-cost-fit v2' or 'geometry_fingerprint' in data:
                raise ValueError('cost fit geometry required')
            from moereplay.cost_adapter import deployment_records
            data, self.adapter = deployment_records(data, Path(fit), self.geometry if 'geometry' in data else None,
                                                     data.get('layouts') if 'geometry' in data else None)
            records = data['per_layout_M']
            self.source = dict(file=Path(fit).name, sha256=sha256(fit))
            for r in records:
                key = (r['layout'], r['M'])
                if key[0] not in self.layouts or type(key[1]) is not int or key[1] <= 0 or key in self.fits:
                    raise ValueError('invalid or duplicate fit layout/M')
                a, b = r['a_us'], r['b_us_per_distinct_local_expert']
                if any(type(x) not in (int, float) or not math.isfinite(x) for x in (a,b)) or b < 0:
                    raise ValueError('fit coefficients must be finite; slope must be nonnegative')
                status = r.get('status', data.get('status', 'measured'))
                if status not in STATUSES:
                    raise ValueError('fit status invalid')
                points = r.get('points', [])
                ds = [p['D'] for p in points]
                if any(type(d) not in (int, float) or not math.isfinite(d) or d < 0 for d in ds):
                    raise ValueError('invalid fit D points')
                self.fits[key] = dict(a_us=a, b_us_per_distinct_local_expert=b, status=status,
                                      source=self.source, D_range=[min(ds),max(ds)] if ds else None,
                                      rmse_us=r.get('rmse_us', 'unknown'), points=points,
                                      raw_fit=r.get('raw_fit','unknown'),
                                      excluded_points=r.get('excluded_points', []),
                                      applicability='exact measured workload signature and matched-route validation required')
                if self.explicit_geometry:
                    self.fits[key].update(geometry_fingerprint=self.geometry.fingerprint,
                                          layout=self.layouts[key[0]].to_dict(), M=key[1])

    def resolve(self, layout, m):
        if isinstance(layout, Layout):
            layout = layout.name
        if layout not in self.layouts:
            raise ValueError('unknown cost layout')
        if self.explicit_geometry and (type(m) is not int or m < 1):
            raise ValueError('positive M required')
        if (layout, m) in self.fits:
            return self.fits[layout,m]
        if self.explicit_geometry:
            return dict(status='unavailable', calibration_status='unavailable',
                        geometry_fingerprint=self.geometry.fingerprint, source=self.source,
                        layout=self.layouts[layout].to_dict(), M=m,
                        reason='no geometry-bound timing fit for layout and M')
        n, _ = LAYOUTS[layout]
        return dict(a_us=self.prior['a_us'], b_us_per_distinct_local_expert=self.prior['b_us_per_distinct_local_expert']*n,
                    status='estimated', calibration_status='measured' if n == 1 else 'estimated',
                    source=self.prior['source'], points=self.prior['points'],
                    D_range=[8,17], M_cells=[p['M'] for p in self.prior['points']],
                    historical_approximation=dict(a_us=20.,b_us_per_distinct_local_expert=21.),
                    rmse_us=self.prior['rmse_us'], missing_fit=self.source is not None,
                    prior=True, M_dependence='cross-M OLS; only the listed M,D cells were measured')

    def evaluate(self, layout, m, ds, work=None):
        if isinstance(layout, Layout):
            layout = layout.name
        if type(m) is not int or m < 1:
            raise ValueError('positive M required')
        fit = self.resolve(layout, m)
        if self.explicit_geometry and (len(ds) != self.geometry.world_size or any(type(d) is not int or not 0 <= d <= min(self.geometry.num_experts, self.geometry.top_k*m) for d in ds)):
            raise ValueError('rank distinct counts do not match geometry')
        if self.explicit_geometry:
            tp = self.layouts[layout].tp
            if any(len(set(ds[start:start+tp])) != 1 for start in range(0, len(ds), tp)):
                raise ValueError('TP group ranks must share distinct expert counts')
        if fit['status'] == 'unavailable':
            return dict(rank_us=[None]*len(ds), layer_us=None, fit=fit, status='unavailable',
                        extrapolation='unknown', calibration_eligible=False,
                        calibration_reasons=[fit['reason']], M_extrapolation=True,
                        negative_prediction_clipped_r=[False]*len(ds))
        raw = [fit['a_us'] + fit['b_us_per_distinct_local_expert']*d for d in ds]
        # OLS may fit a negative intercept. Preserve it; idle-rank extrapolation
        # cannot produce physical negative latency. Clipping is explicit.
        values = [max(0.,v) for v in raw]
        drange = fit['D_range']
        extra = 'unknown' if drange is None else any(not drange[0] <= d <= drange[1] for d in ds)
        reasons = set()
        if extra is not False:
            reasons.add('D extrapolation or unknown range')
        if fit.get('prior'):
            if m not in fit['M_cells']:
                reasons.add('M outside measured prior cells')
            if not all(any(p['M']==m and p['D']==d for p in fit['points']) for d in ds):
                reasons.add('unmeasured prior M,D combination')
            reasons.add('historical routes lack matched workload validation' if layout=='TP4' else 'EP bytes scaling is unmeasured')
        if work is None:
            reasons.add('local slot/multiplicity metadata missing')
        else:
            if work['duplicate_rows']:
                reasons.add('invalid duplicate top-k routing')
            for rank,d in enumerate(ds):
                matches = [p for p in fit.get('points', []) if p['D']==d and
                    p.get('local_slots')==work['routed_slots_r'][rank] and
                    p.get('local_slots_per_row')==work['local_slots_per_row_r'][rank] and
                    p.get('expert_multiplicities')==work['expert_multiplicities_r'][rank] and
                    p.get('local_route_sha256')==work['local_route_sha256_r'][rank] and
                    p.get('duplicate_rows')==0]
                if not matches:
                    reasons.add('no matched local occupancy/multiplicity measurement')
                elif not any(p.get('matched_route_validation') is True for p in matches):
                    reasons.add('matched-route transfer validation unknown')
                elif fit['status'] != 'measured':
                    reasons.add('fit is not measured')
        if any(v < 0 for v in raw):
            reasons.add('negative latency clipped')
        return dict(rank_us=values, layer_us=max(values), fit=fit,
                    status='estimated', extrapolation=extra,
                    calibration_eligible=not reasons, calibration_reasons=sorted(reasons),
                    M_extrapolation=m not in fit.get('M_cells',[m]),
                    negative_prediction_clipped_r=[v < 0 for v in raw])


MIN_BOOTSTRAP = 200


def percentile(values, q):
    """Hyndman-Fan type 7: linear interpolation at (n-1)*q."""
    ordered = sorted(values)
    if not ordered or not 0 <= q <= 1:
        raise ValueError('nonempty values and q in [0,1] required')
    position = (len(ordered)-1)*q
    low, high = math.floor(position), math.ceil(position)
    # Keep serialized intervals stable across supported Python versions; their
    # arithmetic can otherwise differ by one binary-float ULP.
    return round(ordered[low]+(ordered[high]-ordered[low])*(position-low), 15)


def bootstrap(values, clusters, samples=200, seed=1234):
    """Paired inputs supply deltas; steps and request cohorts resampled separately."""
    if type(samples) is not int or samples != 0 and samples < MIN_BOOTSTRAP:
        raise ValueError('bootstrap 0 or >=200 required for a 95% interval')
    if len(values) != len(clusters):
        raise ValueError('one cluster per value required')
    if not values:
        return dict(mean='unknown', steps_ci95='unknown', prompts_ci95='unknown', status='estimated')
    mean = statistics.fmean(values)
    groups = {}
    for v, key in zip(values, clusters):
        groups.setdefault(tuple(key), []).append(v)
    # Overlapping request cohorts are not independent prompt clusters.
    cohorts = list(groups)
    request_cohort, overlap = {}, False
    for cohort in cohorts:
        for request in cohort:
            if request in request_cohort and request_cohort[request] != cohort:
                overlap = True
            request_cohort[request] = cohort
    def interval(mode):
        if samples == 0 or len(values) < 2 or mode == 'prompts' and (len(groups) < 2 or overlap):
            return 'unknown'
        rng = random.Random(seed)
        means = []
        for _ in range(samples):
            if mode == 'steps':
                means.append(sum(values[rng.randrange(len(values))] for _ in values)/len(values))
            else:
                chosen = [groups[rng.choice(cohorts)] for _ in cohorts]
                means.append(sum(sum(g) for g in chosen)/sum(len(g) for g in chosen))
        return [percentile(means,.025), percentile(means,.975)]
    return dict(mean=mean, steps_ci95=interval('steps'), prompts_ci95=interval('prompts'),
                status='estimated', bootstrap_samples=samples, steps=len(values), prompt_cohorts=len(groups),
                percentile_method='Hyndman-Fan type 7', minimum_bootstrap_samples=MIN_BOOTSTRAP,
                prompt_limitation='overlapping cohorts' if overlap else 'unknown with fewer than two independent cohorts',
                scope='sampling uncertainty only; no cost-fit or temporal-correlation uncertainty')
