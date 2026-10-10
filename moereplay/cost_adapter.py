# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 knapcio
"""EP benchmark adapter. Never use duplicate-ID stress cases in deployment OLS."""
import json
import math
import statistics
import hashlib


def ols(points):
    x = [p['D'] for p in points]
    y = [p['median_us'] for p in points]
    xb, yb = statistics.fmean(x), statistics.fmean(y)
    ss = sum((d-xb)**2 for d in x)
    if not ss:
        raise ValueError('valid deployment fit needs >=2 distinct D')
    b = sum((d-xb)*(t-yb) for d,t in zip(x,y))/ss
    a = yb-b*xb
    return dict(a_us=a, b_us_per_distinct_local_expert=b,
                rmse_us=math.sqrt(statistics.fmean((t-a-b*d)**2 for d,t in zip(x,y))))


def bench_signature(layout, m, d):
    """Existing synthetic benchmark column-major occupancy contract.

    Per-expert counts balanced; expert identities rotate, occupancy does not.
    This reconstructs workload metadata, never asserts real-route validation.
    """
    ep = {'TP4':1,'EP2xTP2':2,'EP4':4}[layout]
    if type(m) is not int or type(d) is not int or m<=0 or not 0<d<=min(256//ep,8*m):
        raise ValueError('invalid synthetic benchmark M,D')
    slots = 8*m if ep==1 else max(d, math.ceil(8*m/ep))
    row_counts = [sum(row==r for row,col in [(r,c) for c in range(8) for r in range(m)][:slots]) for r in range(m)]
    # Local fill is row-major AFTER choosing the column-major local positions.
    ids = [[None]*8 for _ in range(m)]
    order = [(r,c) for c in range(8) for r in range(m)]
    for j,(r,c) in enumerate(sorted(order[:slots])):
        ids[r][c] = j%d
    for j,(r,c) in enumerate(sorted(order[slots:])):
        ids[r][c] = 256+j  # remote expert identities are disjoint and unique per row
    pattern = [[e if e<256 else -1 for e in row] for row in ids]
    return dict(local_slots=slots,local_slots_per_row=row_counts,
                expert_multiplicities=sorted([slots//d+(i<slots%d) for i in range(d)]),
                local_route_sha256=hashlib.sha256(json.dumps(pattern,separators=(',',':')).encode()).hexdigest(),
                duplicate_rows=sum(len(set(row))!=8 for row in ids))


def deployment_records(data, path, geometry=None, layouts=None):
    if geometry is None:
        layout_groups = {'TP4':1, 'EP2xTP2':2, 'EP4':4}
        experts, top_k = 256, 8
    else:
        from moereplay.geometry import load_geometry, resolve_layouts
        geometry = load_geometry(geometry)
        layout_groups = {l.name:l.ep for l in resolve_layouts(geometry, layouts)}
        experts, top_k = geometry.num_experts, geometry.top_k
        if 'per_pass' in data:
            raise ValueError('legacy synthetic benchmark adapter requires legacy unbound cost schema')
    records = data.get('per_layout_M')
    if not isinstance(records,list) or not records:
        raise ValueError('cost fit requires nonempty per_layout_M list')
    companion = path.with_name('results.jsonl')
    measurements = []
    if companion.exists() and 'per_pass' in data:
        # Read bounded lines rather than a possibly large whole timing observation.
        with companion.open() as f:
            while True:
                line = f.readline(16*(1<<20)+1)
                if not line:
                    break
                if len(line)>16*(1<<20):
                    raise ValueError('EP measurement record exceeds 16 MiB')
                row = json.loads(line)
                if row.get('type')=='measurement':
                    measurements.append(row)
        if not measurements:
            raise ValueError('EP results companion has no measurements')
    out, excluded = [],[]
    keys = set()
    for record in records:
        r = dict(record)
        key = (r.get('layout'),r.get('M'))
        if key[0] not in layout_groups or type(key[1]) is not int or key[1]<=0 or key in keys:
            raise ValueError('invalid or duplicate fit layout/M')
        keys.add(key)
        if any(type(r.get(k)) not in (int,float) or not math.isfinite(r[k]) for k in ('a_us','b_us_per_distinct_local_expert')):
            raise ValueError('finite fit coefficients required')
        points = r.get('points',[])
        capacity = experts if geometry is not None else experts//layout_groups[r['layout']]
        if any(type(p.get('D')) is not int or p['D']<0 or p['D']>min(capacity,top_k*r['M']) for p in points):
            raise ValueError('invalid fit local D')
        # EP producer's combined OLS uses stress points too. With no companion,
        # its known synthetic routing contract still identifies invalid cases.
        if not measurements and 'per_pass' in data:
            points = [dict(p,**bench_signature(r['layout'],r['M'],p['D']),matched_route_validation=False) for p in points]
        # Any TP4 top8 row with total union <8 is necessarily invalid.
        points = [dict(p,duplicate_rows=r['M']) if layout_groups[r['layout']]==1 and p['D']<top_k else p for p in points]
        if measurements:
            selected = [p for p in measurements if p['layout']==r['layout'] and p['M']==r['M']]
            points = []
            for d in sorted({p['D'] for p in selected}):
                rows = [p for p in selected if p['D']==d]
                sig = bench_signature(r['layout'],r['M'],d)
                if any(any(p.get(k)!=sig[k] for k in ('local_slots','local_slots_per_row','duplicate_rows')) for p in rows):
                    raise ValueError('EP benchmark occupancy contract drift')
                points.append(dict(D=d,median_us=statistics.median(p['median_us'] for p in rows),
                                   matched_route_validation=False, **sig))
        rejected = [p for p in points if p.get('duplicate_rows',0)>0]
        r['excluded_points'] = rejected
        excluded.extend(dict(layout=r['layout'],M=r['M'],D=p['D'],reason='duplicate top-k IDs') for p in rejected)
        valid = [p for p in points if p.get('duplicate_rows')==0]
        # Unknown workload points can remain only as an explicitly unvalidated
        # sensitivity line; evaluate() cannot make them calibration eligible.
        if rejected or measurements or valid:
            if len({p['D'] for p in valid})<2:
                excluded.append(dict(layout=r['layout'],M=r['M'],reason='fewer than two valid deployment D points'))
                continue
            if any(type(p.get('median_us')) not in (int,float) or not math.isfinite(p['median_us']) or p['median_us']<0 for p in valid):
                raise ValueError('valid deployment points need finite nonnegative median_us')
            r.update(ols(valid),points=valid)
            r['raw_fit'] = {k:record.get(k) for k in ('a_us','b_us_per_distinct_local_expert','rmse_us')}
        out.append(r)
    from moereplay.routes import sha256
    return dict(data,per_layout_M=out),dict(status='measured' if measurements else 'assumed',
        companion=companion.name if measurements else 'unknown', excluded=excluded,
        companion_sha256=sha256(companion) if measurements else 'unknown',
        scope='valid-case OLS; matched-route transfer validation remains required')
