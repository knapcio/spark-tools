# SPDX-License-Identifier: Apache-2.0
"""Calibration-derived timing estimates; never a GPU/performance qualification."""
import bisect
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def interpolate(points, x):
    points = sorted(points)
    i = bisect.bisect_right([p[0] for p in points], x)
    if i == 0: return points[0][1]
    a, b = (points[-2],points[-1]) if i == len(points) else (points[i-1],points[i])
    return a[1] + (b[1]-a[1])*(x-a[0])/(b[0]-a[0])


class Timing:
    def __init__(self, calibration=None, cycle_table=None, cyclesim_path=None, matched_mixed=False):
        if calibration is None: raise ValueError('explicit timing calibration required')
        self.data = calibration
        if cyclesim_path: raise ValueError('supply a cycle table in this portable package')
        self.cycles = cycle_table or self.data['cycle_table']
        self.cycle_provider = {'provider':'explicit cycle-table','evidence':'caller-supplied points'}
        self.sources = set()
        self.unsupported = set()
        self.matched_mixed = matched_mixed
        self.prefill_models = self.data['prefill_chunk_model']
        fit_sources = [source for model in self.prefill_models.values() for source in model['fit_sources']]
        fit_sources += [self.data['mixed_fit']['source'],self.data['decode_context_source']]
        fit_sources += [point['source'] for point in self.data['committed_per_cycle'].values()]
        for source in fit_sources:
            if self.data['run_partitions'].get(source.split('/')[0]) != 'in-sample':
                raise ValueError('timing fit source must belong to a complete in-sample run')
        for rows in self.cycles.values():
            if not rows: raise ValueError('empty cycle table')
            widths = set()
            for row in rows:
                from lab.schedreplay.cycle_bridge import validate_contract
                validate_contract(row)
                if (type(row['M']) is not int or row['M'] <= 0 or row['M'] in widths
                        or not isinstance(row['ms'],(int,float)) or not math.isfinite(row['ms']) or row['ms'] <= 0):
                    raise ValueError('invalid cycle table point')
                if row.get('evidence') not in ('measured','estimated','assumed'):
                    raise ValueError('cycle point evidence required')
                widths.add(row['M'])
            if {row['c'] for row in rows} != {1,2,4}:
                raise ValueError('cycle table requires serving c1/c2/c4 contracts')
        if set(self.cycles) != {'prose','code'}: raise ValueError('cycle table needs prose and code')

    def cumulative(self, context):
        # Reconstruct actual adaptive chunks, including each launch overhead.
        regime = 4096 if context >= 16384 else 2048
        return self.prefill([(start,min(context,start+regime))
                             for start in range(0,int(context),regime)],
                            [regime]*int(math.ceil(context/regime)))

    def prefill(self, chunks, regimes=None):
        regimes = regimes if regimes is not None else [4096]*len(chunks)
        if len(regimes) != len(chunks): raise ValueError('one regime per prefill chunk required')
        total = 0
        for (start,end),regime in zip(chunks,regimes):
            if regime not in (2048,4096):
                self.unsupported.add('isolated prefill chunk regime %d has no fit'%regime)
            model = self.prefill_models[str(2048 if regime < 4096 else 4096)]
            if end > model['max_context']:
                self.unsupported.add('isolated prefill chunk %d lacks context anchors above %d tokens'%(regime,model['max_context']))
            if end-start > regime:
                self.unsupported.add('prefill work exceeds fitted per-chunk regime')
            n = end-start
            total += model['overhead_s'] + model['seconds_per_token']*n + model['context_seconds_per_token2']*(end*end-start*start)/2
        return total

    def execution_width(self, count):
        return 4 if count <= 1 else 3

    def cycle_point(self, kind, count):
        points = self.cycles[kind]
        if count in (1,2,4): return next(x for x in points if x['c']==count)
        self.unsupported.add('decode concurrency c%d lacks matched serving cost/acceptance contract'%count)
        # Diagnostics only; interpolate K2 costs/acceptance by concurrency,
        # never interpolate from a K3 graph width into a K2 table.
        return {key:interpolate([(x['c'],x[key]) for x in points if x['k']==2],count)
                for key in ('ms','committed_per_cycle')}

    def committed(self, kind, count=1):
        return self.cycle_point(kind,count)['committed_per_cycle']

    def decode(self, count, width, kind, context):
        if type(width) is not int or width != 4: raise ValueError('only configured K3 (requested width 4) supported')
        base = self.cycle_point(kind,count)['ms']/1000
        # Long-context small-gap cadence from long-during-decode, not a direct cycle bench.
        scale = 1 + self.data['decode_context_scale_per_60k']*min(context,262144)/60043
        return base*scale

    def cost(self, prefills, decoders, width, regimes=None):
        if type(width) is not int or width != 4: raise ValueError('only configured K3 (requested width 4) supported')
        regimes = regimes if regimes is not None else [4096]*len(prefills)
        prefill_s = self.prefill(prefills,regimes) if not decoders else 0
        if decoders:
            if len(decoders)>4: self.unsupported.add('decode concurrency above calibrated serving contracts')
            kind = 'code' if any(x[0]=='code' for x in decoders) else 'prose'
            decode_s = self.decode(len(decoders),width,kind,max(x[1] for x in decoders))
        else: decode_s = 0
        if prefills and decoders:
            tokens = sum(end-start for start,end in prefills)
            fit = self.data['mixed_fit']
            # Fit covers one stream, 60K A and ~100K B. Outside this cell
            # contextual scaling and additional decoder cost are assumptions.
            matched = (len(prefills) == 1 and len(decoders) == 1 and
                    decoders[0][0] == 'prose' and
                    all(r == 4096 for r in regimes) and
                    all(50000 <= x[1] <= 75000 for x in decoders) and
                    all(end <= 105000 for _,end in prefills))
            if self.matched_mixed and not matched:
                self.unsupported.add('conditional mixed-gap subset requires one 60K prose decoder/100K cold prefill, chunk 4096')
            if (len(prefills) != 1 or len(decoders) != 1 or
                    any(r not in (2048,4096) for r in regimes) or
                    any(not 50000 <= x[1] <= 75000 for x in decoders) or
                    any(end > 105000 for _,end in prefills)):
                self.unsupported.add('mixed timing outside one 60K decoder/100K prefill, chunk 2048/4096')
            reference = fit['intercept_s']+fit['seconds_per_token']*tokens
            # A supported mixed estimate cannot charge less than the same
            # isolated work. Only bound where that isolated regime is fitted;
            # isolated 2048 at long context remains a missing capture.
            isolated_bound = 0
            for (start,end),regime in ([] if self.matched_mixed and matched else zip(prefills,regimes)):
                model = self.prefill_models.get(str(regime))
                if model and end <= model['max_context']:
                    isolated_bound += self.prefill([(start,end)],[regime])
                else:
                    self.unsupported.add('mixed chunk %d lacks matched isolated context timing%s'%
                                         (regime,' above %d tokens'%model['max_context'] if model else ' (no isolated regime fit)'))
            if isolated_bound > reference:
                self.unsupported.add('isolated/mixed timing fits disagree; isolated lower bound assumed')
            prefill_s = isolated_bound
            # Isolated chunk costs are unavailable. Transfer the measured
            # mixed-gap fit unchanged across prefill context (assumption).
            # Cold TTFT anchors must not manufacture a per-chunk slope.
            cost = max(prefill_s,reference) + max(0,len(decoders)-1)*decode_s/len(decoders)
            evidence = ('estimated measured mixed-gap fit; conditional matched 4096 scope; isolated lower bound excluded'
                        if self.matched_mixed and matched else
                        'estimated measured mixed-gap fit; assumed context/concurrency scaling')
        elif prefills:
            cost = prefill_s
            evidence = 'estimated chunk size/context model; assumed per-chunk overhead and additive multi-prefill work'
            if len(prefills)>1: self.unsupported.add('multiple isolated prefills lack joint timing')
        elif decoders:
            cost = decode_s
            evidence = 'estimated cycle-table interpolation; assumed width/context/acceptance transfer'
        else: return 0, 'assumed empty scheduler step (zero GPU time)'
        self.sources.add(evidence)
        return max(.000001,cost), evidence
