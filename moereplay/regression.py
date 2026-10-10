# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 knapcio
"""Compatibility reproduction of knapcio's expert-placement analysis.

Legacy c4 greedy fits on half the SAME test prompts: intentional reproduction
ONLY, never the held-out replay policy. Torch randperm controls are preconverted
route bytes; regression runtime has no torch dependency.
"""
from collections import Counter
from pathlib import Path
import json
import math
import random
import statistics
from moereplay.model import placement


def reproduce(directory, version='frozen'):
    if version not in ('frozen','current'):
        raise ValueError('regression version must be frozen or current')
    root = Path(directory)
    data = (root/'layer-3.u8').read_bytes()
    if len(data) != 8192*8:
        raise ValueError('regression expects 8192x8 fixture')
    prom = [[data[(p*2048+i)*8:(p*2048+i+1)*8] for i in range(2048)] for p in range(4)]
    rng = random.Random(1234)
    def segments(test, c, M=4, nsamp=4000):
        for _ in range(nsamp):
            ps = rng.sample(test,c) if c <= len(test) else [rng.choice(test) for _ in range(c)]
            out = set()
            for p in ps:
                s = rng.randrange(0,2048-M)
                out.update(e for row in prom[p][s:s+M] for e in row)
            yield out
    def stats(wins, pl, n):
        totals, maxima = [],[]
        for u in wins:
            per = [0]*n
            for e in u:
                per[pl[e]] += 1
            totals.append(len(u)); maxima.append(max(per))
        T, MX = statistics.fmean(totals),statistics.fmean(maxima)
        MN = T/n
        return dict(D_union=round(T,2),max_rank=round(MX,2),mean_rank=round(MN,2),max_over_mean=round(MX/MN,3))
    res = {}
    for c in (1,2,4):
        row = {}
        for name,pname in (('contig','contiguous'),('rr','round-robin'),('greedy','greedy')):
            for n in (4,2):
                agg = []
                for test,train in (([0,1],[2,3]),([2,3],[0,1])):
                    freq = Counter(e for p in train for row in prom[p] for e in row)
                    pl = placement(n,pname,freq)
                    agg.append(stats(segments(test if version=='current' or c<=2 else [0,1,2,3],c),pl,n))
                row['%s_EP%d' % (name,n)] = {k:round((agg[0][k]+agg[1][k])/2,3) for k in agg[0]}
        uniform = (root/('uniform-c%d.u8' % c)).read_bytes()
        if len(uniform) != 4000*c*4*8:
            raise ValueError('preconverted uniform control length mismatch')
        for n in (4,2):
            size = c*4*8
            row['uniform_random_routing_EP%d' % n] = stats((set(uniform[i:i+size]) for i in range(0,len(uniform),size)),placement(n,'contiguous'),n)
        res['c%d_M%d' % (c,c*4)] = row
    freq = Counter(data)
    f = [freq[e] for e in range(256)]
    mean = statistics.fmean(f)
    res['layer3_freq'] = dict(max_over_mean=round(max(f)/mean,3),cv=round(statistics.stdev(f)/mean,3),
        top16_share=round(sum(sorted(f,reverse=True)[:16])/sum(f),3),
        contig_EP4_rank_share=[round(sum(f[i*64:(i+1)*64])/sum(f),4) for i in range(4)])
    res['union_vs_M_c1'] = {str(M):round(statistics.fmean(len(u) for u in segments([0,1,2,3],1,M,3000)),2) for M in (1,2,3,4,5,6,8)}
    return res


def compare(actual, expected, prefix=''):
    errors = []
    if isinstance(expected,dict):
        if set(actual) != set(expected):
            raise ValueError('regression keys differ at '+prefix)
        for k in expected:
            errors.extend(compare(actual[k],expected[k],prefix+'/'+str(k)))
    elif isinstance(expected,list):
        if len(actual) != len(expected):
            raise ValueError('regression list lengths differ')
        for i,v in enumerate(expected):
            errors.extend(compare(actual[i],v,prefix+'/'+str(i)))
    else:
        errors.append(dict(field=prefix,expected=expected,actual=actual,absolute_error=abs(actual-expected)))
    return errors
