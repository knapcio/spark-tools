#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare ingested profiles with saved gap-audit summaries, never a GPU run."""
import argparse
import json
from pathlib import Path
from cyclesim.utilities import dump, sha256


def validate(reference, c1, c4):
    rows=[]
    for name, doc in (('c1',c1),('c4',c4)):
        for rank in doc['ranks']:
            old=reference[name]['summaries'][str(rank['rank'])]
            for metric in ('cycle_ms','kernel_union_ms','idle_ms'):
                ref, actual=old[metric]['median'],rank[metric]['median']
                rows.append({'anchor':'%s/rank%d/%s'%(name,rank['rank'],metric),'reference':ref,
                             'ingested':actual,'error_pct':100*(actual/ref-1),'status':'measured timing comparison'})
            ref=old['device_activities']['median']
            rows.append({'anchor':'%s/rank%d/device_activity_count'%(name,rank['rank']),'reference':ref,
                         'ingested':rank['device_activity_count']['median'],'error_pct':100*(rank['device_activity_count']['median']/ref-1),
                         'status':'measured count comparison'})
    lead=[r for r in c1['kernels'] if r['rank']==0]
    int8=max((r for r in lead if r['class']=='int8_gemm' and r['phase']=='target'),
             key=lambda r:r['total_us_per_cycle']['median'])
    head=[r for r in lead if r['phase'].startswith('draft') and r['bytes_basis']=='vocab-shard dimensions']
    moe=[r for r in lead if r['class']=='marlin_moe' and r['phase']=='target']
    bandwidth=[{'anchor':'dominant target INT8 template','reference_gbs':192,'ingested_gbs':int8['achieved_gbs'],
                'error_pct':100*(int8['achieved_gbs']/192-1),'status':'estimated bytes/time; approximate prompt reference'},
               {'anchor':'draft BF16 vocab head','reference_gbs':250,'ingested_gbs':sum(r['achieved_gbs'] for r in head)/len(head),
                'error_pct':100*((sum(r['achieved_gbs'] for r in head)/len(head))/250-1),
                'status':'derived weight bytes / measured time; approximate prompt reference'},
               {'anchor':'target Marlin MoE aggregate','reference_interval_gbs':[179,216],
                'ingested_gbs':sum(r['estimated_bytes_read'] for r in moe)/sum(r['total_us_per_cycle']['median']*1000 for r in moe),
                'error_pct':None,'status':'estimated, assumed expert union; interval containment only'}]
    max_timing=max(abs(r['error_pct']) for r in rows)
    return {'schema':1,'timing_rows':rows,'bandwidth_rows':bandwidth,'max_abs_timing_error_pct':max_timing,
            'timing_within_2pct':max_timing<=2,
            'bandwidth_point_checks_within_2pct':all(abs(r['error_pct'])<=2 for r in bandwidth if r['error_pct'] is not None),
            'limits':['The saved reference has exact cycle/union/count references, but no exact per-template INT8/MoE GB/s reference.',
                      'Marlin179..216 is an approximate interval, not two independent point anchors; no endpoint reproduction is claimed.',
                      'Apportioning class bytes by calls cannot establish exact per-projection bandwidth or cache residency.']}


def render(d):
    lines=['# Saved profile validation','','| anchor | reference | ingested | error % |','|---|---:|---:|---:|']
    for r in d['timing_rows']:
        lines.append('| %s | %.6f | %.6f | %+.6f |'%(r['anchor'],r['reference'],r['ingested'],r['error_pct']))
    lines+=['','Times/counts measured. Bandwidth below is estimated from the weight ledger.','',
            '| bandwidth anchor | reference GB/s | ingested GB/s | error % |','|---|---:|---:|---:|']
    for r in d['bandwidth_rows']:
        lines.append('| %s | %s | %.3f | %s |'%(r['anchor'],r.get('reference_gbs',r.get('reference_interval_gbs')),
                     r['ingested_gbs'],'%+.3f'%r['error_pct'] if r['error_pct'] is not None else 'unknown (interval only)'))
    lines+=['']+d['limits']+['']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','c1','c4','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    doc=validate(json.loads(a.reference.read_text()),json.loads(a.c1.read_text()),json.loads(a.c4.read_text()))
    doc['source_sha256']={name:sha256(getattr(a,name)) for name in ('reference','c1','c4')}
    a.out.mkdir(parents=True,exist_ok=True)
    dump(a.out/'profile-validation.json',doc)
    (a.out/'PROFILE_VALIDATION.md').write_text(render(doc))
    print(json.dumps({k:doc[k] for k in ('max_abs_timing_error_pct','timing_within_2pct','bandwidth_point_checks_within_2pct')}))
    if not doc['timing_within_2pct'] or not doc['bandwidth_point_checks_within_2pct']:raise SystemExit(1)


if __name__=='__main__':main()
