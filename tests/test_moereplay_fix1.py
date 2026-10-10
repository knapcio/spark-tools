# SPDX-License-Identifier: Apache-2.0
"""One or more regressions for every finding in the fix1 review."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from collections import Counter

from test_moereplay import ROOT, create
from moereplay.capture_policy import CapturePolicy, remap_reference
from moereplay.cost_adapter import bench_signature
from moereplay.converters import convert_schema1
from moereplay.model import Costs, rank_work, placement, percentile, bootstrap
from moereplay.replay import replay, eligible_segments, prefill_events
from moereplay.regression import reproduce, compare
from moereplay.routes import Trace, Writer


def point(work, rank, us, validated=True):
    return dict(D=work['D_r'][rank],median_us=us,duplicate_rows=work['duplicate_rows'],
                local_slots=work['routed_slots_r'][rank],
                local_slots_per_row=work['local_slots_per_row_r'][rank],
                expert_multiplicities=work['expert_multiplicities_r'][rank],
                local_route_sha256=work['local_route_sha256_r'][rank],matched_route_validation=validated)


class ScratchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'))
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)


class CaptureTests(unittest.TestCase):
    def test_post_remap_dead_and_padded_rows_union(self):
        before=[bytes(range(i*8,i*8+8)) for i in range(5)]
        sources=[0,1,0,0,0]  # kstop dead proposals plus graph padding -> anchor
        after=remap_reference(before,sources)
        c=CapturePolicy([3]);c.install('g','decode-verify',5)
        self.assertTrue(c.begin(7,'g','decode-verify',5))
        c.copy_after_remap(3,after)
        self.assertEqual(len(set(e for row in c.finish()['experts'][0] for e in row)),16)
        self.assertEqual(len(set(e for row in before for e in row)),40)

    def test_remap_uses_snapshot(self):
        ids=[bytes(range(8)),bytes(range(8,16))]
        self.assertEqual(remap_reference(ids,[1,0]),ids[::-1])

    def test_prefill_profile_draft_no_copy_nodes_or_scratch(self):
        c=CapturePolicy([3])
        for phase,target,profile,m in [('prefill',True,False,4096),('decode-verify',True,True,8192),('decode-verify',False,False,16)]:
            self.assertFalse(c.install('skip',phase,m,target,profile))
            self.assertFalse(c.begin(0,'skip',phase,m,target,profile))
            self.assertFalse(c.copy_after_remap(3,[]))
            self.assertEqual(c.scratch,{})
        self.assertEqual((c.graphs,c.drops,c.overflows),(set(),0,0))

    def test_graph_transitions_disable_previously_installed_copy(self):
        c=CapturePolicy([3]);c.install('g4','decode-verify',4);c.install('g16','decode-verify',16)
        c.begin(0,'g4','decode-verify',4);c.copy_after_remap(3,[bytes(range(8))]*4);self.assertIsNotNone(c.finish())
        self.assertFalse(c.begin(1,'g16','decode-verify',16,transition=True))
        self.assertFalse(c.copy_after_remap(3,[bytes(range(8))]*16))
        self.assertIsNone(c.finish());self.assertEqual(c.scratch,{})
        self.assertTrue(c.begin(2,'g16','decode-verify',16))
        c.copy_after_remap(3,[bytes(range(8))]*16);self.assertEqual(c.finish()['step'],2)

    def test_decode_graph_construction_installs_nodes_with_collection_disabled(self):
        c=CapturePolicy([3])
        self.assertTrue(c.install('g','decode-verify',4))
        self.assertFalse(c.copy_after_remap(3,[bytes(range(8))]*4))
        self.assertIsNone(c.finish());self.assertEqual((c.scratch,c.records),({},0))

    def test_wrong_stage_and_missing_layer_drop_instead_of_stale_export(self):
        c=CapturePolicy([3,4]);c.install('g','decode-verify',4)
        c.begin(0,'g','decode-verify',4);self.assertFalse(c.copy_after_remap(3,[bytes(range(8))]*4,'pre-remap'))
        self.assertIsNone(c.finish())
        c.begin(1,'g','decode-verify',4);c.copy_after_remap(3,[bytes(range(8))]*4)
        self.assertIsNone(c.finish());self.assertEqual(c.drops,2)

    def test_bound_overflow_unknown_graph_and_disarmed_replay(self):
        c=CapturePolicy([3],max_rows=4,max_steps=1)
        c.install('g','decode-verify',4)
        self.assertFalse(c.begin(0,'g','decode-verify',16));self.assertEqual(c.overflows,1)
        self.assertFalse(c.begin(0,'unknown','decode-verify',4))
        c.begin(0,'g','decode-verify',4);c.copy_after_remap(3,[bytes(range(8))]*4);c.finish()
        self.assertFalse(c.begin(1,'g','decode-verify',4));self.assertFalse(c.copy_after_remap(3,[]))
        self.assertEqual(c.records,1)


class FitTests(ScratchTests):
    def write_fit(self, records, **extra):
        p=self.root/'fit.json';p.write_text(json.dumps(dict(per_layout_M=records,**extra)));return p

    def record(self,layout,points,m=4):
        return dict(layout=layout,M=m,a_us=2,b_us_per_distinct_local_expert=3,points=points,status='measured')

    def test_local_slots_and_multiplicity_not_just_M_D(self):
        a=[bytes(range(8))]*4
        b=[bytes(range(8)),bytes(range(8,16)),bytes(range(8,16)),bytes(range(8,16))]
        owner=placement(4,'contiguous')
        wa=rank_work(a,'EP4',owner)
        owner_b=[1]*256;owner_b[:8]=[0]*8
        wb=rank_work(b,'EP4',owner_b)
        self.assertEqual(wa['D_r'][0],wb['D_r'][0])
        self.assertEqual((wa['routed_slots_r'][0],wb['routed_slots_r'][0]),(32,8))
        # Two distinct valid D points; includes idle-rank direct measurement.
        wc=rank_work([bytes(range(16))[:8],bytes(range(8,16))]*2,'EP4',owner)
        cost=Costs(self.write_fit([self.record('EP4',[point(wa,0,26),point(wc,0,50)])]))
        self.assertFalse(cost.evaluate('EP4',4,wb['D_r'],wb)['calibration_eligible'])

    def test_multiplicity_and_exact_row_pattern_mismatch(self):
        owner=placement(1,'contiguous')
        a=[bytes(range(8))]*3+[bytes(range(8,16))]
        b=[bytes(range(8))]*2+[bytes(range(8,16))]*2
        wa,wb=(rank_work(r,'TP4',owner) for r in (a,b))
        self.assertEqual(wa['D_r'],wb['D_r']);self.assertEqual(wa['routed_slots_r'],wb['routed_slots_r'])
        self.assertNotEqual(wa['expert_multiplicities_r'],wb['expert_multiplicities_r'])
        wc=rank_work([bytes(range(i*8,i*8+8)) for i in range(4)],'TP4',owner)
        cost=Costs(self.write_fit([self.record('TP4',[point(wa,0,50),point(wc,0,98)])]))
        self.assertTrue(cost.evaluate('TP4',4,wa['D_r'],wa)['calibration_eligible'])
        self.assertFalse(cost.evaluate('TP4',4,wb['D_r'],wb)['calibration_eligible'])
        perm=rank_work(a[::-1],'TP4',owner)
        self.assertEqual(perm['expert_multiplicities_r'],wa['expert_multiplicities_r'])
        self.assertFalse(cost.evaluate('TP4',4,perm['D_r'],perm)['calibration_eligible'])

    def test_invalid_stress_points_excluded_and_valid_fit_recomputed(self):
        w=rank_work([bytes(range(8))]*4,'TP4',placement(1,'contiguous'))
        w2=rank_work([bytes(range(8)),bytes(range(8,16))]*2,'TP4',placement(1,'contiguous'))
        pts=[dict(D=1,median_us=999,duplicate_rows=4),point(w,0,26),point(w2,0,50)]
        c=Costs(self.write_fit([self.record('TP4',pts)]))
        self.assertEqual(c.resolve('TP4',4)['a_us'],2)
        self.assertEqual(c.resolve('TP4',4)['b_us_per_distinct_local_expert'],3)
        self.assertEqual([p['D'] for p in c.resolve('TP4',4)['points']],[8,16])
        self.assertEqual(len(c.adapter['excluded']),1)

    def test_unknown_or_unvalidated_work_cannot_be_eligible(self):
        w=rank_work([bytes(range(8))]*4,'TP4',placement(1,'contiguous'))
        w2=rank_work([bytes(range(8)),bytes(range(8,16))]*2,'TP4',placement(1,'contiguous'))
        c=Costs(self.write_fit([self.record('TP4',[point(w,0,26,False),point(w2,0,50,False)])]))
        t=c.evaluate('TP4',4,w['D_r'],w)
        self.assertFalse(t['calibration_eligible']);self.assertIn('matched-route transfer validation unknown',t['calibration_reasons'])

    def test_declared_duplicate_routes_are_ineligible(self):
        w=rank_work([[0]*8]*4,'TP4',placement(1,'contiguous'))
        t=Costs().evaluate('TP4',4,w['D_r'],w)
        self.assertIn('invalid duplicate top-k routing',t['calibration_reasons'])

    def test_schema1_preserves_post_remap_stage_and_rejects_stage_changes(self):
        source=self.root/'schema1.jsonl';out=self.root/'converted'
        record=dict(schema=1,step=0,layers=[3],physical_rows=4,k=3,concurrency=1,
                    route_stage='post-kstop-remap',accepted={'p':0},
                    rows=[dict(request_id='p',position=i,padding=False) for i in range(4)],
                    experts=[[list(range(8))]*4])
        source.write_text(json.dumps(record)+'\n');convert_schema1(source,out)
        with Trace(out) as t:self.assertEqual(t.manifest['source']['route_stage'],'post-kstop-remap')
        next_record=dict(record,step=1,route_stage='pre-remap')
        source.write_text(json.dumps(record)+'\n'+json.dumps(next_record)+'\n')
        with self.assertRaisesRegex(ValueError,'stage changed'):convert_schema1(source,self.root/'bad')


    def test_current_EP_results_companion_refits_valid_cases(self):
        p=self.write_fit([self.record('TP4',[dict(D=d,median_us=999) for d in (1,8,16)])],per_pass=[])
        rows=[]
        for d in (1,8,16):
            rows.append(dict(type='measurement',layout='TP4',M=4,D=d,median_us=2+3*d,**bench_signature('TP4',4,d)))
        (self.root/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        c=Costs(p);self.assertEqual(c.resolve('TP4',4)['a_us'],2)
        self.assertEqual(c.adapter['companion'],'results.jsonl')
        rows[1]['local_slots']=0;(self.root/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'drift'):Costs(p)

    def test_measured_prior_cells_and_D_M_extrapolation(self):
        c=Costs();p=c.resolve('TP4',4)
        self.assertEqual((p['a_us'],p['b_us_per_distinct_local_expert'],p['D_range']),(12.5,21.46,[8,17]))
        self.assertEqual([(r['M'],r['D']) for r in p['points']],[(1,8),(2,9),(3,9),(4,10),(6,11),(8,11),(12,17)])
        self.assertTrue(c.evaluate('TP4',16,[30]*4)['M_extrapolation'])
        self.assertTrue(c.evaluate('TP4',4,[30]*4)['extrapolation'])
        self.assertFalse(c.evaluate('TP4',4,[10]*4)['extrapolation'])
        self.assertFalse(c.evaluate('TP4',4,[10]*4)['calibration_eligible'])

    def test_candidate_eligible_but_TP4_baseline_not_exported(self):
        out=self.root/'trace';create(out,steps=2)
        # EP2 owns the test's D32 entirely: candidate rank0 has D32; rank1 idle.
        with Trace(out) as t:
            routes=[t.row(3,i) for i in range(4,8)]
        w=rank_work(routes,'EP2xTP2',placement(2,'contiguous'))
        cfile=self.write_fit([self.record('EP2xTP2',[point(w,0,98),point(w,2,2)])])
        cost=Costs(cfile);self.assertTrue(cost.evaluate('EP2xTP2',4,w['D_r'],w)['calibration_eligible'])
        with Trace(out) as t:r=replay(t,cfile,bootstrap_samples=0)
        key='observed/c1_M4_K3/EP2xTP2/contiguous'
        self.assertTrue(r['cycles'][key]['cycle_kernel_ms']['calibration_eligible'])
        self.assertFalse(r['cycles'][key]['TP4_baseline_calibration_eligible'])
        delta=r['cycles'][key]['delta_vs_TP4_ms'];self.assertFalse(delta['calibration_eligible'])
        self.assertTrue(any(x.startswith('TP4 baseline:') for x in delta['calibration_reasons']))
        export=next(x for x in r['cyclesim_what_if'] if x['control']=='observed' and x['layout']=='EP2xTP2' and x['placement']=='contiguous')
        self.assertEqual((export['argv'],export['kernel_ms']),([],{}));self.assertIn('moe_routed',export['sensitivity_kernel_ms'])

    def test_measured_exact_pair_export_and_unknown_stage_refusal(self):
        out=self.root/'trace';create(out,steps=2)
        with Trace(out) as t:routes=[t.row(3,i) for i in range(4,8)]
        w=rank_work(routes,'TP4',placement(1,'contiguous'))
        other=rank_work([bytes(range(8))]*4,'TP4',placement(1,'contiguous'))
        cfile=self.write_fit([self.record('TP4',[point(w,0,98),point(other,0,26)])])
        manifest=out/'manifest.json';m=json.loads(manifest.read_text());m['source'].update(status='measured',route_stage='post-kstop-remap');manifest.write_text(json.dumps(m))
        with Trace(out) as t:r=replay(t,cfile,bootstrap_samples=0)
        export=next(x for x in r['cyclesim_what_if'] if x['control']=='observed' and x['layout']=='TP4' and x['placement']=='contiguous')
        self.assertTrue(export['calibration_eligible']);self.assertEqual(export['kernel_ms'],{'moe_routed':0.})
        self.assertEqual(export['argv'][:4],['--c','1','--k','3'])
        del m['source']['route_stage'];manifest.write_text(json.dumps(m))
        with Trace(out) as t:r=replay(t,cfile,bootstrap_samples=0)
        self.assertFalse(any(x['calibration_eligible'] for x in r['cyclesim_what_if']))


class SegmentAndStatisticsTests(ScratchTests):
    def trace(self,segments):
        out=self.root/'prefill';w=Writer(out,[3],'test',dict(status='assumed'),'prefill-proxy')
        for step,requests in enumerate(segments):
            rows=[dict(request=req,position=pos,padding=False,phase='prefill',k=0,concurrency=1) for req,pos in requests]
            w.append(step,rows,{3:[list(range(8))]*len(rows)})
        w.finish();return Trace(out)

    def test_exact_four_and_last_segment_included(self):
        with self.trace([[('p',i) for i in range(5)],[('q',i) for i in range(4)]]) as t:
            segments=eligible_segments(t,['p','q'])
            self.assertEqual(segments,{'p':[(0,1,2,3),(1,2,3,4)],'q':[(5,6,7,8)]})
            self.assertEqual(len(list(prefill_events(t,3,['q'],1,7))),3)

    def test_request_step_and_position_gaps_never_concatenated(self):
        with self.trace([[('p',0),('p',1),('q',0),('q',1)],
                         [('p',2),('p',3)], [('p',4),('p',5),('p',9),('p',10)]]) as t:
            self.assertEqual(eligible_segments(t,['p','q']),{})
            with self.assertRaisesRegex(ValueError,'eligible'):list(prefill_events(t,3,['p'],1,7))

    def test_type7_percentile_small_example(self):
        self.assertEqual([percentile([0,10],q) for q in (.025,.975)],[.25,9.75])

    def test_minimum_bootstrap_replicates(self):
        for n in (-1,1,2,50,199):
            with self.assertRaisesRegex(ValueError,'>=200'):bootstrap([0,10],[['a'],['b']],n)
        self.assertEqual(bootstrap([0,10],[['a'],['b']],0)['steps_ci95'],'unknown')
        self.assertEqual(bootstrap([0,10],[['a'],['b']],200)['steps_ci95'],[0.,10.])
        with self.assertRaisesRegex(ValueError,'cluster'):bootstrap([1,2],[['a']],200)


if __name__=='__main__':unittest.main()
