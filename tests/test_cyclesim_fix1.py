# SPDX-License-Identifier: Apache-2.0
"""Reviewer failure reproductions: tiny, offline, independently asserted."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from cyclesim.model import Model
from cyclesim.utilities import dump, sha256
from cyclesim.profile import analyze, bytes_for, check_capture, ingest
from cyclesim.build_ledger import build
from test_cyclesim import trace
HERE = Path(__file__).resolve().parents[1] / "examples/cyclesim"


def table(head='bf16'):
    return {'schema':1,'c':1,'k':3,'domain':'prose','policy':'kstop','head':head,
            'executed_passes':2.,'stack':Model(json.loads((HERE/'calibration.json').read_text())).data['stack'],'unprofiled_cycle_ms':73.4,
            'category_ms':{'moe_routed':20.,'draft_graph':10.,'host_sampling':3.},
            'ranks':[{'rank':0}], 'kernels':[{'id':'r0-k0','rank':0,'category':'moe_routed',
            'name':'marlin<a,b>','class':'marlin_moe','total_us_per_cycle':{'mean':20000.,'median':20000.},
            'estimated_bytes_read':2e9,'working_set_bytes':2e9}]}


def capture(root, mode='profile', c=1):
    k=3 if c==1 else 2
    state={'c':c,'k':k,'domain':'prose','head':'bf16','policy':'kstop' if c==1 else 'uniform_k2',
           'capture_layout':'reuse','pad_hygiene':True}
    bracket=dict(state,client_cycle_ms=1.,instrument='client_decode_wall_per_draft',boot='fixture',
                 observations=16,errors=0,preemptions=0)
    for name in ('A1.json','A2.json','P.json'):dump(root/name,dict(bracket,iterations=18,record_shapes=False,peak_memory_debit_gib=.2,retained_memory_debit_gib=.1,
        memory={str(i):{'before_gib':8.,'minimum_gib':7.8,'after_gib':7.9,'peak_debit_gib':.2,'retained_debit_gib':.1} for i in range(4)}))
    env=dict(state,boot='fixture',image='fixture',cmd=['serve'],env={'MTP_KSTOP':'1',
             'MTP_UNIFORM_BATCH':'k2','MTP_CAPTURE_LAYOUT':'reuse','PAD_HYGIENE':'1',
             'DRAFT_HEAD':'0','DRAFT_HEAD_INIT':'0'})
    dump(root/'env.json',env)
    descriptors=[{'rows':q,'norms':16,'graph_id':i+1} for i,q in enumerate((2,3,4) if c==1 else (c*3,))]
    dump(root/'graph-sizes.json',dict(state,boot='fixture',configured=[1,4,12,16],ranks={str(i):dict(state,
        descriptors=descriptors,evidence='captured FULL graphs',executed_passes=2.) for i in range(4)}))
    dump(root/'prompts.json',{'prompts':{'prose':['p']*4,'code':['c']*4}})
    rows=[dict(arm=arm,request_id='%s-s%d'%(arm,s),stream=s,chunk_index=i,monotonic_ns=a*1000+i+1)
          for a,arm in enumerate(('A1','P','A2') if mode=='profile' else ('A1','A2'))
          for s in range(c) for i in range(17)]
    (root/'client-timings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    zero=('vllm:spec_decode_num_drafts_total 0\nvllm:spec_decode_num_draft_tokens_total 0\n'
          'vllm:spec_decode_num_accepted_tokens_total 0\n'+''.join(
          'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="%d"} 0\n'%i for i in range(k)))
    (root/'prom-before.prom').write_text(zero)
    (root/'prom-after.prom').write_text(zero.replace('num_drafts_total 0','num_drafts_total %d'%(16*c)).replace(
        'num_draft_tokens_total 0','num_draft_tokens_total %d'%(16*k*c)))
    if mode=='profile':
        for rank in range(4):
            doc={'distributedInfo':{'rank':rank},'traceEvents':[]}
            for cycle in range(18):
                for node in range(16):
                    doc['traceEvents'].append({'ph':'X','cat':'kernel','name':'fused_add_rms_norm',
                        'ts':1000*cycle+node*2,'dur':1,'args':{'graph id':1+cycle%3 if c==1 else 1,
                        'graph node id':node,'correlation':cycle}})
                for draft in range(1+cycle%3 if c==1 else 2):
                    doc['traceEvents'].append({'ph':'X','cat':'kernel','name':'fused_add_rms_norm',
                        'ts':1000*cycle+100+draft*10,'dur':1,'args':{'graph id':10+draft,
                        'graph node id':0,'correlation':cycle}})
            dump(root/('rank%d.pt.trace.json'%rank),doc)
    from cyclesim.capture import manifest
    manifest(root,mode,c,'prose',1.)


def mutate(root, name, fn):
    p=root/name;doc=json.loads(p.read_text());fn(doc);dump(p,doc)
    m=json.loads((root/'manifest.json').read_text())
    for r in m['files']:
        if r['path']==name:r['sha256']=sha256(p)
    dump(root/'manifest.json',m)


class Fix1(unittest.TestCase):
    def test_01_boot_policy(self):
        for c,rows in ((2,6),(4,12)):
            p=Model(json.loads((HERE/'calibration.json').read_text())).predict(c=c,domain='code')
            self.assertEqual(p['graph_rows'],rows)
            self.assertEqual(p['k'],2)
            self.assertEqual(p['draft_passes_per_cycle'],2)
            self.assertLessEqual(p['cycle_bench_hard_panel']['committed_per_cycle'],3)
            self.assertEqual(p['stale_row_penalty_ms'],0)

    def test_02_all_descriptors_and_optional_mean(self):
        doc=trace()
        for e in doc['traceEvents']:
            if e['ts']//1000 in (2,4):e['args']['graph node id']+=10
        doc['traceEvents'].append({'ph':'X','cat':'kernel','name':'draft_projection','ts':2400,'dur':80,
            'args':{'graph id':2,'correlation':2,'graph node id':1}})
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rank0.pt.trace.json';dump(p,doc)
            s,rs,_=analyze(p,json.loads((HERE/'ledger.json').read_text()),target_norms=1)
            self.assertEqual(s['cycles_kept'],4)
            self.assertEqual(len(s['target_descriptor_frequencies']),2)
            self.assertAlmostEqual(s['category_ms']['draft_graph'],.02)
            optional=next(r for r in rs if r['name']=='draft_projection')
            self.assertEqual(optional['total_us_per_cycle']['median'],0)
            self.assertEqual(optional['total_us_per_cycle']['mean'],20)

    def test_03_capture_domain_and_head(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'table.json';dump(p,table('nvfp4'));m=Model(json.loads((HERE/'calibration.json').read_text()), kernel_table=p)
            a=m.predict(domain='prose',head='nvfp4');b=m.predict(domain='code',head='nvfp4')
            self.assertNotEqual(a['category_ms']['draft_graph'],b['category_ms']['draft_graph'])
            self.assertAlmostEqual(a['category_ms']['draft_graph'],10)
            self.assertAlmostEqual(a['cycle_bench_hard_panel']['cycle_ms'],73.4)

    def test_04_one_traffic_and_final_floor(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'table.json';dump(p,table());m=Model(json.loads((HERE/'calibration.json').read_text()), kernel_table=p)
            a=m.predict(kernel_gbs={'moe_routed':255});b=m.predict(kernel_gbs={'marlin_moe':255})
            self.assertAlmostEqual(a['category_ms']['moe_routed'],b['category_ms']['moe_routed'])
            with self.assertRaises(ValueError):m.predict(kernel_gbs={'moe_routed':270},kernel_ms={'marlin_moe':-8})
            with self.assertRaises(ValueError):m.predict(kernel_ms={'moe_routed':-8,'marlin_moe':-8})

    def test_05_head_quant_and_mtp_traffic(self):
        # Self-contained dimensions also work against the exported old code;
        # missing repository fixtures must not masquerade as a regression.
        config={'hidden_size':6144,'moe_intermediate_size':2048,'intermediate_size':12288,
                'q_lora_rank':2048,'kv_lora_rank':512,'qk_rope_head_dim':64,
                'num_attention_heads':64,'qk_head_dim':256,'v_head_dim':256,
                'qk_nope_head_dim':192,'n_routed_experts':256,'vocab_size':154880}
        with tempfile.TemporaryDirectory() as td:
            release=Path(td);dump(release/'config.json',config)
            ledger=build(release,release/'config.json')
        self.assertEqual(ledger['cycle_budgets']['1']['draft']['nvfp4_gemm'],
            sum(r['weight_bytes_per_layer_per_rank'] for r in ledger['rows'] if r['group'] in ('mtp_attention','mtp_shared') and r['decode_read']))
        self.assertGreater(ledger['cycle_budgets']['1']['draft']['marlin_moe'], 0)
        b,w,_,_=bytes_for(ledger,'nvfp4_gemm','draft',1,True,1)
        self.assertEqual(b,133816324)
        self.assertEqual(w,b)
        b,w,_,_=bytes_for(ledger,'nvfp4_gemm','target',75,False,1)
        self.assertIsNone(w)  # class traffic / calls is not an actual working set

    def test_06_bad_bracket_and_coverage(self):
        for name,fn in [('A1.json',lambda d:d.pop('instrument')),
                        ('A2.json',lambda d:d.update(client_cycle_ms=2)),
                        ('graph-sizes.json',lambda d:d['ranks'].update({'0':{}})),
                        ('P.json',lambda d:d.update(head='nvfp4'))]:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as td:
                root=Path(td);capture(root);mutate(root,name,fn)
                with self.assertRaises(ValueError):check_capture(root)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);capture(root)
            mutate(root,'A1.json',lambda d:d.update(observations=1))
            with self.assertRaises(ValueError):check_capture(root)

    def test_07_prompt_split_and_metric_errors(self):
        from cyclesim.calibrate import split_prompt_roles
        obs=[{'role':'fit','prompt_hashes':['a']},{'role':'holdout','prompt_hashes':['a']},
             {'role':'holdout','prompt_hashes':['b']},{'role':'holdout','prompt_hashes':[]}]
        split_prompt_roles(obs)
        self.assertEqual(obs[1]['role'],'repeated-prompt-transfer')
        self.assertEqual(obs[2]['role'],'holdout')
        self.assertEqual(obs[3]['role'],'unknown-prompt-transfer')
        self.assertIn('metric_errors',Model(json.loads((HERE/'calibration.json').read_text())).data)
        self.assertNotIn('RigMark-calibrated',Model(json.loads((HERE/'calibration.json').read_text())).data['fit_protocol'])
        self.assertEqual(Model(json.loads((HERE/'calibration.json').read_text())).data['rigmark_factors']['prose']['held_out_run'],5)

    def test_08_unidentified_k_refused(self):
        for k in (0,1,2):
            with self.subTest(k=k),self.assertRaises(ValueError):Model(json.loads((HERE/'calibration.json').read_text())).predict(k=k)
        with self.assertRaises(ValueError):Model(json.loads((HERE/'calibration.json').read_text())).predict(draft_pass_delta=-.1)
        p=Model(json.loads((HERE/'calibration.json').read_text())).predict(k=4,head='nvfp4')
        self.assertEqual(p['scenario_basis'],'empirical_paired_arm')
        self.assertIsNone(p['draft_passes_per_cycle'])

    def test_09_minimal_capture(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);capture(root)
            out=ingest(root,json.loads((HERE/'ledger.json').read_text()),1,3,strict=True)
            self.assertEqual(out['ranks'][0]['cycles_kept'],16)
            self.assertEqual(out['head'],'bf16')
            self.assertEqual(len(out['ranks'][0]['target_graph_frequencies']),3)

    def test_capture_anchor_uses_actual_pass_mean(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'table.json';doc=table();doc['executed_passes']=2.5;dump(p,doc)
            result=Model(json.loads((HERE/'calibration.json').read_text()), kernel_table=p).predict()
            self.assertEqual(result['draft_passes_per_cycle'],2.5)
            self.assertAlmostEqual(result['cycle_bench_hard_panel']['cycle_ms'],73.4)

    def test_dh_head_traffic_follows_quantization(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'table.json';doc=table('nvfp4')
            doc['category_ms']={'draft_graph':10.,'host_sampling':3.}
            row=doc['kernels'][0]
            row.update(category='draft_graph',projection='head',phase='draft/1',
                       estimated_bytes_read=133816324,total_us_per_cycle={'mean':10000.,'median':10000.})
            dump(p,doc);m=Model(json.loads((HERE/'calibration.json').read_text()), kernel_table=p)
            self.assertEqual(m.predict(head='nvfp4')['category_bytes']['draft_graph'],133816324)
            self.assertEqual(m.predict(head='bf16')['category_bytes']['draft_graph'],475791360)
            doc['acceptance']={'survival':[0.,0.,0.]};dump(p,doc)
            result=Model(json.loads((HERE/'calibration.json').read_text()), kernel_table=p).predict(head='nvfp4')
            self.assertEqual(result['survival'],[0.,0.,0.])
            self.assertAlmostEqual(result['cycle_bench_hard_panel']['cycle_ms'],73.4)

    def test_explicit_dh_binding_and_real_working_set(self):
        doc=trace()
        for i in range(6):
            for name,ts,node in [('nvfp4_gemm',400,9),('rms_norm',500,10)]:
                doc['traceEvents'].append({'ph':'X','cat':'kernel','name':name,'ts':1000*i+ts,'dur':20,
                    'args':{'graph id':2,'correlation':i,'graph node id':node}})
        ledger=json.loads((HERE/'ledger.json').read_text())
        ledger['kernel_bindings']=[{'graph_id':2,'node_id':9,'phase':'draft','projection':'head',
                                  'quantization':'nvfp4','working_set_bytes':133816324}]
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rank0.pt.trace.json';dump(p,doc)
            _,rs,_=analyze(p,ledger,target_norms=1)
            head=next(r for r in rs if r['projection']=='head')
            self.assertEqual(head['estimated_bytes_read'],133816324)
            self.assertFalse(head['l2_resident'])
            moe=next(r for r in rs if r['class']=='marlin_moe')
            self.assertIsNone(moe['l2_resident'])

    def test_optional_traffic_has_occurrence_frequency(self):
        doc=trace()
        for node in range(4):
            doc['traceEvents'].append({'ph':'X','cat':'kernel','name':'nvfp4_gemm','ts':2400+node*20,'dur':10,
                'args':{'graph id':2,'correlation':2,'graph node id':node,'projection':'mtp_body'}})
        ledger=json.loads((HERE/'ledger.json').read_text())
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rank0.pt.trace.json';dump(p,doc)
            _,rs,_=analyze(p,ledger,target_norms=1)
            row=next(r for r in rs if r['phase']=='draft/1')
            self.assertAlmostEqual(row['estimated_bytes_read'],ledger['cycle_budgets']['1']['draft']['nvfp4_gemm']/4)

    def test_trace_pass_and_coverage_evidence_are_checked(self):
        for kind in ('passes','coverage','head'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                root=Path(td);capture(root)
                if kind=='passes':
                    mutate(root,'graph-sizes.json',lambda d:[r.update(executed_passes=3.) for r in d['ranks'].values()])
                elif kind=='coverage':
                    # IDs must be present in the trace, not merely stated.
                    mutate(root,'graph-sizes.json',lambda d:d['ranks']['0']['descriptors'][0].update(graph_id=99))
                else:
                    for p in root.glob('*.pt.trace.json'):
                        doc=json.loads(p.read_text())
                        for i in range(18):
                            doc['traceEvents'].append({'ph':'X','cat':'kernel','name':'nvfp4_gemm',
                                'ts':1000*i+105,'dur':1,'args':{'graph id':10,'graph node id':1,
                                'correlation':i,'projection':'head','quantization':'nvfp4'}})
                        dump(p,doc)
                    from cyclesim.capture import manifest
                    manifest(root,'profile',1,'prose',1.)
                with self.assertRaises(ValueError):ingest(root,json.loads((HERE/'ledger.json').read_text()),1,3,strict=True)

    def test_memory_peak_not_retention_and_floor(self):
        for fn in (lambda d:d['memory']['0'].update(minimum_gib=5.9),
                   lambda d:d['memory']['0'].update(peak_debit_gib=.1),
                   lambda d:d.pop('retained_memory_debit_gib')):
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);capture(root);mutate(root,'P.json',fn)
                with self.assertRaises(ValueError):check_capture(root)

    def test_c2_minimal_capture_boot_width(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);capture(root,c=2)
            out=ingest(root,json.loads((HERE/'ledger.json').read_text()),2,2,strict=True)
            self.assertEqual(out['policy'],'uniform_k2')
            self.assertEqual(out['executed_passes'],2.)

    def test_alias_edit_order_and_time_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'table.json';dump(p,table());m=Model(json.loads((HERE/'calibration.json').read_text()), kernel_table=p)
            result=m.predict(kernel_ms={'moe_routed':-.5})
            self.assertEqual(result['category_bytes']['moe_routed'],2e9)
            for key,alias in (('moe_routed','marlin_moe'),('marlin_moe','moe_routed')):
                with self.assertRaises(ValueError):m.predict(kernel_gbs={key:270},kernel_ms={alias:-.01})
            with self.assertRaises(ValueError):m.predict(kernel_gbs={'moe_routed':255,'marlin_moe':255})

    def test_stable_schedreplay_api(self):
        for c,rows in ((1,4),(2,6),(4,12)):
            p=Model(json.loads((HERE/'calibration.json').read_text())).predict(c=c,domain='prose',k=3,head='bf16',policy='kstop')
            self.assertEqual(p['graph_rows'],rows)
            self.assertIn('moe_routed',p['category_ms'])
            self.assertEqual(p['sparkdash_rigmark_calibrated']['deprecated_alias_for'],'sparkdash_calibrated')

    def test_raw_head_environment_is_explicit_and_supported(self):
        for fn in (lambda d:d['env'].pop('DRAFT_HEAD'),
                   lambda d:d['env'].update(DRAFT_HEAD='int8'),
                   lambda d:d['env'].update(DRAFT_HEAD='bf16')):
            with self.subTest(fn=fn),tempfile.TemporaryDirectory() as td:
                root=Path(td);capture(root);mutate(root,'env.json',fn)
                with self.assertRaises(ValueError):check_capture(root)

    def test_uniform_k2_rejects_one_three_pass_mixture(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);capture(root,c=2)
            for p in root.glob('*.pt.trace.json'):
                doc=json.loads(p.read_text())
                # Same mean of two passes, but no uniform-K2 execution.
                doc['traceEvents']=[e for e in doc['traceEvents'] if
                    not (e['args']['graph id']==11 and e['args']['correlation']%2)]
                for i in range(0,18,2):
                    doc['traceEvents'].append({'ph':'X','cat':'kernel','name':'fused_add_rms_norm',
                        'ts':1000*i+120,'dur':1,'args':{'graph id':12,'graph node id':0,'correlation':i}})
                dump(p,doc)
            from cyclesim.capture import manifest
            manifest(root,'profile',2,'prose',1.)
            with self.assertRaises(ValueError):ingest(root,json.loads((HERE/'ledger.json').read_text()),2,2,strict=True)

    def test_matching_dh_capture_preserves_acceptance_and_passes(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'table.json';doc=table('nvfp4')
            doc['executed_passes']=2.5
            doc['acceptance']={'survival':[.001,.001,.001]}
            dump(p,doc)
            result=Model(json.loads((HERE/'calibration.json').read_text()), kernel_table=p).predict(head='nvfp4')
            self.assertEqual(result['survival'],doc['acceptance']['survival'])
            self.assertEqual(result['draft_passes_per_cycle'],2.5)
            self.assertAlmostEqual(result['cycle_bench_hard_panel']['cycle_ms'],73.4)

if __name__=='__main__':unittest.main()
