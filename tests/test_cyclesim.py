# SPDX-License-Identifier: Apache-2.0
import copy
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "examples/cyclesim"
from cyclesim.utilities import JsonStream, acceptance, dump, sha256, trace_events, union_us, validate_survival
from cyclesim.model import Model, edits, union_experts
from cyclesim.profile import analyze, classify, ingest, check_capture
from cyclesim.build_ledger import weight


def trace(rank=0, overlap=False):
    events = []
    for cycle in range(6):
        ts = 1000 * cycle
        events += [{'ph': 'X', 'cat': 'kernel', 'name': 'fused_add_rms_norm', 'ts': ts, 'dur': 100,
                    'args': {'graph id': 1, 'correlation': cycle, 'graph node id': 1}},
                   {'ph': 'X', 'cat': 'kernel', 'name': 'marlin_moe', 'ts': ts + 50 if overlap else ts + 100, 'dur': 200,
                    'args': {'graph id': 1, 'correlation': cycle, 'graph node id': 2}},
                   {'ph': 'X', 'cat': 'kernel', 'name': 'oneshot_roce', 'ts': ts + 300, 'dur': 50,
                    'args': {'graph id': 1, 'correlation': cycle, 'graph node id': 3}}]
    return {'traceEvents': events, 'distributedInfo': {'rank': rank}, 'vllm_version': 'test'}


class Arithmetic(unittest.TestCase):
    def test_prefix_not_independent(self):
        validate_survival([.8, .6, .4])
        self.assertEqual(1 + sum([.8, .6, .4]), 2.8)
        with self.assertRaises(ValueError):
            validate_survival([.5, .7])

    def test_union(self):
        self.assertEqual(union_us([(0, 5), (3, 7), (8, 9)]), 8)
        self.assertAlmostEqual(union_experts(4), 17.6)
        self.assertGreater(union_experts(16), union_experts(12))

    def test_weight(self):
        self.assertEqual(weight(512, 6144, 'int4_g128'), 1622016)
        self.assertEqual(weight(512, 6144, 'nvfp4_s16'), 1769476)

    def test_prom_counter_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td)/'a', Path(td)/'b'
            names = [('num_drafts_total', '', 10), ('num_draft_tokens_total', '', 30),
                     ('num_accepted_tokens_total', '', 18), ('num_accepted_tokens_per_pos_total', ',position="0"', 8),
                     ('num_accepted_tokens_per_pos_total', ',position="1"', 6), ('num_accepted_tokens_per_pos_total', ',position="2"', 4)]
            def prom(offset):
                return ''.join('vllm:spec_decode_%s{model_name="x"%s} %s\n' % (n, l, v+offset) for n, l, v in names)
            a.write_text(''.join('vllm:spec_decode_%s{model_name="x"%s} 0\n' % (n,l) for n,l,v in names))
            b.write_text(prom(0))
            out = acceptance(a,b,3)
            self.assertEqual(out['survival'], [.8,.6,.4])
            self.assertAlmostEqual(out['committed'], 2.8)
            b.write_text(b.read_text().replace('} 10\n', '} -1\n'))
            with self.assertRaises(ValueError):
                acceptance(a,b,3)


class Streaming(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'trace.gz'
            with gzip.open(p, 'wt') as f:
                f.write(text)
            meta={}
            return list(trace_events(p, meta)), meta

    def test_boundaries_and_metadata(self):
        events, meta = self.parse(json.dumps({'traceEvents': [{'name':'x'*70000}, {'v':123456}], 'rank':2}))
        self.assertEqual(len(events[0]['name']), 70000)
        self.assertEqual(events[1]['v'], 123456)
        self.assertEqual(meta['rank'], 2)

    def test_bare_array(self):
        self.assertEqual(self.parse('[{"v":1}]')[0], [{'v':1}])

    def test_malformed(self):
        for text in ('{"traceEvents":[{}', '{"traceEvents":[{},]}', '{"traceEvents":[]} junk', '{"traceEvents":[],}', '{}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parse(text)

    def test_empty(self):
        self.assertEqual(self.parse('{"traceEvents":[]}')[0], [])


class Calibration(unittest.TestCase):
    def test_alternate_head_observations_retain_their_arm(self):
        from cyclesim.calibrate import validate
        calibration = json.loads((HERE/'calibration.json').read_text())
        observations = json.loads((HERE/'holdout.json').read_text())
        rows = validate(calibration, observations)
        self.assertEqual(len(rows), 6)
        self.assertAlmostEqual(rows[-2]['predicted'], 70.15895565529814)
        self.assertAlmostEqual(rows[-1]['predicted'], 78.87211168080572)
        self.assertLess(max(abs(row['error_pct']) for row in rows), 3.71)

    def test_holdout_prompt_overlap_rejected(self):
        from cyclesim.calibrate import validate
        calibration = json.loads((HERE/'calibration.json').read_text())
        observations = json.loads((HERE/'holdout.json').read_text())
        observations['observations'][0]['prompt_hashes'] = calibration['fit_prompt_hashes']
        with self.assertRaisesRegex(ValueError, 'holdout prompt leakage'):
            validate(calibration, observations)


class Profiles(unittest.TestCase):
    def setUp(self):
        self.ledger = json.loads((HERE/'ledger-historical.json').read_text())

    def test_tiny_medians_and_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rank0.pt.trace.json.gz'
            with gzip.open(p,'wt') as f:
                json.dump(trace(overlap=True), f)
            s, rows, cs = analyze(p,self.ledger,target_norms=1)
            self.assertEqual(s['cycles_kept'],4)
            self.assertEqual(s['cycle_ms']['median'],1)
            self.assertEqual(s['kernel_union_ms']['median'],.3)
            self.assertEqual(s['idle_ms']['median'],.7)
            self.assertEqual(s['kernel_count']['median'],3)
            self.assertEqual(len(rows),3)

    def test_class_priority(self):
        cases={'marlin_moe_wna16::Marlin':'marlin_moe','kernel_cutlass_roce':'allreduce/roce',
               '_mla_partial':'attention/mla','indexer_k_quant':'indexer','single_group_topk':'router',
               'nvfp4_gemm':'nvfp4_gemm','void marlin::Marlin':'int8_gemm','x_gemm_bf16':'bf16_gemm',
               'sampling':'sampling','rms_norm':'rmsnorm/elementwise'}
        for name, expected in cases.items():
            with self.subTest(name=name):self.assertEqual(classify(name),expected)

    def test_no_graph_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rank0.pt.trace.json'; dump(p,{'traceEvents':[]})
            with self.assertRaises(ValueError):analyze(p,self.ledger)

    def test_variable_kstop_passes_retained(self):
        doc=trace()
        for i in (1,3):
            doc['traceEvents'].append({'ph':'X','cat':'kernel','name':'draft_projection','ts':1000*i+400,'dur':50,
                                     'args':{'graph id':2,'correlation':i,'graph node id':1}})
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rank0.pt.trace.json';dump(p,doc)
            s,rows,cs=analyze(p,self.ledger,target_norms=1)
            self.assertEqual(s['cycles_kept'],4)
            self.assertEqual(s['draft_graph_passes']['p90'],1)

    def test_duplicate_rank_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for i in range(2):dump(root/('rank0-%d.pt.trace.json'%i),trace())
            with self.assertRaises(ValueError):ingest(root,self.ledger,1,3,1)

    def test_manifest_hash_and_companions(self):
        from test_cyclesim_fix1 import capture
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);capture(root,mode='companions-only')
            self.assertEqual(check_capture(root)['mode'],'companions-only')
            self.assertEqual(ingest(root,self.ledger,1,3,strict=True)['status'],'usable-companions-only')
            (root/'A1.json').write_text('corrupt')
            with self.assertRaises(ValueError):check_capture(root)

    def test_full_profile_capture_check(self):
        from test_cyclesim_fix1 import capture
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);capture(root)
            out=ingest(root,self.ledger,1,3,strict=True)
            self.assertEqual(len(out['ranks']),4)
            self.assertEqual(out['ranks'][0]['cycles_kept'],16)
            with self.assertRaises(ValueError):ingest(root,self.ledger,4,2,strict=True)



class Predictions(unittest.TestCase):
    def setUp(self):self.model=Model(json.loads((HERE/'calibration.json').read_text()))

    def test_baseline_fit(self):
        for domain in ('prose','code'):
            p=self.model.predict(domain=domain)
            self.assertAlmostEqual(p['cycle_bench_hard_panel']['cycle_ms'],self.model.data['domains'][domain]['cycle_fit_ms'])
            self.assertEqual(p['delta_vs_baseline']['cycle_ms'],0)

    def test_saved_ms(self):
        a=self.model.predict();b=self.model.predict(kernel_ms={'moe_routed':-.8})
        self.assertAlmostEqual(b['delta_vs_baseline']['cycle_ms'],-.8)
        self.assertGreater(b['cycle_bench_hard_panel']['tok_s_per_stream'],a['cycle_bench_hard_panel']['tok_s_per_stream'])

    def test_head_and_k4(self):
        for domain in ('prose','code','structured'):
            p=self.model.predict(domain=domain,head='nvfp4')
            self.assertLess(p['delta_vs_baseline']['cycle_ms'],0)
            p=self.model.predict(domain=domain,k=4)
            self.assertGreater(p['delta_vs_baseline']['cycle_ms'],0)

    def test_bad_inputs(self):
        for kwargs in ({'kernel_ms':{'moe_routed':-1000}}, {'draft_pass_delta':-100},
                       {'tau':.8}, {'kernel_ms':{'unknown':1}}, {'accept_pos':{'0':1}},
                       {'accept_pos':{'2':.8}}, {'kernel_gbs':{'mla':255}}, {'c':3}, {'k':5}):
            with self.subTest(kwargs=kwargs),self.assertRaises((ValueError,KeyError)):self.model.predict(**kwargs)

    def test_bandwidth_ceiling(self):
        a=self.model.predict(kernel_gbs={'vocab_head':270})
        b=self.model.predict(kernel_gbs={'vocab_head':1000})
        self.assertEqual(a['category_ms']['vocab_head'],b['category_ms']['vocab_head'])
        self.assertAlmostEqual(a['category_ms']['vocab_head'],475791360/270e6)

    def test_bandwidth_traffic_scales_with_union(self):
        a=self.model.predict(kernel_gbs={'moe_routed':255})
        b=self.model.predict(c=2,kernel_gbs={'moe_routed':255})
        ratio=self.model.predict(c=2)['category_ms']['moe_routed']/self.model.predict()['category_ms']['moe_routed']
        self.assertAlmostEqual(b['category_ms']['moe_routed']/a['category_ms']['moe_routed'],ratio)

    def test_prefix_edit(self):
        p=self.model.predict(accept_pos={'0':.02,'1':.03})
        self.assertAlmostEqual(p['delta_vs_baseline']['committed_per_cycle'],.05)
        self.assertEqual(edits(['template<a,b>=-0.5']), {'template<a,b>':-.5})

    def test_sensitivity_derivative(self):
        s=self.model.sensitivity(1,'prose')['category_per_ms_saved']['moe_routed']['hard_tok_s']
        p=self.model.predict(kernel_ms={'moe_routed':-.0001})['delta_vs_baseline']['hard_tok_s']/.0001
        self.assertAlmostEqual(s,p,places=5)

    def test_targets(self):
        rows=self.model.targets()
        self.assertEqual(len(rows),4)
        self.assertLess(rows[2]['cycle_ms_at_current_acceptance'],rows[0]['cycle_ms_at_current_acceptance'])

    def test_profile_replaces_prior_and_refits_closure(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'table.json'
            dump(p,{'schema':1,'c':1,'k':3,'policy':'kstop','head':'bf16','executed_passes':2.,'domain':'prose','stack':self.model.data['stack'], 'unprofiled_cycle_ms':73.4,
                    'category_ms':{'moe_routed':20,'draft_graph':10,'host_sampling':3},
                    'ranks':[{'rank':0}], 'kernels':[{'id':'r0-k0','rank':0,'category':'moe_routed',
                    'name':'marlin<a,b>','class':'marlin_moe',
                    'total_us_per_cycle':{'mean':10000,'median':10000},'estimated_bytes_read':2e9}]})
            model=Model(json.loads((HERE/'calibration.json').read_text()), kernel_table=p)
            a=model.predict();b=model.predict(kernel_ms={'r0-k0':-.8})
            self.assertEqual(set(a['category_ms']),{'moe_routed','draft_graph','host_sampling'})
            self.assertAlmostEqual(a['cycle_bench_hard_panel']['cycle_ms'],73.4)
            self.assertAlmostEqual(b['delta_vs_baseline']['cycle_ms'],-.8)
            self.assertAlmostEqual(model.predict(kernel_ms={'marlin<a,b>':-.8})['delta_vs_baseline']['cycle_ms'],-.8)
            self.assertAlmostEqual(model.predict(kernel_ms={'marlin_moe':-.8})['delta_vs_baseline']['cycle_ms'],-.8)
            with self.assertRaises(ValueError):model.predict(kernel_ms={'r0-k0':-11})
            with self.assertRaises(ValueError):model.predict(kernel_ms={'r0-k0':-3,'marlin_moe':-3})

    def test_cli_matrix(self):
        p=subprocess.run([sys.executable,'-m','cyclesim','whatif','--config',str(HERE/'calibration.json'),'--c','1,2,4','--domain','prose,code',
                          '--kernel-ms','moe_routed=-0.8'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(len(json.loads(p.stdout)),6)

    def test_c4_profile_closure_is_anchored_at_capture(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'table.json'
            dump(p,{'schema':1,'c':4,'k':2,'policy':'uniform_k2','head':'bf16','executed_passes':2.,'domain':'prose','stack':self.model.data['stack'],
                    'unprofiled_cycle_ms':150, 'category_ms':{'moe_routed':100,'draft_graph':10,'host_sampling':3},
                    'ranks':[{'rank':0}], 'kernels':[{'id':'r0-k0','rank':0,'category':'moe_routed',
                    'total_us_per_cycle':{'mean':100000,'median':100000},'estimated_bytes_read':2e9}]})
            out=Model(json.loads((HERE/'calibration.json').read_text()), kernel_table=p).predict(c=4)
            self.assertAlmostEqual(out['cycle_bench_hard_panel']['cycle_ms'],150)


if __name__=='__main__':unittest.main()
