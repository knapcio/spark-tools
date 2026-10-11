# SPDX-License-Identifier: Apache-2.0
import contextlib
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import os
import time
import unittest
from unittest import mock
from moereplay import cli as moereplay_cli
from moereplay.routes import Writer, Trace, sha256, valid_ids, valid_row
from moereplay.model import placement, rank_work, Costs, BYTES_TP4, bootstrap
from moereplay.converters import convert_schema1
from moereplay.replay import replay, train_frequency, union_growth
from moereplay.regression import reproduce, compare
from moereplay.cli import run_cli as main

ROOT = Path(__file__).resolve().parents[1]
def create(out,layers=(3,),steps=4,rows_per_request=4,padding=0,c=1,requests=None):
    w = Writer(out,layers,'test',dict(status='assumed',kind='synthetic'),'decode-verify')
    for step in range(steps):
        rows, experts = [],{l:[] for l in layers}
        for req in range(c):
            request = requests[step] if requests else 'request-%d' % step
            if c>1:
                request += '-%d' % req
            for pos in range(rows_per_request):
                rows.append(dict(request=request,position=pos+100,padding=False,phase='decode',k=rows_per_request-1,concurrency=c))
                for l in layers:
                    experts[l].append(list(range((step+req+pos)*8%248,(step+req+pos)*8%248+8)))
        for _ in range(padding):
            rows.append(dict(request=None,position=None,padding=True,phase='decode',k=rows_per_request-1,concurrency=c))
            for l in layers:
                experts[l].append(list(range(248,256)))
        w.append(step,rows,experts)
    return w.finish()


class RoutesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'));self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name);self.out = self.root/'trace'

    def rehash(self,file):
        m=json.loads((self.out/'manifest.json').read_text())
        targets=[m['row_index']]+m['data']
        next(d for d in targets if d['file']==file)['sha256']=sha256(self.out/file)
        (self.out/'manifest.json').write_text(json.dumps(m))

    def test_roundtrip(self):
        create(self.out,layers=(3,4),padding=1)
        with Trace(self.out) as t:
            self.assertEqual(t.check()['rows'],20)
            self.assertEqual(t.row(4,4),bytes(range(248,256)))

    def test_wrong_hash(self):
        create(self.out);(self.out/'layer-3.u8').write_bytes(b'x')
        with self.assertRaisesRegex(ValueError,'sha256'):
            Trace(self.out)

    def test_truncated_binary_with_valid_hash(self):
        create(self.out);p=self.out/'layer-3.u8';p.write_bytes(p.read_bytes()[:-1]);self.rehash(p.name)
        with self.assertRaisesRegex(ValueError,'length'):
            Trace(self.out)

    def test_row_index_short(self):
        create(self.out);p=self.out/'rows.jsonl';p.write_text('\n'.join(p.read_text().splitlines()[:-4])+'\n');self.rehash(p.name)
        with Trace(self.out) as t,self.assertRaisesRegex(ValueError,'length'):
            t.check()

    def test_path_escape(self):
        create(self.out);m=json.loads((self.out/'manifest.json').read_text());m['data'][0]['file']='../escape'
        (self.out/'manifest.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'escapes'):
            Trace(self.out)

    def test_symlink_escape(self):
        create(self.out);p=self.out/'layer-3.u8';data=p.read_bytes();p.unlink()
        outside=self.root/'outside';outside.write_bytes(data);p.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'escapes'):
            Trace(self.out)

    def test_duplicate_ids(self):
        with self.assertRaises(ValueError): valid_ids([1]*8)

    def test_id_range(self):
        for row in ([0,1,2,3,4,5,6,256],[0,1,2,3,4,5,6,-1],[0,1,2,3,4,5,6,True]):
            with self.assertRaises(ValueError):valid_ids(row)

    def test_invalid_index_metadata(self):
        good=dict(row=0,step=0,request='a',position=0,padding=False,phase='decode',k=3,concurrency=1)
        for key,val in [('row',True),('step',-1),('position',None),('padding',0),('phase','unknown'),('k',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):valid_row(dict(good,**{key:val}),0)

    def test_writer_shape_abort(self):
        w=Writer(self.out,[3],'test',dict(status='assumed'),'decode-verify')
        with self.assertRaises(ValueError):w.append(0,[],{3:[]})
        w.abort();self.assertFalse(self.out.exists());self.assertEqual(list(self.root.iterdir()),[])

    def test_create_only(self):
        create(self.out)
        with self.assertRaisesRegex(ValueError,'exists'):create(self.out)

    def test_cli_small_output_is_atomic_idempotent_and_conflict_safe(self):
        output = self.root / 'presets.json'
        with contextlib.redirect_stdout(io.StringIO()):
            main(['presets', '--out', str(output)])
            main(['presets', '--out', str(output)])
        expected = output.read_bytes()
        output.write_bytes(b'different')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['presets', '--out', str(output)])
        self.assertEqual(output.read_bytes(), b'different')
        self.assertNotEqual(expected, b'different')

    def test_cli_small_output_recovers_from_prepublication_crash(self):
        output = self.root / 'presets.json'
        real = moereplay_cli.publish_bytes

        class InjectedCrash(BaseException):
            pass

        def crash(path, data, **kwargs):
            def fault(stage):
                if stage == 'full_before_publish':
                    raise InjectedCrash(stage)
            return real(path, data, _fault=fault, **kwargs)

        with mock.patch.object(moereplay_cli, 'publish_bytes', crash), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(InjectedCrash):
            main(['presets', '--out', str(output)])
        self.assertFalse(output.exists())
        self.assertEqual(len(list(self.root.glob('.presets.json.atomic-*'))), 1)
        with contextlib.redirect_stdout(io.StringIO()):
            main(['presets', '--out', str(output)])
        self.assertTrue(output.is_file())

    def test_min_steps_and_all_layers(self):
        create(self.out)
        with Trace(self.out) as t:
            with self.assertRaisesRegex(ValueError,'few steps'):t.check(min_steps=5)
            with self.assertRaisesRegex(ValueError,'layers'):t.check(range(3,78))

    def test_position_gap(self):
        create(self.out);p=self.out/'rows.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];rows[1]['position']=999
        p.write_text(''.join(json.dumps(x)+'\n' for x in rows));self.rehash(p.name)
        with Trace(self.out) as t,self.assertRaisesRegex(ValueError,'consecutive'):t.check()

    def test_group_count(self):
        create(self.out);p=self.out/'rows.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];rows[0]['request']='other'
        p.write_text(''.join(json.dumps(x)+'\n' for x in rows));self.rehash(p.name)
        with Trace(self.out) as t,self.assertRaisesRegex(ValueError,'request count'):t.check()

    def test_padding_metadata(self):
        create(self.out,padding=1);p=self.out/'rows.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];rows[4]['request']='a'
        p.write_text(''.join(json.dumps(x)+'\n' for x in rows));self.rehash(p.name)
        with Trace(self.out) as t,self.assertRaisesRegex(ValueError,'padding'):t.check()

    def test_duplicate_layers(self):
        with self.assertRaises(ValueError):Writer(self.out,[3,3],'x',dict(status='assumed'),'decode-verify')

    def test_bad_schema(self):
        create(self.out);p=self.out/'manifest.json';m=json.loads(p.read_text());m['schema']='moe-routes v2';p.write_text(json.dumps(m))
        with self.assertRaises(ValueError):Trace(self.out)


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'));self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.record=dict(schema=1,step=0,layers=[3,4],physical_rows=5,k=3,concurrency=1,accepted={'a':2},
            rows=[dict(request_id='a',position=i,padding=False) for i in range(4)]+[dict(padding=True)],
            experts=[[list(range(8)) for _ in range(5)] for _ in range(2)])

    def convert(self,records=None,gz=False):
        source=self.root/('in.jsonl.gz' if gz else 'in.jsonl')
        content=''.join(json.dumps(r)+'\n' for r in (records or [self.record]))
        if gz:
            with gzip.open(source,'wt') as f:f.write(content)
        else:source.write_text(content)
        return convert_schema1(source,self.root/'converted',status='assumed')

    def test_gzip_converter(self):
        self.convert(gz=True)
        with Trace(self.root/'converted') as t:self.assertEqual(t.check()['rows'],5)

    def test_missing_acceptance(self):
        del self.record['accepted']
        with self.assertRaises(ValueError):self.convert()
        self.assertFalse((self.root/'converted').exists())

    def test_acceptance_coverage(self):
        self.record['accepted']={}
        with self.assertRaisesRegex(ValueError,'cover every'):self.convert()

    def test_acceptance_bounds(self):
        self.record['accepted']={'a':4}
        with self.assertRaisesRegex(ValueError,'prefix'):self.convert()

    def test_accept_flags(self):
        del self.record['accepted'];self.record['accept_flags']={'a':[True,False,True]}
        self.convert()
        with Trace(self.root/'converted') as t:self.assertEqual(t.check()['rows'],5)

    def test_acceptance_disagreement(self):
        self.record['accept_flags']={'a':[True,False,True]}
        with self.assertRaisesRegex(ValueError,'disagrees'):self.convert()

    def test_missing_padding(self):
        del self.record['rows'][0]['padding']
        with self.assertRaises(ValueError):self.convert()

    def test_partial_layers(self):
        self.record['experts'].pop()
        with self.assertRaisesRegex(ValueError,'all layer'):self.convert()

    def test_accepted_only_capture(self):
        self.record['rows']=self.record['rows'][:2];self.record['physical_rows']=2
        self.record['experts']=[l[:2] for l in self.record['experts']]
        with self.assertRaisesRegex(ValueError,r'c\*'):self.convert()

    def test_step_order(self):
        with self.assertRaisesRegex(ValueError,'increase'):self.convert([self.record,self.record])

    def test_partial_json_tail(self):
        p=self.root/'in.jsonl';p.write_text(json.dumps(self.record)+'\n{')
        with self.assertRaises(ValueError):convert_schema1(p,self.root/'converted',status='assumed')
        self.assertFalse((self.root/'converted').exists())


class ModelTests(unittest.TestCase):
    def test_shape_bytes(self):
        self.assertEqual(BYTES_TP4,3*6144*512//2+3*6144*512//128*2)
        for n in (1,2,4):self.assertEqual(256//n*BYTES_TP4*n,1245708288)

    def test_contiguous_known(self):
        routes=[bytes((0,1,64,65,128,129,192,193))]*4
        for lay,ds in [('TP4',[8]*4),('EP2xTP2',[4]*4),('EP4',[2]*4)]:
            r=rank_work(routes,lay,placement({'TP4':1,'EP2xTP2':2,'EP4':4}[lay],'contiguous'))
            self.assertEqual(r['D_r'],ds);self.assertEqual(r['D_union'],8)
            self.assertEqual(r['bytes_r'],[8*BYTES_TP4]*4)

    def test_straggler_known(self):
        r=rank_work([bytes(range(8))],'EP4',placement(4,'contiguous'))
        self.assertEqual(r['D_r'],[8,0,0,0]);self.assertEqual(r['max_rank_D'],8)
        self.assertEqual(Costs().evaluate('EP4',4,r['D_r'])['layer_us'],12.5+21.46*4*8)

    def test_greedy_capacity_and_ties(self):
        p=placement(4,'greedy',Counter())
        self.assertEqual(Counter(p),{0:64,1:64,2:64,3:64});self.assertEqual(p,placement(4,'contiguous'))
        p=placement(4,'greedy',Counter({e:256-e for e in range(256)}))
        loads=[sum(256-e for e,r in enumerate(p) if r==g) for g in range(4)]
        self.assertLessEqual(max(loads)-min(loads),1)

    def test_replica_one_execution(self):
        r=rank_work([bytes(range(8))]*4,'EP4',placement(4,'contiguous'),range(8))
        self.assertEqual(sum(r['routed_slots_r']),32);self.assertEqual(r['D_r'],[2]*4)
        self.assertEqual(r['D_union'],8)

    def test_uniform_union_expectation(self):
        rng=random.Random(7)
        unions=[len(set(e for _ in range(16) for e in rng.sample(range(256),8))) for _ in range(500)]
        self.assertLess(abs(sum(unions)/len(unions)-256*(1-(1-8/256)**16)),1.)

    def test_bootstrap_paired_and_deterministic(self):
        a=bootstrap([1.,2.,3.,4.],[['a'],['a'],['b'],['b']],200)
        self.assertEqual(a,bootstrap([1.,2.,3.,4.],[['a'],['a'],['b'],['b']],200))
        self.assertEqual(a['mean'],2.5);self.assertNotEqual(a['prompts_ci95'],'unknown')

    def test_overlapping_prompt_ci_unknown(self):
        a=bootstrap([1,2,3],[['a','b'],['b','c'],['d']],200)
        self.assertEqual(a['prompts_ci95'],'unknown')

    def test_singleton_ci_unknown(self):
        a=bootstrap([1],[['a']],200);self.assertEqual(a['steps_ci95'],'unknown')

    def test_bench_fit_schema(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as d:
            p=Path(d)/'fit.json';p.write_text(json.dumps(dict(per_layout_M=[dict(layout='EP4',M=4,a_us=2,b_us_per_distinct_local_expert=3,points=[dict(D=1),dict(D=8)])])))
            cost=Costs(p);v=cost.evaluate('EP4',4,[2,4,6,8])
            self.assertEqual(v['layer_us'],26);self.assertFalse(v['extrapolation']);self.assertEqual(v['fit']['status'],'measured')
            self.assertTrue(cost.evaluate('EP4',4,[0]*4)['extrapolation'])
            self.assertTrue(cost.resolve('TP4',4)['missing_fit'])

    def test_negative_intercept_preserved_and_clipped(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as d:
            p=Path(d)/'fit.json';p.write_text(json.dumps(dict(per_layout_M=[dict(layout='EP4',M=4,a_us=-2,b_us_per_distinct_local_expert=3)])))
            v=Costs(p).evaluate('EP4',4,[0,1,2,8])
            self.assertEqual(v['fit']['a_us'],-2);self.assertEqual(v['rank_us'],[0,1,4,22])
            self.assertEqual(v['negative_prediction_clipped_r'],[True,False,False,False])
            p.write_text(json.dumps(dict(per_layout_M=[dict(layout='EP4',M=4,a_us=2,b_us_per_distinct_local_expert=-3)])))
            with self.assertRaises(ValueError):Costs(p)

    def test_invalid_fit(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as d:
            p=Path(d)/'fit.json'
            for coeff in (float('nan'),float('inf'),True):
                p.write_text(json.dumps(dict(per_layout_M=[dict(layout='EP4',M=4,a_us=coeff,b_us_per_distinct_local_expert=3)])))
                with self.assertRaises(ValueError):Costs(p)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'));self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.out=self.root/'trace'

    def test_decode_paired_layouts_and_padding(self):
        create(self.out,padding=1,layers=(3,4))
        details=self.root/'details.jsonl'
        with Trace(self.out) as t:r=replay(t,bootstrap_samples=200,replicate_h=2,details=details)
        key='observed/c1_M5_K3/TP4/contiguous'
        self.assertEqual(r['cycles'][key]['delta_vs_TP4_ms']['mean'],0)
        self.assertEqual(r['cycles'][key]['cycle_kernel_ms']['steps'],2)
        ds=[json.loads(x) for x in details.read_text().splitlines()]
        self.assertEqual({x['step'] for x in ds},{2,3})
        self.assertEqual(ds[0]['work']['D_union'],40)
        self.assertEqual(r['collective']['delta_ms'],0)
        self.assertEqual(len(r['cyclesim_what_if']),24)

    def test_train_frequency_excludes_test(self):
        create(self.out)
        with Trace(self.out) as t:
            f=train_frequency(t,3,{0})
            self.assertEqual(f[0],1);self.assertEqual(f[32],0)

    def test_union_never_crosses_verify_steps(self):
        create(self.out,requests=['same']*4)
        with Trace(self.out) as t:g=union_growth(t,3)
        self.assertEqual(g['4']['U'],32);self.assertEqual(g['5']['U'],'unknown')

    def test_union_growth_covers_full_corpus(self):
        w=Writer(self.out,[3],'test',dict(status='assumed'),'prefill-proxy')
        rows=[dict(request='p',position=i,padding=False,phase='prefill',k=0,concurrency=1) for i in range(5001)]
        w.append(0,rows,{3:[list(range(8))]*5001});w.finish()
        with Trace(self.out) as t:g=union_growth(t,3)
        self.assertEqual(g['1']['segments'],5001);self.assertEqual(g['4']['segments'],4998)
        self.assertEqual(g['4']['coverage'],'all eligible segments')

    def test_layer_extrapolation(self):
        create(self.out)
        with Trace(self.out) as t:
            a=replay(t,bootstrap_samples=0);b=replay(t,bootstrap_samples=0,extrapolate_layers=75)
        key='observed/c1_M4_K3/TP4/contiguous'
        self.assertAlmostEqual(b['cycles'][key]['cycle_kernel_ms']['mean'],75*a['cycles'][key]['cycle_kernel_ms']['mean'])
        self.assertEqual(b['layer_scale']['status'],'assumed')

    def test_no_tensor_dependency(self):
        self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules)

    def test_all_75_layer_sum(self):
        create(self.out,layers=range(3,78),steps=2)
        with Trace(self.out) as t:
            self.assertEqual(t.check(range(3,78))['layers'],list(range(3,78)))
            r=replay(t,bootstrap_samples=0)
        self.assertEqual(len(r['per_layer']),75)
        v=r['cycles']['observed/c1_M4_K3/TP4/contiguous']['cycle_kernel_ms']['mean']
        self.assertAlmostEqual(v,75*(12.5+21.46*32)/1000)

    def test_external_prompt_holdout(self):
        create(self.out,steps=1,requests=['test'])
        train_dir=self.root/'train';create(train_dir,steps=2,requests=['train-a','train-b'])
        with Trace(self.out) as t,Trace(train_dir) as train:
            r=replay(t,bootstrap_samples=0,train_trace=train)
        self.assertEqual(r['holdout']['training_steps'],0)
        self.assertEqual(r['placement_fits']['3']['training_rows'],8)
        self.assertEqual(r['cycles']['observed/c1_M4_K3/TP4/contiguous']['cycle_kernel_ms']['steps'],1)

    def test_external_request_overlap_refused(self):
        create(self.out,steps=1,requests=['overlap'])
        train_dir=self.root/'train';create(train_dir,steps=2,requests=['overlap','other'])
        with Trace(self.out) as t,Trace(train_dir) as train,self.assertRaisesRegex(ValueError,'disjoint'):
            replay(t,bootstrap_samples=0,train_trace=train)

    def test_known_prior_range_explicit(self):
        create(self.out)
        with Trace(self.out) as t:r=replay(t,bootstrap_samples=0)
        p=r['per_layer']['3']['observed/c1_M4_K3/TP4/contiguous']
        self.assertEqual(p['fit_extrapolation_fraction'],1.)
        self.assertEqual(p['unknown_calibration_range_fraction'],0.)

    def test_total_bank_and_replica_bytes(self):
        create(self.out,layers=(3,4))
        with Trace(self.out) as t:r=replay(t,bootstrap_samples=0,replicate_h=8)
        base=r['cycles']['observed/c1_M4_K3/EP4/contiguous']['bank_bytes_r']
        replica=r['cycles']['observed/c1_M4_K3/EP4/replicate-hottest']['bank_bytes_r']
        self.assertEqual(base,[2*1245708288]*4)
        self.assertEqual(sum(replica)-sum(base),2*8*3*BYTES_TP4*4)
        self.assertFalse(r['cycles']['observed/c1_M4_K3/EP4/contiguous']['serving_compatible'])

    def test_cli_decode_check_rejects_prefill(self):
        w=Writer(self.out,[3],'x',dict(status='assumed'),'prefill-proxy')
        rows=[dict(request='p',position=0,padding=False,phase='prefill',k=0,concurrency=1)]
        w.append(0,rows,{3:[list(range(8))]});w.finish()
        with self.assertRaises(SystemExit),contextlib.redirect_stderr(io.StringIO()):
            main(['--check',str(self.out),'--decode-only'])

    def test_single_step_needs_training(self):
        create(self.out,steps=1)
        with Trace(self.out) as t,self.assertRaisesRegex(ValueError,'>=2'):replay(t,bootstrap_samples=0)

    def test_identical_external_training_refused(self):
        create(self.out)
        with Trace(self.out) as t,self.assertRaisesRegex(ValueError,'differ'):replay(t,train_trace=t)

    def test_cli_check_alias(self):
        create(self.out);p=self.root/'check.json'
        main(['--check',str(self.out),'--out',str(p)])
        self.assertEqual(json.loads(p.read_text())['structural_check'],'PASS')
        main(['--check',str(self.out),'--out',str(p)])
        p.write_text('different')
        with self.assertRaises(SystemExit):main(['--check',str(self.out),'--out',str(p)])
        self.assertEqual(p.read_text(),'different')

    def test_prefill_prompt_holdout(self):
        w=Writer(self.out,[3],'test',dict(status='assumed'),'prefill-proxy')
        for p in range(4):
            rows=[dict(request='p%d'%p,position=i,padding=False,phase='prefill',k=0,concurrency=1) for i in range(8)]
            w.append(p,rows,{3:[list(range(p*8,p*8+8))]*8})
        w.finish()
        with Trace(self.out) as t:r=replay(t,samples=2,bootstrap_samples=0)
        self.assertEqual(r['holdout']['train'],['p0','p1']);self.assertEqual(r['holdout']['test'],['p2','p3'])
        self.assertEqual(len(r['cycles']),54)

