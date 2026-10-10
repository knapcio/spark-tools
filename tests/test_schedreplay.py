# SPDX-License-Identifier: Apache-2.0
import io
import gzip
import json
from pathlib import Path
import random
import tempfile
import unittest
from lab.schedreplay.native_fixture import Pause, request, scheduler
from lab.schedreplay.policy import Fast, Policy, make_scheduler
from lab.schedreplay.replay import BLOCK, POOL, Replay as BaseReplay, demand, pareto, percentiles
from lab.schedreplay.timing import Timing as BaseTiming
from lab.schedreplay.workloads import Work, load, synthetic, write

ROOT = Path(__file__).resolve().parents[1]

def Timing(calibration=None, *args, **kwargs):
    calibration = calibration or json.loads((ROOT / 'configs/scheduler-timing.json').read_text())
    return BaseTiming(calibration, *args, **kwargs)

def Replay(*args, **kwargs):
    kwargs.setdefault('timing', Timing(json.loads((ROOT / 'configs/scheduler-timing.json').read_text())))
    return BaseReplay(*args, **kwargs)

def work(rid='r',arrival=0,prompt=1000,output=16,maximum=None,**extra):
    return Work.parse(dict(request_id=rid,arrival_s=arrival,prompt_tokens=prompt,
                           max_tokens=maximum or output,output_tokens=output,**extra))


def continuous(s,width=4):
    for r in s.running:
        if r.num_computed_tokens>=r.num_prompt_tokens:
            r.num_tokens=r.num_computed_tokens+1
            r.num_tokens_with_spec=r.num_computed_tokens+width
            r.spec_token_ids=list(range(width-1))


class Inputs(unittest.TestCase):
    def test_distribution_seed_and_bounds(self):
        row=work().row();row['max_tokens']=100
        row['output_tokens']={'distribution':'uniform','low':20,'high':80}
        self.assertEqual(Work.parse(row,rng=random.Random(1)),Work.parse(row,rng=random.Random(1)))
        row['output_tokens']={'distribution':'choice','values':[1,2,100]}
        self.assertIn(Work.parse(row).output_tokens,[1,2,100])
        for output in (0,101,True,{'distribution':'uniform','low':4,'high':2}):
            with self.subTest(output=output),self.assertRaises(ValueError):Work.parse({**row,'output_tokens':output})

    def test_content_and_invalid_fields_rejected(self):
        for change in [{'content':'never allowed'},{'prompt':'never allowed'},{'prompt_tokens':True},
                       {'arrival_s':float('nan')},{'cached_prefix_tokens':1001},
                       {'max_tokens':262144},{'thinking':1},{'cancel_at_s':-1},
                       {'evidence':{'content':'forbidden'}},{'observed':{'response':'forbidden'}}]:
            with self.subTest(change=change),self.assertRaises(ValueError):Work.parse({**work().row(),**change})

    def test_jsonl_gzip_roundtrip_dependencies_duplicates(self):
        works,meta=synthetic('agents',sessions=2,rounds=2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tiny.jsonl';write(path,works,meta)
            reread,header=load(path)
            self.assertEqual(works,reread);self.assertEqual(header,meta)
            zipped=path.with_suffix('.gz')
            with gzip.open(zipped,'wt') as stream:stream.write(path.read_text())
            self.assertEqual(load(zipped)[0],works)
            with self.assertRaisesRegex(ValueError,'bounded'):load(path,max_records=1)
            path.write_text(json.dumps(work().row())+'\n'+json.dumps(work().row()))
            with self.assertRaisesRegex(ValueError,'duplicate'):load(path)
            path.write_text(json.dumps(work(after_request_id='unknown').row()))
            with self.assertRaisesRegex(ValueError,'earlier'):load(path)

    def test_bounded_lines_and_metadata_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'large.jsonl'
            path.write_text('x'*1048577)
            with self.assertRaisesRegex(ValueError,'1 MiB'):load(path)
            with self.assertRaisesRegex(ValueError,'evidence'):
                Work.parse({**work().row(),'evidence':{'source':'x'*257}})
            path.write_text(json.dumps({'format':'sched-workload v1','notes':{'content':'forbidden'}}))
            with self.assertRaisesRegex(ValueError,'header notes enum'):load(path)


class FastPolicy(unittest.TestCase):
    def test_invalid_policy_and_parameter(self):
        for value in ('fair:99:40','fair:512:-1','bogus','fair:4096:bad'):
            with self.assertRaises(ValueError):Policy.parse(value)
        with self.assertRaises(ValueError):Policy(variant='invalid')
        with self.assertRaises(ValueError):Policy(prefill_cap=4096)

    def test_audit_tail_only_newcomer_and_n40(self):
        for decoder in (False,True):
            s=make_scheduler();s.waiting.append(request('A',128000))
            if decoder:s.running.append(request('C',300,300,decode=True))
            stepper=Fast(Policy());chunks=[];first=None
            for i in range(1272 if decoder else 32):
                continuous(s,width=3)
                if i==1:s.waiting.append(request('B',300))
                emitted=stepper.schedule(s)
                if 'A' in emitted:chunks.append(emitted['A'])
                if 'B' in emitted and first is None:first=i
            self.assertEqual(chunks,([4093]*31+[1117]) if decoder else ([4096]*31+[1024]))
            self.assertEqual(first,1271 if decoder else 31)

    def test_reserve_short_newcomer_and_cap_progress(self):
        for variant in ('reserve','cap'):
            s=make_scheduler();s.waiting.append(request('A',128000))
            f=Fast(Policy(off=True,variant=variant))
            self.assertEqual(f.schedule(s),{'A':4096})
            s.waiting.append(request('B',300));emitted=f.schedule(s)
            self.assertEqual(emitted['B'],300)
            self.assertLess(emitted['A'],4096)

    def test_reserve_newcomer_uses_aggregate_prefill_cap_in_every_chunk(self):
        for chunk in (512,1024,2048,4096):
            s=make_scheduler();s.running.append(request('C',300,300,decode=True));s.waiting.append(request('A',128000))
            f=Fast(Policy(chunk,0,variant='reserve'))
            f.schedule(s);continuous(s,width=3);s.waiting.append(request('B',300))
            emitted=f.schedule(s)
            self.assertEqual(emitted['B'],300)
            self.assertLessEqual(emitted['A']+emitted['B'],chunk)

    def test_shortest_reorders_waiting_only(self):
        s=make_scheduler();s.waiting.extend([request('long',128000),request('short',300)])
        emitted=Fast(Policy(off=True,variant='shortest')).schedule(s)
        self.assertEqual(list(emitted)[0],'short')
        s=make_scheduler();s.running.append(request('long',128000,4096));s.waiting.append(request('short',300))
        self.assertNotIn('short',Fast(Policy(off=True,variant='shortest')).schedule(s))

    def test_pause_and_native_eligibility(self):
        s=make_scheduler();s.running.append(request('d',300,300,decode=True));s.waiting.append(request('p',30000))
        s.running[0].next_decode_eligible_step=99
        self.assertNotIn('d',Fast(Policy()).schedule(s))
        s._pause_state=Pause.PAUSED_ALL
        self.assertEqual(Fast(Policy()).schedule(s),{})

    def test_full_cached_prompt_recomputes_last_token(self):
        result=Replay(Policy()).run([work(prompt=1000,cached_prefix_tokens=1000)])
        self.assertEqual(result['requests'][0]['prefill_tokens'],1)

    def test_fast_kv_waiting_refusal(self):
        s=make_scheduler();s.waiting.append(request('a',1000))
        s.kv_cache_manager.allocate_slots=lambda *a,**kw:None
        self.assertEqual(Fast(Policy()).schedule(s),{})
        self.assertEqual(len(s.waiting),1)


class TimingAndReplay(unittest.TestCase):
    def test_cold_fit_is_additive_and_apc_marginal(self):
        t=Timing()
        for total in (8192,16154,127814,249837):
            cap=4096 if total>=16384 else 2048
            chunks=[(a,min(total,a+cap)) for a in range(0,total,cap)]
            self.assertAlmostEqual(t.prefill(chunks,[cap]*len(chunks)),t.cumulative(total))
        self.assertLess(t.prefill([(120000,128000)]),t.cumulative(128000))

    def test_invalid_cycle_table_rejected(self):
        for rows in ([],[{'M':4,'ms':float('nan'),'evidence':'measured'}],
                     [{'M':4,'ms':10,'evidence':'fabricated'}]):
            with self.assertRaises(ValueError):Timing(cycle_table={'prose':rows,'code':rows})

    def test_width_rows_and_context_cost(self):
        t=Timing()
        self.assertGreater(t.decode(4,4,'code',0),t.decode(1,4,'code',0))
        self.assertGreater(t.decode(1,4,'prose',60043),t.decode(1,4,'prose',0))
        cost,label=t.cost([(0,4096)],[('prose',60043)],4)
        self.assertIn('measured mixed-gap',label);self.assertGreater(cost,5)

    def test_two_deep_commit_before_first_completion(self):
        trace=io.StringIO()
        Replay(Policy(off=True),trace=trace).run([work(prompt=128000)])
        jobs=[json.loads(x) for x in trace.getvalue().splitlines()]
        self.assertEqual(jobs[0]['scheduled_at_s'],0)
        self.assertEqual(jobs[1]['scheduled_at_s'],0)
        self.assertEqual(jobs[1]['start'],jobs[0]['end'])
        self.assertEqual(jobs[2]['scheduled_at_s'],jobs[0]['end'])

    def test_apc_reduces_prefill_but_not_block_reservation(self):
        cold=work(prompt=100000);warm=work(prompt=100000,cached_prefix_tokens=99000)
        self.assertEqual(demand(cold),demand(warm))
        a=Replay(Policy()).run([cold]);b=Replay(Policy()).run([warm])
        self.assertGreater(a['requests'][0]['ttft_s'],b['requests'][0]['ttft_s'])
        self.assertEqual(b['requests'][0]['prefill_tokens'],1000)

    def test_block_reservations_ceil_prompt_and_output_separately(self):
        w=work(prompt=65,output=65)
        self.assertEqual(demand(w),256)

    def test_fifo_head_of_line_no_short_bypass(self):
        works=[work('A',prompt=140000,output=256),work('B',arrival=1,prompt=130000,output=16),
               work('C',arrival=2,prompt=10,output=2)]
        result=Replay(Policy(),admission=True).run(works)
        a,b,c=result['requests']
        self.assertGreaterEqual(b['admitted_s'],a['end_s'])
        self.assertEqual(b['admitted_s'],c['admitted_s'])
        self.assertLessEqual(result['peak_reserved_tokens'],POOL)

    def test_admission_engine_floor_and_540_deadline(self):
        result=Replay(Policy(),admission=True,engine_floor_tokens=POOL).run([work(prompt=10)])
        row=result['requests'][0]
        self.assertEqual(row['status'],'max-wait');self.assertEqual(row['queue_wait_s'],540)
        self.assertIsNone(row['ttft_s'])

    def test_fifo_admission_arrival_during_engine_block_is_immediate(self):
        result=Replay(Policy(off=True),admission=True).run([work('A',prompt=128000),work('B',arrival=1,prompt=300)])
        b=result['requests'][1]
        self.assertEqual(b['admitted_s'],1)
        self.assertEqual(b['queue_wait_s'],0)
        self.assertGreater(b['engine_queue_wait_s'],0)

    def test_equal_time_fifo_follows_input_order(self):
        result=Replay(Policy(),admission=True).run([work('z-first',prompt=140000),work('a-second',prompt=130000)])
        first,second=result['requests']
        self.assertEqual(first['admitted_s'],0)
        self.assertGreaterEqual(second['admitted_s'],first['end_s'])

    def test_queue_capacity_64_and_deadlines(self):
        works=[work('r%d'%i,prompt=10) for i in range(66)]
        result=Replay(Policy(),admission=True,engine_floor_tokens=POOL).run(works)
        self.assertEqual(sum(r['status']=='queue-full' for r in result['requests']),2)
        self.assertEqual(sum(r['status']=='max-wait' for r in result['requests']),64)

    def test_serving_c2_k2_reuses_six_rows(self):
        t=Timing()
        expected=next(x['ms'] for x in t.cycles['code'] if x['M']==6)/1000
        self.assertAlmostEqual(t.decode(2,4,'code',0),expected)

    def test_cancel_before_prefill_and_during_blocks(self):
        for cancel in (0,1):
            result=Replay(Policy(),admission=True).run([work(prompt=128000,cancel_at_s=cancel)])
            row=result['requests'][0]
            self.assertEqual(row['status'],'cancelled');self.assertIsNone(row['first_s'])
            self.assertEqual(row['end_s'],cancel)
            self.assertEqual(row['output_tokens'],0)

    def test_cancel_drain_hold_blocks_new_admission_until_device_idle(self):
        works=[work('A',prompt=140000,cancel_at_s=1),work('B',arrival=1.2,prompt=130000)]
        result=Replay(Policy(),admission=True).run(works)
        a,b=result['requests']
        self.assertGreater(b['admitted_s'],1.2)
        self.assertEqual(a['status'],'cancelled');self.assertEqual(b['status'],'completed')

    def test_sessions_follow_actual_completion_plus_think(self):
        works,meta=synthetic('agents',sessions=1,rounds=2,think_s=8)
        result=Replay(Policy()).run(works,meta)
        a,b=result['requests'];self.assertAlmostEqual(b['arrival_s'],a['end_s']+8)
        self.assertGreater(b['prompt_tokens'],a['prompt_tokens'])

    def test_max_output_near_context_cap_makes_progress(self):
        result=Replay(Policy()).run([work(prompt=262100,output=44,cached_prefix_tokens=262099)])
        row=result['requests'][0];self.assertEqual(row['status'],'completed');self.assertEqual(row['output_tokens'],44)

    def test_stall_and_ttft_percentiles_have_no_first_counts(self):
        result=Replay(Policy()).run([work('cancel',cancel_at_s=0),work('finish')])
        self.assertEqual(result['no_first'],1)
        self.assertEqual(result['ttft_by_prompt_bucket']['0-8K']['submitted'],2)
        self.assertEqual(result['ttft_by_prompt_bucket']['0-8K']['n'],1)
        self.assertGreater(result['max_decode_stall_s'],0)
        self.assertEqual(percentiles([]),{'p50':None,'p90':None,'p99':None})

    def test_bound_failure_no_partial_result(self):
        with self.assertRaisesRegex(ValueError,'step bound'):Replay(Policy(),max_steps=1).run([work()])
        with self.assertRaisesRegex(ValueError,'time bound'):Replay(Policy(),max_time_s=.01).run([work()])

    def test_max_seqs_six_explicit_memory_gate(self):
        result=Replay(Policy(),max_seqs=6).run([work()])
        self.assertIn('unqualified',result['memory_gate'])

    def test_pareto_failures_not_artificially_fast(self):
        good=Replay(Policy(),admission=True).run([work()])
        failed=Replay(Policy(),admission=True,engine_floor_tokens=POOL).run([work()])
        rows=pareto([good,failed],good)
        self.assertNotEqual(rows[1]['vs_current'],'dominates')
