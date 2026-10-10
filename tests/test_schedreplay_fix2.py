# SPDX-License-Identifier: Apache-2.0
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from lab.schedreplay.policy import Policy
from lab.schedreplay.replay import pareto
from test_schedreplay import Timing, Replay
from lab.schedreplay.workloads import synthetic


class Fix2(unittest.TestCase):


    def test_explicit_table_requires_contract_and_acceptance(self):
        table=copy.deepcopy(Timing().cycles)
        for mutation in ({'k':3},{'M':12},{'committed_per_cycle':4},{'capture_layout':'pad'},
                         {'survival':[True,.2]},{'c':True}):
            modified=copy.deepcopy(table);modified['prose'][1].update(mutation)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):Timing(cycle_table=modified)
        modified=copy.deepcopy(table)
        for kind in modified:
            for p in modified[kind]:
                for key in ('c','k','policy','capture_layout','survival','committed_per_cycle'):p.pop(key)
        with self.assertRaisesRegex(ValueError,'contract'):Timing(cycle_table=modified)

    def test_unsupported_c3_never_claims_support(self):
        t=Timing()
        t.decode(3,4,'prose',0)
        self.assertTrue(any('c3' in x for x in t.unsupported))
        self.assertEqual(t.execution_width(3),3)

    def test_replay_schedules_k2_and_uses_per_job_acceptance(self):
        works,_=synthetic('long-during-decode')
        # Two already decoding streams; no prefill approximation in this test.
        a=works[0];b=copy.deepcopy(a)
        b.request_id='other';a.output_tokens=b.output_tokens=40
        trace=io.StringIO()
        table=copy.deepcopy(Timing().cycles)
        for rows in table.values():
            for point in rows:
                point['survival']=[.6,.3,.1][:point['k']]
                point['committed_per_cycle']=1+sum(point['survival'])
                point['ms']={1:70,2:94.06,4:142.05}[point['c']]
        replay=Replay(Policy(off=True),timing=Timing(cycle_table=table),trace=trace)
        result=replay.run([a,b],already_decoding=(a.request_id,b.request_id))
        jobs=[json.loads(line) for line in trace.getvalue().splitlines()]
        pairs=[j for j in jobs if j['decode_concurrency']==2]
        self.assertGreater(len(pairs),10)
        for job in pairs:
            self.assertEqual(job['effective_decode_width'],3)
            self.assertEqual(set(job['scheduled_tokens'].values()),{3})
        # First c2 completion has 1.9 expected outputs, rather than K3's 2.0.
        self.assertEqual(replay.states[a.request_id].events[1][1],1)
        self.assertEqual(result['completed'],2)

    def test_supported_frontier_not_poisoned_by_unrelated_case(self):
        good=Replay(Policy()).run(synthetic('long-during-decode')[0])
        good['timing_supported']=True;good['unsupported_timing_regimes']=[]
        bad=copy.deepcopy(good);bad['policy']='unsupported';bad['timing_supported']=False
        bad['unsupported_timing_regimes']=['joint prefills missing timing']
        rows=pareto([good,bad],good)
        self.assertTrue(rows[0]['pareto'])
        self.assertIsNone(rows[1]['pareto'])
        self.assertTrue(any('joint prefills' in x for x in rows[1]['ranking_withheld_reasons']))

    def test_metric_comparison_ignores_other_missing_objective(self):
        from lab.schedreplay.comparisons import compare
        base=Replay(Policy()).run(synthetic('long-during-decode')[0])
        base['timing_supported']=True
        r=copy.deepcopy(base);r['ttft_s']['p90']*=.9;r['decode_during_prefills_tok_s']=None
        rows=pareto([r],base)
        self.assertEqual(rows[0]['vs_current'],'indeterminate')
        self.assertIn('missing objective: decode_during_prefills_tok_s',rows[0]['ranking_withheld_reasons'])
        self.assertEqual(rows[0]['metric_comparisons']['ttft_p90_s']['vs_current'],'dominates')
        base['timing_supported']=r['timing_supported']=False
        base['metric_support']=r['metric_support']={'ttft_p90_s':[]}
        self.assertEqual(compare([r],base,['ttft_p90_s'],'explicit TTFT-only scope')[0]['vs_current'],'dominates')

    def test_conditional_subset_order_exclusions_and_no_startup(self):
        from lab.schedreplay.subsets import supported_subset
        works,meta=synthetic('long-during-decode')
        result=supported_subset(works,meta,Timing)
        self.assertEqual(result['status'],'estimated-supported')
        ranking=result['ranking']
        self.assertEqual([r['policy'] for r in ranking],['fair:4096:%d'%n for n in (0,10,20,40,80)])
        self.assertEqual([r['pareto'] for r in ranking],[True,False,False,False,False])
        self.assertEqual([r['vs_current'] for r in ranking],['dominates','dominates','dominates','equal','dominated'])
        stalls=[r['values']['A_event_gap_stall_s'] for r in ranking]
        self.assertLess(max(stalls)-min(stalls),1e-8)
        self.assertAlmostEqual(stalls[0],.53+.00115234375*(4096-4))
        self.assertTrue(any('APC startup' in x for x in result['exclusions']))
        self.assertTrue(any('isolated prefill lower bound' in x for x in result['exclusions']))
        self.assertTrue(all(r['values']['B_TTFT_s'] is not None for r in ranking))
        self.assertGreater(ranking[-1]['decode_throughput_excluded_tok_s'],ranking[0]['decode_throughput_excluded_tok_s'])
        self.assertGreaterEqual(len(result['capture_requests']),7)

    def test_subset_unsupported_inputs_have_exact_reasons(self):
        from lab.schedreplay.subsets import supported_subset
        works,meta=synthetic('newcomer')
        result=supported_subset(works,meta,Timing)
        self.assertEqual(result['status'],'indeterminate');self.assertEqual(result['ranking'],[])
        self.assertTrue(any('cold ~100K' in x for x in result['blocking_reasons']))
        works,meta=synthetic('long-during-decode')
        result=supported_subset(works,meta,Timing,floor=1)
        self.assertIn('external engine KV floor is outside declared supported settings',result['blocking_reasons'])
        works[0].output_tokens=2
        result=supported_subset(works,meta,Timing)
        self.assertEqual(result['status'],'indeterminate')
        self.assertTrue(all(r['pareto'] is None for r in result['ranking']))
        self.assertTrue(any('overlap interval missing' in x for x in result['blocking_reasons']))

    def test_supported_frontier_survives_unsupported_baseline(self):
        from lab.schedreplay.subsets import supported_subset
        works,meta=synthetic('long-during-decode')
        works[0].output_tokens=1000
        result=supported_subset(works,meta,Timing)
        ranking=result['ranking']
        rows={r['policy']:r for r in ranking}
        self.assertEqual(result['status'],'indeterminate')
        self.assertTrue(all(r['vs_current']=='indeterminate' for r in ranking))
        self.assertTrue(all(r['frontier_supported_members']==2 for r in ranking))
        self.assertTrue(all(r['frontier_excluded_members']==3 for r in ranking))
        self.assertTrue(all(any('overlap interval missing' in reason
                                for reason in r['ranking_withheld_reasons']) for r in ranking))
        self.assertIs(rows['fair:4096:0']['pareto'],True)
        self.assertIs(rows['fair:4096:10']['pareto'],False)
        for n in (20,40,80):
            self.assertIsNone(rows['fair:4096:%d'%n]['pareto'])

    def test_no_decode_gap_observation_stays_unknown(self):
        from test_schedreplay import work
        result=Replay(Policy()).run([work(output=1)])
        self.assertIsNone(result['max_decode_stall_s'])
        self.assertIsNone(result['requests'][0]['max_decode_stall_s'])
        self.assertTrue(any('missing objective: max_decode_stall_s' in reason
                            for reason in pareto([result],result)[0]['ranking_withheld_reasons']))


if __name__=='__main__':unittest.main()
