# SPDX-License-Identifier: Apache-2.0
import copy
import unittest
from abharness import logic as l

def t0_fixture():
    out=dict(basis='token_ids',sequence=[42],token_ids=[42],tokens=[dict(token_id=42,token='x',bytes=[120],logprob=-.5)],
        text='x',reasoning_text='',finish_reason='stop',usage=dict(completion_tokens=1),raw_response=dict(id='request-id'))
    return dict(status='COMPLETE',apc_mode='cold',panel_sha256='panel',model='example-model',
        configuration=dict(seed=1729),rows=[dict(row=i,category='prose',prompt='p'+str(i),nonce=str(i),gate_eligible=True,
            repeats=[dict(request_body='body'+str(i),output=copy.deepcopy(out),attempts=[dict(reset=dict(http_status=200),prefix_cache_hits=dict(delta=0))]) for _ in range(3)]) for i in range(1,31)])


def cycle_fixture(cycle=8.,tps=100.,committed=3.):
    return dict(rows=[dict(kind=k,i=i,cycle_ms=cycle,tps=tps,committed=committed,text_sha256=k+str(i)) for k in ('prose','code') for i in range(10)])


class DecisionTests(unittest.TestCase):
    def panels(self,cycle=8.,tps=100.,committed=3.):return [cycle_fixture(cycle,tps,committed) for _ in range(3)]
    def test_bracketed_per_prompt_not_old_a_or_panel_medians(self):
        a=self.panels(8);ap=self.panels(12);b=self.panels(9.7)
        result=l.decision(a,ap,b)
        self.assertEqual(result['decision'],'PASS');self.assertAlmostEqual(result['cycle_saving_ms'],.3)
        self.assertEqual(len(result['pairs']),60);self.assertEqual(result['groups']['prose']['n'],30)
        for pair in result['pairs']:
            self.assertAlmostEqual(pair['delta_ms'],-.3);self.assertEqual(pair['reference_cycle_ms'],10)
        # Reorder prompt rows and use heterogeneous values: match by kind+i, never list order.
        for j in range(3):
            for i,row in enumerate(a[j]['rows']):row['cycle_ms']=8+i
            for i,row in enumerate(ap[j]['rows']):row['cycle_ms']=12+i
            for i,row in enumerate(b[j]['rows']):row['cycle_ms']=9.7+i
            b[j]['rows'].reverse()
        self.assertAlmostEqual(l.decision(a,ap,b)['cycle_saving_ms'],.3)

    def test_exact_operator_thresholds_no_prose_tps_gate(self):
        a=self.panels();b=self.panels(7.8,tps=1,committed=2.985)
        for panel in b:
            for row in panel['rows']:
                if row['kind']=='code':row['cycle_ms']=8.04
        result=l.decision(a,a,b)
        self.assertEqual(result['decision'],'PASS');self.assertAlmostEqual(result['code_regression'],.005)
        self.assertAlmostEqual(result['committed_ratio'],.995)

    def test_each_new_numeric_threshold_kills(self):
        for fault in ('saving','code_cycle','committed','ci'):
            a=self.panels();b=self.panels(7.7)
            for panel in b:
                for row in panel['rows']:
                    if fault=='saving' and row['kind']=='prose':row['cycle_ms']=7.81
                    if fault=='code_cycle' and row['kind']=='code':row['cycle_ms']=8.041
                    if fault=='committed':row['committed']=2.984
                    if fault=='ci' and row['kind']=='prose':row['cycle_ms']=4 if row['i']<5 else 11.5
            result=l.decision(a,a,b)
            with self.subTest(fault=fault):self.assertEqual(result['decision'],'KILL')
            if fault=='ci':self.assertGreater(result['cycle_saving_ms'],.2);self.assertLessEqual(result['ci95_ms'][0],0)

    def test_text_committed_equality_reported_and_drift_not_extra_gate(self):
        a=self.panels(8);ap=self.panels(12);b=self.panels(9.7,committed=3.1)
        b[0]['rows'][0]['text_sha256']='changed'
        result=l.decision(a,ap,b)
        self.assertEqual(result['decision'],'PASS');self.assertEqual(result['groups']['prose']['text_equal'],29)
        self.assertEqual(result['groups']['prose']['committed_equal'],0)

    def test_missing_panels_missing_prompts_nan_fail_closed(self):
        a=self.panels()
        for bad in ([],a[:2],a+[a[0]],self.panels(float('nan')),self.panels(-1),[dict(rows=[])]*3):
            with self.assertRaises(ValueError):l.decision(a,a,bad)

    def test_bootstrap_reproducible_separate_prose_code_and_three_panels_pooled(self):
        a=self.panels();b=self.panels(7.7)
        for row in b[2]['rows']:
            if row['kind']=='prose':row['cycle_ms']=7.4
        result=l.decision(a,a,b)
        self.assertEqual(result,l.decision(a,a,b));self.assertAlmostEqual(result['cycle_saving_ms'],.4)
        self.assertGreater(result['groups']['prose']['ci95_ms'][0],0)
        self.assertAlmostEqual(result['groups']['code']['cycle_saving_ms'],.3)



def reference(a):
    return dict(rows=[dict(row=r['row'], repeat=i, seq=[dict(tok=42, top=[[42,-.1],[43,-.2],[44,-.3],[45,-.4],[46,-.5]])])
                      for r in a['rows'] for i in (1,2)])


class NearTieTests(unittest.TestCase):
    def test_first_divergence_valid_top_five(self):
        a=t0_fixture();b=copy.deepcopy(a)
        for repeat in b['rows'][0]['repeats']:
            repeat['output']['token_ids']=[42,43]
        for repeat in a['rows'][0]['repeats']:
            repeat['output']['token_ids']=[42,42]
        ref=reference(a)
        for row in ref['rows']:
            if row['row']==1:row['seq'].append(copy.deepcopy(row['seq'][0]))
        result=l.compare_t0(a,b,'draft',ref)
        self.assertEqual(result['decision'],'PASS_NEAR_TIE')
        self.assertAlmostEqual(result['rows'][0]['margin_nats'],.1)
        for row in ref['rows']:
            if row['row']==1:row['seq'][1]['top'][1][1]=-.7
        self.assertEqual(l.compare_t0(a,b,'draft',ref)['decision'],'HOLD')

    def test_first_token_instability_missing_reference_and_cold_evidence(self):
        a=t0_fixture();b=copy.deepcopy(a)
        self.assertEqual(l.compare_t0(a,b,'draft',reference(a))['decision'],'PASS')
        b['rows'][0]['repeats'][0]['output']['token_ids']=[43]
        self.assertEqual(l.compare_t0(a,b,'draft',reference(a))['decision'],'HOLD')
        self.assertEqual(l.compare_t0(a,a,'draft',None)['decision'],'HOLD')
        a['rows'][0]['repeats'][0]['attempts'][0]['prefix_cache_hits']['delta']=1
        with self.assertRaises(ValueError):l.compare_t0(a,a)

    def test_native_ids_and_request_drift(self):
        with self.assertRaises(ValueError):l.native_ids({'basis':'text_characters'})
        self.assertEqual(l.native_ids({'tokens':[{'token':'token_id:42'}]}),[42])
        a=t0_fixture();b=copy.deepcopy(a);b['rows'][0]['repeats'][0]['request_body']='changed'
        with self.assertRaises(ValueError):l.compare_t0(a,b)
