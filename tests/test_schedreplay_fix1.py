# SPDX-License-Identifier: Apache-2.0
import copy
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from lab.schedreplay.policy import Fast, Policy, make_scheduler, request
from lab.schedreplay.replay import pareto
from test_schedreplay import Timing, Replay
from lab.schedreplay.workloads import Work, write
from test_schedreplay import work

class Fix1(unittest.TestCase):
    def test_chunk_regimes_and_unsupported_ranking(self):
        t=Timing()
        for start in (8192,60000,95000):
            a,_=t.cost([(start,start+4096)],[],4)
            b,_=t.cost([(start,start+4096)],[('prose',60043)],4)
            self.assertLessEqual(a,b)
        r=Replay(Policy(chunk=512,decode_steps=0)).run([
            work('A',prompt=60043,output=1000,cached_prefix_tokens=60042),
            work('B',arrival=1,prompt=30000)])
        self.assertFalse(r['timing_supported'])
        self.assertIsNone(pareto([r],r)[0]['pareto'])

    def test_k3_fixture(self):
        self.assertEqual(make_scheduler().num_spec_tokens,3)


    def test_submitted_waiting_cancellation_holds_reservation(self):
        r=Replay(Policy(off=True),admission=True)
        result=r.run([work('busy',prompt=140000,output=2),
                      work('cancel',arrival=.1,prompt=120000,cancel_at_s=1),
                      work('next',arrival=1.1,prompt=120000)])
        busy,cancel,nxt=result['requests']
        self.assertIsNone(cancel['started_s'])
        self.assertEqual(nxt['admitted_s'],16)
        self.assertEqual(cancel['submitted_s'],.1)
        self.assertGreater(busy['end_s'],16)
        idle=Replay(Policy(),admission=True).run([
            work('cancel',prompt=140000,cancel_at_s=.1),work('next',arrival=.2,prompt=130000)])
        self.assertGreater(idle['requests'][1]['admitted_s'],.2)
        self.assertLess(idle['requests'][1]['admitted_s'],15.1)

    def test_unknown_metrics_indeterminate(self):
        a=Replay(Policy()).run([work()]);b=copy.deepcopy(a)
        a['decode_during_prefills_tok_s']=None;b['decode_during_prefills_tok_s']=5
        for base in (a,b):
            for row in pareto([b if base is a else a],base):
                self.assertEqual(row['vs_current'],'indeterminate')
                # Only the member's missing objective withholds its frontier membership.
                self.assertIs(row['pareto'],True if base is a else None)


    def test_async_short_output_no_tail(self):
        for output in (1,2,3):
            trace=io.StringIO()
            r=Replay(Policy(off=True),trace=trace)
            result=r.run([work(prompt=100,output=output)])
            row=result['requests'][0]
            if output <= 2:self.assertEqual(result['duration_s'],row['end_s'])
            jobs=[json.loads(x) for x in trace.getvalue().splitlines()]
            if output==1:self.assertEqual(len(jobs),1)
            self.assertEqual(r.states[work().request_id].row.num_output_placeholders,0)


    def test_k3_only_cost_and_acceptance(self):
        for width in (1,2,3,5):
            with self.assertRaises(ValueError):Replay(Policy(),width=width)
            with self.assertRaises(ValueError):Timing().decode(1,width,'prose',0)

if __name__=='__main__':unittest.main()
