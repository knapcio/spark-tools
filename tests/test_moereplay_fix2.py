# SPDX-License-Identifier: Apache-2.0
"""Serving-width proxies, shape exports and frozen legacy isolation."""
import contextlib
import io
from pathlib import Path
import json

import test_moereplay_fix1
from test_moereplay import create
from moereplay.replay import prefill_events, replay, proxy_widths, parse_proxy_widths
from moereplay.regression import reproduce, compare
from moereplay.cli import run_cli as main
from moereplay.routes import Trace, Writer


class ServingWidthTests(test_moereplay_fix1.ScratchTests):
    def trace(self):
        out=self.root/'prefill'
        w=Writer(out,[3],'test',dict(status='assumed'),'prefill-proxy')
        for step in range(4):
            rows=[dict(request='p%d'%step,position=i,padding=False,phase='prefill',k=0,concurrency=1) for i in range(8)]
            w.append(step,rows,{3:[list(range(i*8,i*8+8)) for i in range(8)]})
        w.finish()
        return Trace(out)

    def test_default_shapes_and_route_counts(self):
        with self.trace() as t:
            events=list(prefill_events(t,3,['p2','p3'],10,1234))
        self.assertEqual({(e['c'],e['M'],e['k']) for e in events},{(1,4,3),(2,6,2),(4,12,2)})
        for e in events:
            self.assertEqual(len(e['routes']),e['M'])
            width=e['k']+1
            for start in range(0,e['M'],width):
                first=e['routes'][start][0]
                self.assertEqual([r[0] for r in e['routes'][start:start+width]],list(range(first,first+8*width,8)))

    def test_partial_override_and_legacy_widths(self):
        self.assertEqual(parse_proxy_widths('2=4'),{1:4,2:4,4:3})
        with self.trace() as t:
            events=list(prefill_events(t,3,['p2','p3'],3,1234,{2:4,4:4}))
        self.assertEqual({(e['c'],e['M'],e['k']) for e in events},{(1,4,3),(2,8,3),(4,16,3)})

    def test_invalid_width_contract(self):
        for value in ('','1=4,1=3','3=4','2=0','4=17','2=3.0','2','1=4,'):
            with self.subTest(value=value), self.assertRaises(ValueError):parse_proxy_widths(value)
        for mapping in ({True:4},{2:True},{2:3.0},[],{2:1}):
            with self.subTest(mapping=mapping),self.assertRaises(ValueError):proxy_widths(mapping)

    def test_exports_match_all_cells_and_stay_ineligible(self):
        with self.trace() as t:r=replay(t,samples=3,bootstrap_samples=0)
        self.assertEqual(r['proxy_rows_per_request'],{'1':4,'2':3,'4':3})
        expected={1:(4,3),2:(6,2),4:(12,2)}
        for e in r['cyclesim_what_if']:
            s=e['required_shape'];m,k=expected[s['concurrency']]
            self.assertEqual((s['physical_rows'],s['K']),(m,k))
            self.assertIn(e['cell'],['c1_M4_K3','c2_M6_K2','c4_M12_K2'])
            self.assertEqual((e['argv'],e['kernel_ms'],e['calibration_eligible']),([],{},False))

    def test_three_row_segments_keep_boundaries_and_last(self):
        from moereplay.replay import eligible_segments
        out=self.root/'boundaries'
        w=Writer(out,[3],'test',dict(status='assumed'),'prefill-proxy')
        segments=[[('p',0),('p',1)],[('p',2),('p',3),('q',0),('q',1)],
                  [('p',4),('p',5),('p',9),('p',10)], [('q',2),('q',3),('q',4),('q',5)]]
        for step,segment in enumerate(segments):
            rows=[dict(request=req,position=pos,padding=False,phase='prefill',k=0,concurrency=1) for req,pos in segment]
            w.append(step,rows,{3:[list(range(8))]*len(rows)})
        w.finish()
        with Trace(out) as t:
            self.assertEqual(eligible_segments(t,['p','q'],3),{'q':[(10,11,12),(11,12,13)]})

    def test_cli_invalid_override_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as exit:
            main(['replay','missing','--rows-per-request','2=0'])
        self.assertEqual(exit.exception.code,2)

    def test_cli_default_and_override(self):
        with self.trace() as t:
            for suffix in ([],['--rows-per-request','2=4,4=4']):
                with contextlib.redirect_stdout(io.StringIO()):
                    r=main(['replay',str(t.root),'--samples','2','--bootstrap','0']+suffix)
                self.assertEqual(r['proxy_rows_per_request']['2'],4 if suffix else 3)

    def test_decode_ignores_proxy_override(self):
        out=self.root/'decode';create(out,c=2,rows_per_request=3)
        with Trace(out) as t:r=replay(t,bootstrap_samples=0,rows_per_request={2:4})
        self.assertIsNone(r['proxy_rows_per_request'])
        self.assertTrue(all(e['cell']=='c2_M6_K2' for e in r['cyclesim_what_if']))
