# SPDX-License-Identifier: Apache-2.0
"""Fix-round-2 counterexamples; all fixtures are synthetic, offline captures."""
import json
import tempfile
import unittest
from pathlib import Path

from cyclesim.build_ledger import build
from cyclesim.utilities import dump
from cyclesim.profile import analyze, check_capture, ingest
from cyclesim.capture import manifest
from test_cyclesim import trace
from test_cyclesim_fix1 import capture, mutate

HERE = Path(__file__).resolve().parents[1] / "examples/cyclesim"
CONFIG = {'hidden_size':6144, 'moe_intermediate_size':2048, 'intermediate_size':12288,
          'q_lora_rank':2048, 'kv_lora_rank':512, 'qk_rope_head_dim':64,
          'num_attention_heads':64, 'qk_head_dim':256, 'v_head_dim':256,
          'qk_nope_head_dim':192, 'n_routed_experts':256, 'vocab_size':154880}


def bracket(root, mode='profile', c=1):
    capture(root, mode, c)
    rows = [dict(arm=arm, request_id='%s-s%d' % (arm, stream), stream=stream,
                 chunk_index=i, monotonic_ns=1000*a+i+1)
            for a, arm in enumerate(('A1', 'P', 'A2') if mode=='profile' else ('A1', 'A2'))
            for stream in range(c) for i in range(17)]
    write_timings(root, rows)
    return rows


def write_timings(root, rows):
    (root/'client-timings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    refresh(root)


def refresh(root):
    doc = json.loads((root/'manifest.json').read_text())
    manifest(root, doc['mode'], doc['c'], doc['domain'], doc['unprofiled_cycle_ms'])


def widths_only(doc, nulls=False):
    for rank in doc['ranks'].values():
        rank['executed_passes'] = None
        rank['evidence'] = 'FULL graph descriptor sizes recorded at boot'
        rank['descriptors'] = [dict(rows=d['rows'], **({'graph_id':None, 'norms':None} if nulls else {}))
                               for d in rank['descriptors']]


class Fix2(unittest.TestCase):
    def test_mtp_dense_and_expert_ingestion(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dump(root/'config.json', CONFIG)
            for historical in (False, True):
                ledger = build(root, root/'config.json', historical)
                dense = sum(r['weight_bytes_per_layer_per_rank'] for r in ledger['rows']
                            if r['group'] in ('mtp_attention', 'mtp_shared') and r['decode_read'])
                experts = sum(r['weight_bytes_per_layer_per_rank'] for r in ledger['rows']
                              if r['group']=='mtp_experts' and r['decode_read'])
                family = 'int8_gemm' if historical else 'nvfp4_gemm'
                name = 'marlin::Marlin W8A16' if historical else 'nvfp4_gemm'
                for c in (1, 2, 4):
                    with self.subTest(historical=historical, c=c):
                        doc = trace()
                        for cycle in range(6):
                            for node, kernel in enumerate((name, 'marlin_moe_nvfp4')):
                                doc['traceEvents'].append({'ph':'X', 'cat':'kernel', 'name':kernel,
                                    'ts':1000*cycle+500+20*node, 'dur':10,
                                    'args':{'graph id':2, 'graph node id':node, 'correlation':cycle,
                                            'projection':'mtp_body'}})
                        dump(root/'rank0.pt.trace.json', doc)
                        _, rows, _ = analyze(root/'rank0.pt.trace.json', ledger, c, 2, target_norms=1)
                        dense_row = next(r for r in rows if r['class']=='draft/'+family)
                        expert_row = next(r for r in rows if r['class']=='draft/marlin_moe')
                        self.assertEqual(dense_row['estimated_bytes_read'], dense)
                        self.assertEqual(expert_row['estimated_bytes_read'], experts*c*8)
                        self.assertEqual(dense_row['achieved_gbs'], dense/10000)
                        self.assertEqual(expert_row['achieved_gbs'], experts*c*8/10000)
                        self.assertIsNone(expert_row['working_set_bytes'])

    def test_profile_three_arm_bracket(self):
        for c in (1, 2, 4):
            with self.subTest(c=c), tempfile.TemporaryDirectory() as td:
                root = Path(td); bracket(root, c=c)
                out = ingest(root, json.loads((HERE/'ledger.json').read_text()), c, 3 if c==1 else 2, strict=True)
                self.assertEqual(len(out['ranks']), 4)
                self.assertEqual(out['ranks'][0]['cycles_kept'], 16)

    def test_each_arm_and_stream_requires_coverage(self):
        for mode in ('profile', 'companions-only'):
            for arm in (('A1', 'P', 'A2') if mode=='profile' else ('A1', 'A2')):
                with self.subTest(mode=mode, arm=arm), tempfile.TemporaryDirectory() as td:
                    root = Path(td); rows = bracket(root, mode, 4)
                    rows = [r for r in rows if not (r['arm']==arm and r['stream']==3 and r['chunk_index']==16)]
                    write_timings(root, rows)
                    with self.assertRaisesRegex(ValueError, '16 consecutive'):
                        check_capture(root)

    def test_missing_and_unexpected_arms_rejected(self):
        for mode in ('profile', 'companions-only'):
            for change in ('missing', 'unknown', 'unlabeled'):
                with self.subTest(mode=mode, change=change), tempfile.TemporaryDirectory() as td:
                    root = Path(td); rows = bracket(root, mode)
                    if change=='missing': rows = [r for r in rows if r['arm']!='A2']
                    elif change=='unknown': rows[0]['arm'] = 'P' if mode=='companions-only' else 'other'
                    else: rows[0].pop('arm')
                    write_timings(root, rows)
                    with self.assertRaises(ValueError): check_capture(root)

    def test_duplicate_request_and_bad_intervals_rejected_per_arm(self):
        for change in ('request', 'stream', 'time', 'index'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as td:
                root = Path(td); rows = bracket(root, c=2)
                row = next(r for r in rows if r['arm']=='A2' and r['stream']==1 and r['chunk_index']==8)
                if change=='request': row['request_id'] = 'another-request'
                elif change=='stream': row['stream'] = 2
                elif change=='time': row['monotonic_ns'] -= 1
                else: row['chunk_index'] += 1
                write_timings(root, rows)
                with self.assertRaises(ValueError): check_capture(root)

    def test_no_boot_widths_only(self):
        for c in (1, 2, 4):
            for nulls in (False, True):
                with self.subTest(c=c, nulls=nulls), tempfile.TemporaryDirectory() as td:
                    root = Path(td); bracket(root, 'companions-only', c)
                    mutate(root, 'graph-sizes.json', lambda d: widths_only(d, nulls))
                    out = ingest(root, json.loads((HERE/'ledger.json').read_text()), c, 3 if c==1 else 2, strict=True)
                    self.assertEqual(out['status'], 'usable-companions-only')
                    self.assertEqual(out['kernels'], [])
                    self.assertIsNone(check_capture(root)['executed_passes'])

    def test_profile_requires_resolved_graph_metadata(self):
        for field in ('graph_id', 'norms', 'executed_passes'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                root = Path(td); bracket(root)
                def remove(doc):
                    for rank in doc['ranks'].values():
                        if field=='executed_passes': rank[field] = None
                        else:
                            for descriptor in rank['descriptors']: descriptor[field] = None
                mutate(root, 'graph-sizes.json', remove)
                with self.assertRaises(ValueError): check_capture(root)

    def test_no_boot_still_requires_valid_sizes_and_state(self):
        for change in ('width', 'ranks', 'evidence', 'head', 'id', 'norms'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as td:
                root = Path(td); bracket(root, 'companions-only')
                def invalid(doc):
                    widths_only(doc)
                    rank = doc['ranks']['0']
                    if change=='width': rank['descriptors'][0]['rows'] = 8
                    elif change=='ranks': doc['ranks'].pop('3')
                    elif change=='evidence': rank['evidence'] = ''
                    elif change=='head': rank['head'] = 'nvfp4'
                    elif change=='id': rank['descriptors'][0]['graph_id'] = 'unknown'
                    else: rank['descriptors'][0]['norms'] = -1
                mutate(root, 'graph-sizes.json', invalid)
                with self.assertRaises(ValueError): check_capture(root)
