# SPDX-License-Identifier: Apache-2.0
"""Reviewer regressions; each failure class reproduces on the pre-fix tree."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from lab.rankharness import api as rh

FAULTS = [rh.Fault(n, 'vote') for n in
          ('descendants', 'descendants_after_exit', 'exit_17', 'exit_0', 'recursive')]
EXPECTED = dict(descendants='HANG', descendants_after_exit='PASS',
                exit_17='FATAL', exit_0='FATAL', recursive='FALLBACK')


def worker(ctx):
    name = ctx.fault['name']
    hit = ctx.fault_point('vote')
    if name in {'descendants', 'descendants_after_exit'} and hit:
        # Child ignores TERM and launches a grandchild in the same rank group.
        code = ('import subprocess,sys,signal,time; '
                'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                'print(p.pid,flush=True); time.sleep(60)')
        child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True)
        grandchild = int(child.stdout.readline())
        (ctx.output / 'descendants.json').write_text(json.dumps([child.pid, grandchild]))
        ctx.stage('child_started')
        if name == 'descendants':
            while True:
                time.sleep(.02)
    if name.startswith('exit_') and hit:
        ctx.stage('abrupt_exit')
        os._exit(int(name.split('_')[1]))
    payload = {'on': True}
    if name == 'recursive' and hit:
        payload = []
        for _ in range(max(10000, sys.getrecursionlimit() + 100)):
            payload = [payload]
    approved = rh.all_rank_vote(True, payload)
    return dict(verdict='PASS' if approved else 'FALLBACK', decision=approved,
                state={'on': approved})


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class CollectiveRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.result = rh.run_matrix(worker, FAULTS,
                                   timeout_s=.5, repeat=2, out=Path(cls.temp.name) / 'matrix',
                                   expected=EXPECTED)
        cls.runs = [json.loads(line) for line in
                    (cls.result.out / 'verdicts.jsonl').read_text().splitlines()]

    @classmethod
    def tearDownClass(cls):
        # Baseline must not leave the deliberately leaked descendants behind.
        for path in cls.result.out.rglob('descendants.json'):
            for pid in json.loads(path.read_text()):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        cls.temp.cleanup()

    def test_descendant_group_cleanup(self):
        paths = list(self.result.out.rglob('descendants.json'))
        self.assertEqual(len(paths), 4)
        pids = [pid for path in paths for pid in json.loads(path.read_text())]
        deadline = time.monotonic() + 2
        while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertEqual([pid for pid in pids if alive(pid)], [])
        for run in self.runs:
            if run['fault'] == 'descendants':
                self.assertEqual(run['verdict'], 'HANG')
                self.assertEqual(run['last_stages']['2'], 'child_started')
                self.assertTrue(all(r['process_reaped'] for r in run['results']))
            if run['fault'] == 'descendants_after_exit':
                self.assertEqual(run['verdict'], 'PASS')
                self.assertEqual([r['process_exitcode'] for r in run['results']], [0] * 4)

    def test_premature_exit_is_fatal_with_exit_code(self):
        for run in self.runs:
            if run['fault'].startswith('exit_'):
                self.assertEqual(run['verdict'], 'FATAL')
                self.assertIsNone(run['timeout_phase'])
                self.assertEqual(run['results'][2]['process_exitcode'], int(run['fault'].split('_')[1]))
                self.assertFalse(run['results'][2]['finished'])
                self.assertLess(run['elapsed_s'], 10)

    def test_recursive_payload_collective_refusal(self):
        for run in self.runs:
            if run['fault'] == 'recursive':
                self.assertEqual(run['verdict'], 'FALLBACK')
                self.assertEqual([r['state'] for r in run['results']], [{'on': False}] * 4)


class StorageRegressions(unittest.TestCase):
    def test_empty_mac_group_permission_vs_live_rank(self):
        for live in (False, True):
            process = NS(pid=123, is_alive=lambda: live)
            with patch.object(rh.os, 'killpg', side_effect=PermissionError('empty group')), \
                 patch.object(rh.os, 'getpgid', return_value=123):
                if live:
                    with self.assertRaises(PermissionError):
                        rh._signal_rank_group(process, signal.SIGKILL)
                else:
                    rh._signal_rank_group(process, signal.SIGKILL)
        process = NS(pid=123, is_alive=lambda: True)
        with patch.object(rh.os, 'killpg', side_effect=PermissionError('before setsid')), \
             patch.object(rh.os, 'getpgid', return_value=456):
            rh._signal_rank_group(process, signal.SIGKILL)

    def test_direct_storage_leak_and_tensor_alias_deduplication(self):
        import torch
        retained = []
        probe = rh.leak_probe(lambda i: retained.append(torch.zeros(32768).untyped_storage()), toggles=4)
        self.assertTrue(probe['leak_detected'])
        self.assertEqual(probe['bytes_per_toggle']['cpu_tensor_bytes'], 131072)
        retained.clear()
        baseline = rh.cpu_tensor_bytes()
        tensor = torch.zeros(32768)
        view = tensor[:16]
        storage = tensor.untyped_storage()
        self.assertEqual(view.untyped_storage().data_ptr(), storage.data_ptr())
        self.assertEqual(rh.cpu_tensor_bytes() - baseline, 131072)
        del tensor, view
        self.assertEqual(rh.cpu_tensor_bytes() - baseline, 131072)

    def test_any_serialization_exception_invalidates_packet(self):
        for exc in (RecursionError('nested'), RuntimeError('encoder')):
            with patch.object(rh, '_json', side_effect=exc):
                self.assertEqual(rh._digest_packet({}), (bytes(32), False))


if __name__ == '__main__':
    unittest.main()
