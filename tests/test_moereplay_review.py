# SPDX-License-Identifier: Apache-2.0
"""Review regressions; export checks opt in via SPARK_EXPORT_ROOT."""
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

from common.audit import audit_tree
from moereplay.replay import replay
from moereplay.routes import Trace
from test_moereplay import create
from test_schedreplay import Timing
from lab.schedreplay.subsets import supported_subset
from lab.schedreplay.workloads import synthetic


class ReviewRegressions(unittest.TestCase):
    def test_unpinned_runtime_compatibility_is_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            create(Path(directory) / 'routes')
            with Trace(Path(directory) / 'routes') as trace:
                result = replay(trace, bootstrap_samples=0)
        for key, row in result['cycles'].items():
            with self.subTest(cell=key):
                self.assertIsNone(row['supported_in_pinned_vllm'])
                self.assertIsNone(row['serving_compatible'])
                self.assertIn('no runtime pin or compatibility check', row['serving_limitation'])

    def test_reversed_independent_workload_preserves_supported_subset(self):
        works, metadata = synthetic('long-during-decode')
        expected = supported_subset(works, metadata, Timing)
        self.assertEqual(expected['status'], 'estimated-supported')
        actual = supported_subset(list(reversed(works)), metadata, Timing)
        self.assertEqual(actual['status'], 'estimated-supported')
        self.assertEqual(actual, expected)


@unittest.skipUnless(os.environ.get('SPARK_EXPORT_ROOT'), 'requires explicit local export root')
class ExportRegressions(unittest.TestCase):
    def setUp(self):
        self.root = Path(os.environ['SPARK_EXPORT_ROOT']).resolve()

    def invoke(self, command):
        if command[0] == 'moereplay':
            command = [sys.executable, '-B', '-m', 'moereplay.cli', *command[1:]]
        elif command[0] in ('python3', sys.executable):
            command = [sys.executable, '-B', *command[1:]]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(self.root))
        proc = subprocess.run(command, cwd=self.root, env=env, capture_output=True,
                              text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout),
                         json.loads((self.root / 'examples/moereplay/expected.json').read_text()))

    def test_readme_moe_command_matches_snapshot(self):
        text = (self.root / 'README.md').read_text().replace('\\\n', '')
        commands = [shlex.split(line) for line in text.splitlines()
                    if line.startswith(('moereplay replay ', 'python3 examples/moereplay/run.py'))]
        self.assertEqual(len(commands), 1)
        self.invoke(commands[0])

    def test_smoke_moe_command_matches_snapshot(self):
        tree = ast.parse((self.root / 'tests/smoke_examples.py').read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'invoke'
                 and any(k.arg == 'snapshot' and isinstance(k.value, ast.Constant)
                         and k.value.value == 'examples/moereplay/expected.json'
                         for k in node.keywords)]
        self.assertEqual(len(calls), 1)
        command = []
        for item in calls[0].args[0].elts:
            if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name):
                self.assertEqual((item.value.id, item.attr), ('sys', 'executable'))
                command.append(sys.executable)
            else:
                command.append(ast.literal_eval(item))
        self.invoke(command)

    def test_export_all_objects_and_metadata_are_clean(self):
        report = audit_tree(self.root)
        self.assertGreater(report['git_objects'], 0)
        self.assertEqual(report['findings'], [], report['findings'])


if __name__ == '__main__':
    unittest.main()
