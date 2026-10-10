# SPDX-License-Identifier: Apache-2.0
"""Compare the existing local fixtures with deterministic expected outputs."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    commands = [
        ([sys.executable, '-m', 'cyclesim', 'predict', '--config',
          'examples/cyclesim/calibration.json'], 'expected-prediction.json'),
        ([sys.executable, '-m', 'examples.cyclesim.compare'], 'expected-comparison.json'),
        ([sys.executable, '-m', 'examples.cyclesim.validate'], 'expected-validation.json'),
    ]
    for command, expected in commands:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                              check=True, timeout=30)
        assert json.loads(proc.stdout) == json.loads((ROOT / 'examples/cyclesim' / expected).read_text()), expected
    for fixture, expected in [('profile-fixture', {'status': 'measured-time/estimated-traffic', 'ranks': 4, 'kernel_rows': 16}),
                              ('companion-fixture', {'status': 'usable-companions-only', 'ranks': 0, 'kernel_rows': 0})]:
        proc = subprocess.run([sys.executable, '-m', 'cyclesim.profile',
                               'examples/cyclesim/' + fixture, '--ledger',
                               'examples/cyclesim/ledger.json', '--check'], cwd=ROOT,
                              capture_output=True, text=True,
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                              check=True, timeout=30)
        assert json.loads(proc.stdout) == expected, fixture
    print('PASS: 5 cyclesim fixture commands; no GPU required')


if __name__ == '__main__':
    main()
