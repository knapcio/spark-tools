# SPDX-License-Identifier: Apache-2.0
"""Local fixture smoke tests; no HTTP requests or CUDA compiler invocation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def invoke(command, expected_code=0, snapshot=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    if proc.returncode != expected_code:
        raise AssertionError(f'{command[0]} exit {proc.returncode}: {proc.stdout}\n{proc.stderr}')
    if snapshot:
        actual = json.loads(proc.stdout)
        expected = json.loads((ROOT / snapshot).read_text())
        if actual != expected: raise AssertionError('example output differs from '+snapshot)
    print('PASS '+ ' '.join(command[:2]))
    return proc.stdout


def main():
    invoke(['gb10check', 'parse', '--config', 'examples/gb10check/pass.json'],
           snapshot='examples/gb10check/expected-pass.json')
    invoke(['gb10check', 'parse', '--config', 'examples/gb10check/fail.json'], 1,
           'examples/gb10check/expected-fail.json')
    invoke(['cyclesim', 'predict', '--config', 'examples/cyclesim/calibration.json'],
           snapshot='examples/cyclesim/expected-prediction.json')
    invoke([sys.executable, '-m', 'examples.cyclesim.compare'],
           snapshot='examples/cyclesim/expected-comparison.json')
    invoke([sys.executable, '-m', 'examples.cyclesim.validate'],
           snapshot='examples/cyclesim/expected-validation.json')
    invoke([sys.executable, '-m', 'examples.cyclesim.smoke'])
    invoke([sys.executable, 'examples/moereplay/run.py'],
           snapshot='examples/moereplay/expected.json')
    for name in ('newcomer', 'agents'):
        invoke(['lab', 'schedreplay', 'run', '--config', 'configs/scheduler.json',
                '--workload', 'examples/schedreplay/'+name+'.jsonl'],
               snapshot='examples/schedreplay/expected-'+name+'.json')
    invoke(['abharness', 'decide', '--prereg', 'examples/abharness/prereg.json',
            '--input', 'examples/abharness/panels.json'], snapshot='examples/abharness/expected-decision.json')
    invoke(['abharness', 'near-tie', '--input', 'examples/abharness/near-tie.json'],
           snapshot='examples/abharness/expected-near-tie.json')
    invoke(['qpanel', 'decide', '--prereg', 'examples/qpanel/prereg.json', '--a', 'examples/qpanel/A.json',
            '--a2', 'examples/qpanel/A2.json', '--b', 'examples/qpanel/B.json'],
           snapshot='examples/qpanel/expected-decision.json')
    invoke(['lab', 'acceptance', '--before', 'examples/lab/before.prom',
            '--after', 'examples/lab/after.prom', '--k', '2'], snapshot='examples/lab/expected-acceptance.json')
    invoke(['lab', 'sha256', 'examples/lab/before.prom'], snapshot='examples/lab/expected-hash.json')
    invoke(['lab', 'exl3', '--help'])
    invoke(['lab', 'exlr8', '--help'])
    invoke(['lab', 'tensorfold', '--help'])
    invoke(['lab', 'conformance', 'cpu'], snapshot='examples/conformance/expected-cpu.json')
    invoke(['lab', 'conformance', 'manifest', '--runner', 'gr8-dense'],
           snapshot='examples/conformance/gr8-dense.json')
    invoke(['lab', 'observations', 'import', '--adapter', 'tensorfold-native-v1',
            '--input', 'examples/observations/tensorfold-streaming.json',
            '--metadata', 'examples/observations/tensorfold-native-import.json'])
    invoke(['lab', 'experiment', 'check', 'examples/experiment/manifest.json'])
    if importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'):
        with tempfile.TemporaryDirectory(prefix='spark-example-') as output:
            invoke(['lab','rankharness','run','examples/rankharness/toy_toggle.py','--cases','none,invalid_one',
                    '--repeat','2','--out',str(Path(output)/'toggle')])
            invoke(['lab','rankharness','run','examples/rankharness/rank_local_bug.py','--repeat','2',
                    '--timeout','0.5','--out',str(Path(output)/'local')])
    else:
        print('SKIP rank fixtures: install the ranks extra for CPU PyTorch/Gloo and NumPy')
    print('All available fixture examples passed; GPU and endpoint execution excluded.')


if __name__ == '__main__':
    main()
