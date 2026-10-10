# SPDX-License-Identifier: Apache-2.0
"""Quality CLI and frozen request/identity regressions using mocked transport."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from common.config import dump
from common.prereg import freeze, verify
from qpanel import cli

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples/qpanel'


def test_module_cli_matches_saved_output():
    args = [sys.executable, '-B', '-m', 'qpanel', 'decide', '--prereg', str(EXAMPLE / 'prereg.json')]
    for name in ('a', 'a2', 'b'):
        args += ['--'+name, str(EXAMPLE / (name.upper()+'.json'))]
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == json.loads((EXAMPLE / 'expected-decision.json').read_text())


@pytest.mark.parametrize('config', [
    {'items': 0}, {'items': True}, {'items': 1, 'rule': 'other'},
    {'items': 1, 'decision_rule': 'always pass'}, {'tasks': []},
    {'items': 2, 'tasks': [{'key': 'one', 'body': {}, 'expected': ''}]},
])
def test_invalid_plan_refused_before_creation(config, tmp_path):
    path, plan = tmp_path / 'config.json', tmp_path / 'plan.json'
    dump(config, path)
    with pytest.raises(SystemExit) as error:
        cli.main(['preregister', '--config', str(path), '--prereg', str(plan)])
    assert error.value.code == 2
    assert not plan.exists()


@pytest.mark.parametrize('fault', ['arm', 'digest', 'count', 'identity'])
def test_decide_refuses_frozen_panel_drift(fault, tmp_path):
    plan = tmp_path / 'plan.json'
    cfg = {'tasks': [{'key': 'probe', 'body': {}, 'expected': 'OK'}]}
    freeze(cfg, plan, cli.SOURCES)
    sha = verify(plan, cli.SOURCES)[1]
    row = {'key': 'probe', 'pass': True, 'truncated': False,
           'response_sha256': 'sample', 'output': {'content': 'OK'}}
    args = ['decide', '--prereg', str(plan)]
    for arm in ('A', 'A2', 'B'):
        report = {'status': 'COMPLETE', 'arm': arm, 'prereg_sha256': sha, 'results': [dict(row)]}
        if arm == 'B':
            if fault == 'arm': report['arm'] = 'A'
            if fault == 'digest': report['prereg_sha256'] = 'changed'
            if fault == 'count': report['results'] = []
            if fault == 'identity': report['results'][0]['key'] = 'different'
        path = tmp_path / (arm+'.json')
        dump(report, path)
        args += ['--'+arm.lower(), str(path)]
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2


def test_run_uses_frozen_endpoint_body_and_cleans_up_on_failure(tmp_path, monkeypatch):
    config = {'endpoints': {'A': 'http://127.0.0.1:8000'}, 'tasks': [
        {'key': 'probe', 'body': {'model': 'example-model', 'stream': False}, 'expected': 'OK'}]}
    plan, output = tmp_path / 'plan.json', tmp_path / 'output.json'
    freeze(config, plan, cli.SOURCES)
    seen = []

    class Transport:
        def __init__(self, endpoint, token): seen.append(endpoint)
        def ask(self, body, timeout):
            seen.append(body)
            raise OSError('mock transport failure')
        def cancel(self): seen.append('cancel')

    monkeypatch.setattr(cli, 'HTTP', Transport)
    with pytest.raises(SystemExit):
        cli.main(['run', '--prereg', str(plan), '--arm', 'A', '--out', str(output)])
    assert seen == [config['endpoints']['A'], config['tasks'][0]['body'], 'cancel']
    assert not output.exists()


def test_quality_failure_returns_nonzero(tmp_path, capsys):
    args = ['decide', '--prereg', str(EXAMPLE / 'prereg.json')]
    for arm in ('A', 'A2', 'B'):
        report = json.loads((EXAMPLE / (arm+'.json')).read_text())
        if arm == 'B':
            for row in report['results']: row['pass'] = False
        path = tmp_path / (arm+'.json')
        dump(report, path)
        args += ['--'+arm.lower(), str(path)]
    assert cli.main(args) == 1
    assert json.loads(capsys.readouterr().out)['decision'] == 'FAIL'
