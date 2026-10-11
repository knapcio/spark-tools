# SPDX-License-Identifier: Apache-2.0
"""CLI, preregistration and saved-fixture regressions."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from abharness import cli
from abharness.cli import main, RULE, SOURCES
from common.config import dump
from common.prereg import freeze, verify

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples/abharness'


@pytest.mark.parametrize('command, input_name, expected_name', [
    ('decide', 'panels.json', 'expected-decision.json'),
    ('near-tie', 'near-tie.json', 'expected-near-tie.json'),
])
def test_module_cli_matches_saved_output(command, input_name, expected_name):
    args = [sys.executable, '-B', '-m', 'abharness', command, '--input', str(EXAMPLE / input_name)]
    if command == 'decide':
        args += ['--prereg', str(EXAMPLE / 'prereg.json')]
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == json.loads((EXAMPLE / expected_name).read_text())


def test_create_only_preregistration_and_reject_rule_before_freezing(tmp_path):
    config, plan = tmp_path / 'config.json', tmp_path / 'plan.json'
    dump(RULE, config)
    assert main(['preregister', '--config', str(config), '--prereg', str(plan)]) == 0
    assert verify(plan, SOURCES)[0] == RULE
    assert main(['preregister', '--config', str(config), '--prereg', str(plan)]) == 0
    invalid = tmp_path / 'invalid.json'
    dump(dict(RULE, prose_saving_ms=0), config)
    with pytest.raises(SystemExit):
        main(['preregister', '--config', str(config), '--prereg', str(invalid)])
    assert not invalid.exists()


def test_decision_rejects_plan_and_digest_drift(tmp_path):
    plan, data = tmp_path / 'plan.json', tmp_path / 'panels.json'
    freeze(dict(RULE, prose_saving_ms=0), plan, SOURCES)
    panels = json.loads((EXAMPLE / 'panels.json').read_text())
    panels['prereg_sha256'] = verify(plan, SOURCES)[1]
    dump(panels, data)
    with pytest.raises(SystemExit) as error:
        main(['decide', '--input', str(data), '--prereg', str(plan)])
    assert error.value.code == 2
    panels['prereg_sha256'] = 'changed'
    dump(panels, data)
    with pytest.raises(SystemExit):
        main(['decide', '--input', str(data), '--prereg', str(plan)])


def test_near_tie_hold_returns_nonzero(tmp_path, capsys):
    data = json.loads((EXAMPLE / 'near-tie.json').read_text())
    data.pop('reference')
    path = tmp_path / 't0.json'
    dump(data, path)
    assert main(['near-tie', '--input', str(path)]) == 1
    assert json.loads(capsys.readouterr().out)['decision'] == 'HOLD'


def test_cycle_kill_returns_nonzero(tmp_path, capsys):
    data = json.loads((EXAMPLE / 'panels.json').read_text())
    for panel in data['B']:
        for row in panel['rows']:
            row['cycle_ms'] = 20
    path = tmp_path / 'panels.json'
    dump(data, path)
    assert main(['decide', '--prereg', str(EXAMPLE / 'prereg.json'), '--input', str(path)]) == 1
    assert json.loads(capsys.readouterr().out)['decision'] == 'KILL'


@pytest.mark.parametrize('failed', [False, True])
def test_request_uses_configured_body_and_cleans_up(tmp_path, monkeypatch, failed):
    body = {'model': 'example-model', 'messages': [{'role': 'user', 'content': 'Reply with exactly OK.'}],
            'max_tokens': 4, 'temperature': 0, 'stream': False}
    config, output = tmp_path / 'config.json', tmp_path / 'response.json'
    dump({'endpoint': 'http://127.0.0.1:8000', 'body': body, 'timeout_s': 1}, config)
    calls = []

    class Transport:
        def __init__(self, endpoint, token): calls.append(endpoint)
        def ask(self, request, timeout):
            calls.append((request, timeout))
            if failed: raise OSError('mock transport failure')
            return b'{"choices":[{"message":{"content":"OK"},"finish_reason":"stop"}]}'
        def cancel(self): calls.append('cancel')

    monkeypatch.setattr(cli, 'HTTP', Transport)
    args = ['request', '--config', str(config), '--out', str(output)]
    if failed:
        with pytest.raises(SystemExit): main(args)
        assert not output.exists()
    else:
        assert main(args) == 0
        assert json.loads(output.read_text())['choices'][0]['message']['content'] == 'OK'
    assert calls == ['http://127.0.0.1:8000', (body, 1), 'cancel']
