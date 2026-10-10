# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
from pathlib import Path

from lab.schedreplay.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_scheduler_example_commands_match_saved_reports():
    for name in ('newcomer', 'agents'):
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            assert main(['run', '--config', str(ROOT / 'configs/scheduler.json'),
                         '--workload', str(ROOT / 'examples/schedreplay' / (name + '.jsonl'))]) == 0
        assert json.loads(captured.getvalue()) == json.loads(
            (ROOT / 'examples/schedreplay' / ('expected-' + name + '.json')).read_text())


def test_existing_policy_sweep_keeps_baseline_and_withholds_unsupported_scope():
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        assert main(['sweep', '--config', str(ROOT / 'configs/scheduler.json'),
                     '--workload', str(ROOT / 'examples/schedreplay/newcomer.jsonl'),
                     '--chunks', '4096', '--decode-steps', '0,40', '--admissions', 'on']) == 0
    report = json.loads(captured.getvalue())
    assert report['schema'] == 'schedreplay-sweep v2'
    assert [row['policy'] for row in report['results']] == ['OFF', 'fair:4096:0', 'fair:4096:40']
    assert all(row['completed'] == 2 for row in report['results'])
    assert report['supported_subset']['status'] == 'indeterminate'
    assert report['supported_subset']['ranking'] == []
    assert all('requests' not in row for row in report['results'])
