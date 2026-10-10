# SPDX-License-Identifier: Apache-2.0
"""Verify the shipped synthetic inputs through the actual command adapter."""
import contextlib
import io
import json
from pathlib import Path

from moereplay.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_schema1_example_cli_matches_saved_report(tmp_path):
    for name in ('routes', 'training'):
        with contextlib.redirect_stdout(io.StringIO()):
            assert main(['convert-schema1', str(ROOT / 'examples/moereplay' / (name + '.jsonl')),
                         '--source-status', 'assumed', '--model', 'example-model',
                         '--out', str(tmp_path / name)]) == 0
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        assert main(['check', str(tmp_path / 'routes'), '--decode-only', '--min-steps', '4']) == 0
    assert json.loads(captured.getvalue())['structural_check'] == 'PASS'
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        assert main(['replay', str(tmp_path / 'routes'), '--train-trace', str(tmp_path / 'training'),
                     '--bootstrap', '200']) == 0
    result = json.loads(captured.getvalue())
    assert result == json.loads((ROOT / 'examples/moereplay/expected.json').read_text())
    assert result['source']['status'] == 'assumed'
    assert not any(point['calibration_eligible'] for point in result['cyclesim_what_if'])
