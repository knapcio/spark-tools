# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
from pathlib import Path

from lab.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_counter_and_hash_example_commands_match_saved_reports():
    for command, snapshot in [
        (['acceptance', '--before', str(ROOT / 'examples/lab/before.prom'),
          '--after', str(ROOT / 'examples/lab/after.prom'), '--k', '2'], 'expected-acceptance.json'),
        (['sha256', str(ROOT / 'examples/lab/before.prom')], 'expected-hash.json'),
    ]:
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            assert main(command) == 0
        assert json.loads(captured.getvalue()) == json.loads((ROOT / 'examples/lab' / snapshot).read_text())
