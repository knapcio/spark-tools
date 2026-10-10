# SPDX-License-Identifier: Apache-2.0
"""The source archive must retain every public fixture used by its docs/tests."""
import os
from pathlib import Path
import shutil
import tarfile

from setuptools import build_meta


ROOT = Path(__file__).resolve().parents[1]


def test_sdist_contains_public_docs_configs_examples_and_smoke(tmp_path):
    source = tmp_path / 'source'
    shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
        '.git', '.pytest_cache', '__pycache__', '*.pyc', '*.pyo', '*.egg-info',
        'build', 'dist', '.DS_Store'))
    output = tmp_path / 'dist'
    output.mkdir()
    previous = Path.cwd()
    try:
        os.chdir(source)
        archive = output / build_meta.build_sdist(str(output))
    finally:
        os.chdir(previous)
    with tarfile.open(archive, 'r:gz') as package:
        names = {name.split('/', 1)[1] for name in package.getnames() if '/' in name}
    required = {
        'CREDITS.md',
        'CHANGELOG.md',
        'tests/AUDIT_REGRESSIONS.md',
        'tests/run_examples.sh',
        'configs/glm53-tp4.json',
        'docs/scope.md',
        'examples/cyclesim/calibration.json',
        'examples/moereplay/run.py',
        'examples/exl3-tensorfold/README.md',
        'docs/exl3-tensorfold.md',
        'LICENSES/MIT-exllamav3.txt',
        'tests/smoke_examples.py',
    }
    assert required <= names
