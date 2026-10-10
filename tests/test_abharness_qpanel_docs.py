# SPDX-License-Identifier: Apache-2.0
"""Check that the documented setup and screening example are usable."""
from pathlib import Path
from copy import deepcopy
import re
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from abharness.screening import PROFILES, bracket, det_pair_stats


ROOT = Path(__file__).resolve().parents[1]


class ExampleDocumentationTests(unittest.TestCase):
    def test_both_examples_install_declared_pytest_extra(self):
        config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        self.assertTrue(any(dep.startswith('pytest')
                            for dep in config['project']['optional-dependencies']['test']))
        for package in ('abharness', 'qpanel'):
            with self.subTest(package=package):
                text = (ROOT / 'examples' / package / 'README.md').read_text()
                self.assertIn("pip install -e '.[test]'", text)

    def test_screening_example_supplies_schema_and_profile_contract(self):
        text = (ROOT / 'examples/abharness/README.md').read_text()
        examples = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
        self.assertEqual(len(examples), 1, 'one executable screening input example required')
        namespace = {}
        exec(compile(examples[0], 'documented screening example', 'exec'), namespace)
        self.assertEqual(namespace['result']['decision'], 'PASS')
        self.assertFalse(namespace['result']['release_gate'])
        for position in range(2):
            candidate = deepcopy(namespace['candidate'])
            candidate['probe']['token_ids'][position] = 3
            result = bracket(namespace['before'], namespace['after'], candidate, det_pair_stats)
            self.assertEqual(result['decision'], 'PASS')
        for profile in PROFILES:
            self.assertRegex(text, r'\| `' + profile + r'` \|')
        for contract in ('milliseconds', 'seconds', 'tokens per second', 'nats',
                         'first-token equality', 'cold-cache', 'two frozen',
                         'not release qualification'):
            with self.subTest(contract=contract):
                self.assertIn(contract, text)


if __name__ == '__main__':
    unittest.main()
