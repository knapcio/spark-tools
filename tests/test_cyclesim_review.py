# SPDX-License-Identifier: Apache-2.0
"""Regressions for holdout identities and trace boundary accounting."""
import json
import tempfile
import unittest
from pathlib import Path

from cyclesim.calibrate import validate
from cyclesim.profile import analyze
from cyclesim.utilities import dump, stats, trace_events
from test_cyclesim import HERE, trace


class HoldoutIdentities(unittest.TestCase):
    def fixtures(self):
        return (json.loads((HERE / 'calibration.json').read_text()),
                json.loads((HERE / 'holdout.json').read_text()))

    def assert_unknown(self, calibration, observations):
        rows = validate(calibration, observations)
        self.assertEqual(len(rows), 6)
        self.assertEqual({row['role'] for row in rows}, {'unknown-prompt-transfer'})
        self.assertEqual({entry['cohort'] for entry in calibration['metric_errors']},
                         {'unknown-prompt-transfer'})

    def test_six_holdouts_without_prompt_hashes_are_unknown(self):
        calibration, observations = self.fixtures()
        for observation in observations['observations']:
            del observation['prompt_hashes']
        self.assert_unknown(calibration, observations)

    def test_empty_or_invalid_prompt_identities_are_unknown(self):
        for hashes in ([], None, [''], [' '], [None], 'holdout-1'):
            with self.subTest(hashes=hashes):
                calibration, observations = self.fixtures()
                for observation in observations['observations']:
                    observation['prompt_hashes'] = hashes
                self.assert_unknown(calibration, observations)

    def test_unavailable_fit_identities_cannot_verify_disjointness(self):
        for hashes in (None, [], ['']):
            with self.subTest(hashes=hashes):
                calibration, observations = self.fixtures()
                if hashes is None:
                    del calibration['fit_prompt_hashes']
                else:
                    calibration['fit_prompt_hashes'] = hashes
                self.assert_unknown(calibration, observations)

    def test_identified_disjoint_holdouts_remain_holdouts(self):
        calibration, observations = self.fixtures()
        self.assertEqual({row['role'] for row in validate(calibration, observations)},
                         {'holdout'})


class TraceBoundaries(unittest.TestCase):
    def test_busy_activity_crossing_into_next_cycle_is_clipped(self):
        for cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            with self.subTest(cat=cat), tempfile.TemporaryDirectory() as td:
                document = trace()
                # This starts in the discarded first cycle and remains busy
                # through two complete kept cycles and part of the third.
                document['traceEvents'].append({
                    'ph': 'X', 'cat': cat, 'name': 'background_copy',
                    'ts': 900, 'dur': 2500, 'args': {}})
                path = Path(td) / 'rank0.pt.trace.json'
                dump(path, document)
                ledger = json.loads((HERE / 'ledger-historical.json').read_text())
                summary, _, _ = analyze(path, ledger, target_norms=1)
                self.assertEqual(summary['kernel_union_ms'], stats([1.0, 1.0, .4, .35]))
                self.assertEqual(summary['idle_ms'], stats([0.0, 0.0, .6, .65]))
                self.assertAlmostEqual(summary['idle_ms']['mean'], .3125)

    def check_numeric_boundary(self, prefix, suffix, expected):
        # JsonStream reads 65536 characters: split exactly after 1. or 1e.
        header = '{"traceEvents": [], "numeric": '
        document = header + ' ' * (65536 - len(header) - len(prefix)) + prefix + suffix + '}'
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'trace.json'
            path.write_text(document)
            metadata = {}
            self.assertEqual(list(trace_events(path, metadata)), [])
            self.assertEqual(metadata['numeric'], expected)

    def test_top_level_fraction_split_after_decimal_point(self):
        self.check_numeric_boundary('1.', '25', 1.25)

    def test_top_level_exponent_split_after_exponent_marker(self):
        self.check_numeric_boundary('1e', '3', 1000)

    def test_top_level_exponent_split_after_sign(self):
        self.check_numeric_boundary('-1.25e-', '3', -.00125)

    def test_incomplete_or_invalid_numeric_metadata_is_rejected(self):
        for prefix, suffix in (('1.', ''), ('1e', ''), ('1e-', ''), ('1.', 'x'), ('1e', '+x')):
            with self.subTest(prefix=prefix, suffix=suffix), self.assertRaises(ValueError):
                self.check_numeric_boundary(prefix, suffix, None)


if __name__ == '__main__':
    unittest.main()
