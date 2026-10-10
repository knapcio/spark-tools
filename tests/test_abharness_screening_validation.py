# SPDX-License-Identifier: Apache-2.0
"""False-PASS regressions for the offline screening API."""
import unittest

from abharness.screening import bracket, det_flip, det_pair_stats, hard_equal


def bank(speed=1, token=1):
    return {kind+'-'+str(i): dict(
        metadata=dict(kind=kind, i=i, pair_key=kind+'-'+str(i), thinking=False),
        token_ids=[token, 2],
        top_logprobs=[[dict(token_id=n, logprob=-.1*n) for n in range(1, 6)]
                      for _ in range(2)],
        cycles=100, committed_tokens=300, seconds=1/speed, committed=3,
        cycle_ms=10/speed, tps=300*speed, time_source='client_decode_wall')
        for kind in ('prose', 'code', 'structured') for i in range(2)}


class ScreeningValidationTests(unittest.TestCase):
    def test_unknown_profile_rejects_ten_times_slower_candidate(self):
        with self.assertRaisesRegex(ValueError, 'profile'):
            bracket(bank(), bank(), bank(.1), det_pair_stats, profile='typo')

    def test_margin_uses_candidate_token_9_9_nats_below_baseline(self):
        before, after = bank(), bank()
        for panel in (before, after):
            for row in panel.values():
                row['top_logprobs'][0] = [dict(token_id=n, logprob=p) for n, p in
                                         zip(range(1, 6), (-.1, -.2, -1, -2, -10))]
        result = bracket(before, after, bank(1.1, 5), det_pair_stats, profile='tau')
        self.assertEqual(result['decision'], 'HOLD')
        for comparisons in result['groups'].values():
            for comparison in comparisons:
                for pair in comparison['pairs']:
                    self.assertAlmostEqual(pair['margin_nats'], 9.9)
                    self.assertTrue(pair['B_in_A_top5'])
                    self.assertFalse(pair['near_tie'])

    def test_reference_missing_baseline_selected_token_is_rejected(self):
        before, after = bank(), bank()
        for panel in (before, after):
            for row in panel.values():
                row['top_logprobs'][0] = [dict(token_id=n, logprob=-.1*n)
                                         for n in range(2, 7)]
        with self.assertRaisesRegex(ValueError, 'top-5'):
            bracket(before, after, bank(1.1, 3), det_pair_stats, profile='tau')

    def test_all_empty_token_sequences_are_rejected(self):
        panels = bank(), bank(), bank(1.1)
        for panel in panels:
            for row in panel.values():
                row['token_ids'] = []
        with self.assertRaisesRegex(ValueError, 'token'):
            bracket(*panels, det_pair_stats)

    def test_nan_cycle_measurements_are_rejected(self):
        candidate = bank(1.1)
        for row in candidate.values():
            row['cycle_ms'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'observation'):
            bracket(bank(), bank(), candidate, det_pair_stats)

    def test_nan_committed_measurements_are_rejected(self):
        candidate = bank(1.1)
        for row in candidate.values():
            row['committed'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'observation'):
            bracket(bank(), bank(), candidate, det_pair_stats)

    def test_invalid_native_token_types_are_rejected_before_equality(self):
        for ids in ([True, 2], [1.0, 2], ['1', 2], [-1, 2], '12', (1, 2)):
            with self.subTest(ids=ids):
                panels = bank(), bank(), bank(1.1)
                for panel in panels:
                    for row in panel.values():
                        row['token_ids'] = ids
                with self.assertRaisesRegex(ValueError, 'token'):
                    bracket(*panels, det_pair_stats)

    def test_invalid_observations_in_every_arm_are_rejected(self):
        fields = ('cycles', 'committed_tokens', 'seconds', 'committed', 'cycle_ms', 'tps')
        for arm in range(3):
            for field in fields:
                for value in (float('nan'), float('inf'), 0, -1, True, '1', None):
                    with self.subTest(arm=arm, field=field, value=value):
                        panels = bank(), bank(), bank(1.1)
                        panels[arm]['code-0'][field] = value
                        with self.assertRaisesRegex(ValueError, 'observation'):
                            bracket(*panels, det_pair_stats)

    def test_direct_helpers_reject_invalid_evidence(self):
        a, b = bank(), bank()
        for row in a.values():
            row['token_ids'] = []
        for row in b.values():
            row['token_ids'] = []
        with self.assertRaisesRegex(ValueError, 'token'):
            hard_equal(a, b)
        with self.assertRaisesRegex(ValueError, 'token'):
            det_flip(a['code-0'], b['code-0'])
        a, b = bank(), bank()
        b['code-0']['committed'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'observation'):
            det_pair_stats(a, b, ['code-0'])


if __name__ == '__main__':
    unittest.main()
