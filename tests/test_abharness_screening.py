# SPDX-License-Identifier: Apache-2.0
"""Retained synthetic bracket-screen fixtures; no endpoint or runtime actions."""
import copy

import pytest

from abharness.screening import bracket, det_pair_stats, hard_equal


def bank(speed=1, token=1):
    return {kind+'-'+str(i): dict(
        metadata=dict(kind=kind, i=i, pair_key=kind+'-'+str(i), thinking=False),
        token_ids=[token, 2],
        top_logprobs=[[dict(token_id=n, logprob=-.1*n) for n in range(1, 6)]]*2,
        cycles=100, committed_tokens=300, seconds=1/speed, committed=3,
        cycle_ms=10/speed, tps=300*speed, time_source='client_decode_wall')
        for kind in ('prose', 'code', 'structured') for i in range(2)}


def test_hard_token_gate_wins_over_performance():
    result = bracket(bank(), bank(), bank(1.1, 3), det_pair_stats, exact=True)
    assert result['decision'] == 'HOLD'
    assert 'HARD token-ID divergence' in result['reasons']


@pytest.mark.parametrize('profile', ['glue', 'async', 'trim'])
def test_exact_neutral_gain(profile):
    result = bracket(bank(), bank(), bank(1.1), det_pair_stats, exact=True, profile=profile)
    assert result['decision'] == 'PASS'
    assert result['release_gate'] is False


def test_control_drift_and_candidate_identity_refuse():
    with pytest.raises(ValueError):
        bracket(bank(), bank(token=3), bank(1.1), det_pair_stats)
    candidate = bank(1.1)
    candidate['prose-0']['metadata']['i'] = 99
    with pytest.raises(ValueError, match='identity drift'):
        bracket(bank(), bank(), candidate, det_pair_stats)
    with pytest.raises(ValueError, match='incomplete'):
        hard_equal({}, {})


@pytest.mark.parametrize('token, expected', [(3, 'PASS'), (9, 'HOLD')])
def test_draft_near_tie_and_absent_top_five(token, expected):
    result = bracket(bank(), bank(), bank(1.1, token), det_pair_stats, profile='tau')
    assert result['decision'] == expected


def test_length_only_flip_fails_near_tie():
    candidate = bank(1.1)
    next(iter(candidate.values()))['token_ids'] = [1]
    result = bracket(bank(), bank(), candidate, det_pair_stats, profile='tau')
    assert result['decision'] == 'HOLD'


def test_candidate_compared_with_each_control():
    # Averages would hide this regression against the faster surrounding control.
    result = bracket(bank(), bank(1.2), bank(1.1), det_pair_stats, exact=True)
    assert result['decision'] == 'HOLD'
    assert 'cycle regression >0.5%' in result['reasons']


def test_pair_statistics_preserve_units_and_repeatability():
    a, b = bank(), bank(1.1)
    keys = sorted(a)
    result = det_pair_stats(a, b, keys)
    assert result == det_pair_stats(a, b, keys)
    assert result['ratio'] == pytest.approx(1.1)
    assert result['n'] == 6
    assert result['divergent_prompts'] == 0
    b = copy.deepcopy(b)
    b[keys[0]]['time_source'] = 'different clock'
    with pytest.raises(ValueError, match='identity drift'):
        det_pair_stats(a, b, keys)
