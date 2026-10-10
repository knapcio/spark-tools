# SPDX-License-Identifier: Apache-2.0
import math
import pytest
from common.config import digest, finite, integer, loads, validate_survival
from common.stats import interval_union, paired_bootstrap, quantile


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_strict_json(text):
    with pytest.raises(ValueError):
        loads(text)


@pytest.mark.parametrize("value", [True, "2", math.inf, math.nan, -1])
def test_finite_rejects_invalid(value):
    with pytest.raises(ValueError):
        finite(value, low=0)


def test_integer_and_digest():
    with pytest.raises(ValueError):
        integer(1.0)
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


@pytest.mark.parametrize("values", [[], [.2, .3], [1.1], [math.nan]])
def test_survival_invalid(values):
    with pytest.raises(ValueError):
        validate_survival(values)


def test_statistics():
    assert quantile([1, 3], .25) == 1.5
    assert interval_union([(0, 2), (1, 3), (4, 5)]) == 4
    assert paired_bootstrap([2, 2, 2])["ci95"] == [2, 2]
    assert paired_bootstrap([2, 3], clusters=["a", "a"])["ci95"] is None
