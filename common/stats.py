# SPDX-License-Identifier: Apache-2.0
"""Paired descriptive statistics; intervals cover sampling uncertainty only."""
import math
import random
import statistics
from common.config import finite, integer


def quantile(values, probability):
    values = sorted(values)
    finite(probability, "probability", 0, 1)
    if not values:
        raise ValueError("nonempty observations required")
    for value in values:
        finite(value)
    pos = (len(values) - 1) * probability
    low, high = math.floor(pos), math.ceil(pos)
    return values[low] + (values[high] - values[low]) * (pos - low)


def summary(values):
    if not values:
        raise ValueError("nonempty observations required")
    return {"n": len(values), "mean": statistics.fmean(values),
            "median": quantile(values, .5), "p95": quantile(values, .95)}


def paired_bootstrap(values, samples=2000, seed=1729, clusters=None):
    """Resample paired deltas, optionally whole independent request cohorts."""
    integer(samples, "bootstrap samples", 200, 100000)
    if not values or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError("finite paired observations required")
    groups = [[value] for value in values]
    if clusters is not None:
        if len(clusters) != len(values):
            raise ValueError("one cluster per observation required")
        grouped = {}
        for key, value in zip(clusters, values):
            grouped.setdefault(key, []).append(value)
        groups = list(grouped.values())
    result = {"mean": statistics.fmean(values), "ci95": None, "n": len(values),
              "clusters": len(groups), "scope": "sampling uncertainty only"}
    if len(groups) < 2:
        return result
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        selected = rng.choices(groups, k=len(groups))
        means.append(sum(map(sum, selected)) / sum(map(len, selected)))
    result["ci95"] = [quantile(means, .025), quantile(means, .975)]
    return result


def interval_union(intervals):
    """Adapted from knapcio's CPU trace utilities (Apache-2.0)."""
    edge, total = -math.inf, 0.0
    for start, end in sorted(intervals):
        finite(start); finite(end)
        if end < start:
            raise ValueError("interval end precedes start")
        total += max(0, end - max(start, edge))
        edge = max(edge, end)
    return total
