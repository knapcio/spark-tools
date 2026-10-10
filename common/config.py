# SPDX-License-Identifier: Apache-2.0
"""Strict JSON configuration, reproducible digests and bounded arithmetic."""
import hashlib
import json
import math
from pathlib import Path


def finite(value, label="value", low=-math.inf, high=math.inf):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be finite in [{low}, {high}]")
    return value


def integer(value, label="value", low=0, high=2**63 - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def loads(text):
    def invalid(value):
        raise ValueError("non-finite JSON constant: " + value)
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid)


def load(path):
    value = loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def dump(value, path=None):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path:
        Path(path).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return value


def validate_survival(values):
    if not isinstance(values, list) or not values:
        raise ValueError("nonempty prefix survival list required")
    for value in values:
        finite(value, "prefix survival", 0, 1)
    if any(a < b for a, b in zip(values, values[1:])):
        raise ValueError("prefix survival must be nonincreasing")
    return values
