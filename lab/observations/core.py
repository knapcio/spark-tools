# SPDX-License-Identifier: Apache-2.0
"""Bounded local I/O and the versioned observation boundary."""
import hashlib
import json
import math
import os
import re
import stat

MAX_BYTES = 32 * 1024 * 1024
MAX_REQUESTS = 100_000
PINS = {
    "vllm-bench-serve-v1": "276fbcff2717bd934cfa37c8a2e4c391f3e7237b",
    "tensorfold-native-v1": "db281878ddb836fd0df510d8771ecb7e0fe47d26",
    "tensorfold-prometheus-v1": "db281878ddb836fd0df510d8771ecb7e0fe47d26",
}


class InvalidObservation(ValueError):
    """A fixed diagnostic; input content must never enter an exception."""


def require(condition, message):
    if not condition:
        raise InvalidObservation(message)


def number(value, *, integer=False, positive=False):
    require(type(value) in (int, float) and abs(value) <= 1e18 and math.isfinite(value), "invalid finite numeric field")
    require(value > 0 if positive else value >= 0, "numeric field outside supported range")
    require(not integer or value == int(value), "nonintegral count")
    return int(value) if integer else value


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def bad(_):
        raise InvalidObservation("nonfinite JSON constant")
    def finite(text):
        value = float(text)
        require(math.isfinite(value), "nonfinite JSON number")
        return value
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=bad, parse_float=finite)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise InvalidObservation("malformed JSON") from exc


def read_local(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_BYTES, "input must be a bounded regular file")
            with os.fdopen(os.dup(fd), "rb") as handle:
                data = handle.read(MAX_BYTES + 1)
            after = os.fstat(fd)
            require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                    and len(data) == before.st_size, "input changed during read")
            return data
        finally:
            os.close(fd)
    except OSError as exc:
        raise InvalidObservation("local input cannot be read safely") from exc


def opaque(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}", value), "invalid opaque identity")
    return value


def sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256 identity")
    return value


def identity(metadata, adapter):
    require(isinstance(metadata, dict) and metadata.get("schema") == "spark-observation-import"
            and type(metadata.get("version")) is int and metadata["version"] == 1, "unsupported import metadata")
    require(metadata.get("adapter") == adapter, "adapter metadata mismatch")
    runtime = metadata.get("runtime")
    require(isinstance(runtime, dict), "runtime identity required")
    expected = "vllm" if adapter.startswith("vllm") else "tensorfold"
    runtime_id = runtime.get("id")
    allowed_ids = {"vllm"} if expected == "vllm" else {"tensorfold", "tensorfold-native-zig"}
    revision = runtime.get("source_revision")
    require(runtime_id in allowed_ids and isinstance(revision, str)
            and re.fullmatch(re.escape(PINS[adapter]) + r"(?:\+mia-[0-9a-f]{40})?", revision), "runtime source pin mismatch")
    workload = metadata.get("workload")
    require(isinstance(workload, dict), "workload identity required")
    result = {"source_id": opaque(metadata.get("source_id")),
              "runtime": {"id": runtime_id, "source_revision": revision},
              "config_sha256": sha(metadata.get("config_sha256")),
              "model_sha256": sha(metadata.get("model_sha256")),
              "workload": {"id": opaque(workload.get("id")),
                           "concurrency": number(workload.get("concurrency"), integer=True, positive=True)}}
    for key in ("binary_sha256", "image_sha256", "patches_sha256"):
        if key in runtime:
            result["runtime"][key] = sha(runtime[key])
    return result


def request_ids(metadata, n):
    ids = metadata.get("request_ids")
    require(isinstance(ids, list) and len(ids) == n and 0 < n <= MAX_REQUESTS, "complete request IDs required")
    ids = [opaque(value) for value in ids]
    require(len(set(ids)) == n, "duplicate request identity")
    return ids


def metric(value, unit, basis):
    return {"value": number(value), "unit": unit, "basis": basis}


def report(adapter, metadata, source, recognized, missing, requests, aggregates, rejected=0):
    return {"schema": "spark-observations", "version": 1,
            "adapter": {"id": adapter, "version": 1, "upstream_revision": PINS[adapter]},
            "identity": identity(metadata, adapter), "source_sha256": digest(source),
            "recognized_fields": sorted(set(recognized)), "missing_fields": sorted(set(missing)),
            "counts": {"accepted": len(requests), "rejected": rejected},
            "requests": requests, "aggregates": aggregates,
            "scope": {"position_survival": "unavailable", "kernel_costs": "unavailable",
                      "glm_cyclesim_inference": "unavailable"}}
