# SPDX-License-Identifier: Apache-2.0
"""Exact-cell empirical means with disjoint, explicit holdout requests."""
from statistics import mean
import re

from .core import PINS, canonical, digest, identity, number, opaque, require, sha

METRICS = {"decode_rate": "token/second", "ttft": "second", "latency": "second",
           "prefill": "second", "time_per_output_token": "second/token",
           "acceptance": "ratio", "tokens_per_round": "token/round",
           "client_ttft": "second", "client_latency": "second", "client_decode_rate": "token/second",
           "client_time_per_output_token": "second/token"}


def _cell(identity, row):
    metrics = row["metrics"]
    require(all(k in metrics for k in ("input_tokens", "output_tokens")), "per-request token dimensions required for calibration")
    require(all(isinstance(metrics[k], dict) and metrics[k].get("unit") == "token"
                for k in ("input_tokens", "output_tokens")), "invalid token dimension unit")
    return {"identity": identity,
            "input_tokens": number(metrics["input_tokens"]["value"], integer=True, positive=True),
            "output_tokens": number(metrics["output_tokens"]["value"], integer=True, positive=True)}


def calibrate(observations, split):
    require(isinstance(observations, list) and observations and isinstance(split, dict)
            and split.get("schema") == "spark-observation-split" and type(split.get("version")) is int
            and split["version"] == 1, "unsupported explicit calibration split")
    partitions = []
    for name in ("train_request_ids", "holdout_request_ids"):
        values = split.get(name)
        require(isinstance(values, list) and values, "nonempty train and holdout IDs required")
        values = [opaque(value) for value in values]
        require(len(values) == len(set(values)), "duplicate split request ID")
        partitions.append(set(values))
    train, holdout = partitions
    require(not train & holdout, "train and holdout request IDs overlap")
    requested = split.get("metrics")
    require(isinstance(requested, list) and requested and all(isinstance(key, str) and key in METRICS for key in requested)
            and len(requested) == len(set(requested)), "unsupported calibration metrics")
    rows, sources, cells = {}, [], {}
    source_ids, source_hashes = set(), set()
    for observation in observations:
        require(isinstance(observation, dict) and observation.get("schema") == "spark-observations"
                and type(observation.get("version")) is int and observation["version"] == 1, "unsupported observations for calibration")
        ident = observation.get("identity")
        require(isinstance(ident, dict) and isinstance(ident.get("runtime"), dict)
                and isinstance(ident.get("workload"), dict), "complete calibration identity required")
        adapter = observation.get("adapter")
        require(isinstance(adapter, dict) and isinstance(adapter.get("id"), str) and adapter["id"] in PINS
                and type(adapter.get("version")) is int and adapter["version"] == 1
                and adapter.get("upstream_revision") == PINS[adapter["id"]], "unsupported calibration adapter identity")
        identity(dict(ident, schema="spark-observation-import", version=1, adapter=adapter["id"]), adapter["id"])
        # Copy only importer-owned identity fields; never arbitrary report keys.
        revision = ident["runtime"].get("source_revision")
        require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}(?:\+mia-[0-9a-f]{40})?", revision), "invalid calibration runtime revision")
        safe = {"runtime": {"id": opaque(ident["runtime"].get("id")),
                            "source_revision": revision},
                "config_sha256": sha(ident.get("config_sha256")), "model_sha256": sha(ident.get("model_sha256")),
                "workload": {"id": opaque(ident["workload"].get("id")),
                             "concurrency": number(ident["workload"].get("concurrency"), integer=True, positive=True)},
                "adapter": opaque(observation.get("adapter", {}).get("id"))}
        for key in ("binary_sha256", "image_sha256", "patches_sha256", "observed_settings_sha256"):
            if key in ident["runtime"]:
                safe["runtime"][key] = sha(ident["runtime"][key])
        source_id, source_hash = opaque(ident.get("source_id")), sha(observation.get("source_sha256"))
        require(source_id not in source_ids and source_hash not in source_hashes, "source observation reused in calibration")
        source_ids.add(source_id)
        source_hashes.add(source_hash)
        sources.append({"source_id": source_id, "sha256": source_hash,
                        "observation_sha256": digest(canonical(observation))})
        require(isinstance(observation.get("requests"), list), "request observations required")
        counts = observation.get("counts")
        require(isinstance(counts, dict) and number(counts.get("accepted"), integer=True) == len(observation["requests"]),
                "calibration accepted request count mismatch")
        number(counts.get("rejected"), integer=True)
        for row in observation["requests"]:
            require(isinstance(row, dict) and isinstance(row.get("metrics"), dict), "invalid request observations")
            rid = opaque(row.get("request_id"))
            require(rid not in rows, "request ID reused across source observations")
            rows[rid] = row
            dimensions = _cell(safe, row)
            for name in requested:
                if name not in row["metrics"]:
                    continue
                value = row["metrics"][name]
                require(isinstance(value, dict) and value.get("unit") == METRICS[name], "calibration metric unit mismatch")
                basis = value.get("basis")
                # A hash binds timing instrumentation without exporting free text.
                require(isinstance(basis, str) and 0 < len(basis) <= 128, "calibration timing basis required")
                cell = dict(dimensions, metric=name, unit=METRICS[name], basis_sha256=digest(basis.encode()))
                key = digest(canonical(cell))
                cells.setdefault(key, {"cell": cell, "train": [], "holdout": []})
                if rid in train or rid in holdout:
                    cells[key]["train" if rid in train else "holdout"].append((rid, number(value.get("value"))))
    require(set(rows) == train | holdout, "split must cover exactly the accepted request IDs")
    output = []
    for key, item in sorted(cells.items()):
        fit, validation = item["train"], item["holdout"]
        supported = bool(fit and validation)
        estimate = mean(value for _, value in fit) if fit else None
        errors = []
        if fit:
            for rid, observed in validation:
                errors.append({"request_id": rid, "observed": observed, "predicted": estimate,
                               "signed_error": estimate - observed,
                               "absolute_error": abs(estimate - observed),
                               "absolute_error_percent": abs(estimate - observed) / observed * 100 if observed else None})
        output.append({"cell_id": key, "cell": item["cell"],
                       "status": "supported" if supported else "unavailable",
                       "unavailable_reason": None if supported else "missing-training-or-holdout-cell",
                       "estimate": estimate if supported else None,
                       "train_request_ids": [rid for rid, _ in fit],
                       "holdout_request_ids": [rid for rid, _ in validation], "holdout_errors": errors})
    unavailable = [{"request_id": rid, "metric": name, "status": "unavailable", "reason": "metric-not-observed"}
                   for rid, row in sorted(rows.items()) for name in requested if name not in row["metrics"]]
    return {"schema": "spark-observation-calibration", "version": 1, "method": "exact-cell-mean-v1",
            "split_sha256": digest(canonical(split)), "sources": sources,
            "train_request_ids": sorted(train), "holdout_request_ids": sorted(holdout),
            "cells": output, "unavailable": unavailable,
            "scope": {"extrapolation": "unavailable", "position_survival": "unavailable",
                      "kernel_costs": "unavailable", "glm_cyclesim_inference": "unavailable"}}


def lookup(calibration, cell):
    require(isinstance(calibration, dict) and calibration.get("schema") == "spark-observation-calibration"
            and calibration.get("version") == 1 and calibration.get("method") == "exact-cell-mean-v1",
            "unsupported calibration lookup")
    key = digest(canonical(cell))
    for item in calibration["cells"]:
        if item["cell_id"] == key and item["cell"] == cell and item["status"] == "supported":
            return {"status": "supported", "estimate": item["estimate"], "unit": cell["unit"],
                    "holdout_errors": item["holdout_errors"]}
    return {"status": "unavailable", "estimate": None, "reason": "unsupported-exact-cell"}
