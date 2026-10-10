# SPDX-License-Identifier: Apache-2.0
"""Pure adapters: recompute frozen panel arithmetic; preserve other tool outcomes."""
import hashlib
import os

from .manifest import InvalidExperiment, MAX_MANIFEST_BYTES, read_fd, require


def preregistration(plan, plan_sha, sidecar, tool):
    # Importing CLI modules obtains the unchanged original rules/source list only.
    # No CLI, endpoint, compiler, discovery or subprocess entry point is called.
    if tool == "abharness":
        from abharness.cli import SOURCES, validate_rule as validate
    else:
        from qpanel.cli import SOURCES, validate_config as validate
    require(sidecar == plan_sha, "preregistration sidecar drift")
    current = {}
    for path in SOURCES:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            current[path.name] = hashlib.sha256(read_fd(fd, MAX_MANIFEST_BYTES)).hexdigest()
        finally:
            os.close(fd)
    require(type(plan.get("schema")) is int and plan["schema"] == 1
            and plan.get("sources") == current, "source/preregistration drift")
    config = plan["configuration"]
    require(isinstance(config, dict), "invalid preregistration configuration")
    validate(config)
    return config


def bound_panels(data):
    for arm in ("A", "Aprime", "B"):
        panels = data[arm]
        require(isinstance(panels, list) and len(panels) == 3, "three complete panels required")
        for panel in panels:
            require(isinstance(panel, dict) and isinstance(panel.get("rows"), list)
                    and len(panel["rows"]) == 20, "complete cycle panel required")
            for row in panel["rows"]:
                require(isinstance(row, dict) and type(row.get("i")) is int
                        and row.get("kind") in ("prose", "code"), "invalid cycle row identity")
                require(isinstance(row.get("text_sha256"), str) and bool(row["text_sha256"]),
                        "cycle text hash required")


def abharness(data, sha):
    from abharness.logic import decision
    require(data.get("prereg_sha256") == sha, "panel/preregistration drift")
    bound_panels(data)
    result = decision(data["A"], data["Aprime"], data["B"])
    # Do not export row IDs, prompt text, outputs, supplied decisions or reasons.
    return {"decision": result["decision"], "status": None, "basis": "recomputed",
            "prereg_sha256": sha, "gates": result["gates"], "groups": result["groups"],
            "committed_ratio": result["committed_ratio"]}


def qpanel(runs, cfg, sha):
    from qpanel.logic import decide
    n = cfg.get("items", len(cfg.get("tasks", [])))
    require(type(n) is int and 0 < n <= 10_000, "quality item count limit exceeded")
    for arm, run in zip(("A", "A2", "B"), runs):
        require(run.get("prereg_sha256") == sha and run.get("arm") == arm,
                "arm/preregistration drift")
        rows = run.get("results")
        require(isinstance(rows, list) and len(rows) == n, "preregistered item count drift")
        keys = []
        for row in rows:
            require(isinstance(row, dict) and isinstance(row.get("key"), str)
                    and bool(row["key"]), "invalid quality row identity")
            keys.append(row["key"])
            output = row.get("output")
            require(isinstance(output, dict) and isinstance(output.get("content"), str),
                    "quality output content required")
            require(isinstance(row.get("response_sha256"), str) and bool(row["response_sha256"]),
                    "quality response hash required")
        require(len(set(keys)) == n, "duplicate quality row identity")
        if "tasks" in cfg:
            require(keys == [t["key"] for t in cfg["tasks"]], "preregistered item identity drift")
    result = decide(*runs)
    safe = {key: result[key] for key in ("decision", "A_pass", "A2_pass", "B_pass", "items",
                                        "A_rate", "noise_net_items", "binomial_SE_items", "threshold_items", "truncations")}
    for key in ("A_vs_A2", "A_vs_B"):
        safe[key] = {name: result[key][name] for name in
                     ("pass_to_fail", "fail_to_pass", "pass_changed", "net_loss", "raw_changed", "output_changed")}
    safe.update(status="COMPLETE", basis="recomputed", prereg_sha256=sha)
    return safe


# Only these tool-owned literals can cross the export boundary. Arbitrary status
# strings are rejected rather than redacted into a misleading replacement.
CYCLE_STATUSES = {
    "estimated", "measured", "assumed", "derived", "unknown", "usable-companions-only",
    "measured-time/estimated-traffic", "estimated local derivative",
    "estimated from measured anchors", "measured timing comparison", "measured count comparison",
    "measured paired prompt arm; transfer to thinking/default sampling estimated",
    "measured error of estimated predictions; cohort/instrument-specific",
    "estimated sparkDash-only instrument transfer; RigMark unfitted",
    "estimated independent RigMark T=0 low-effort panel fit; run5 held out; prompt generalization unknown",
    "unknown; no matched instrument/concurrency fit",
    "estimated throughput transfer max-error envelope; not a confidence interval",
    "measured matching capture or estimated calibrated prefix/arm transfer",
    "assumed prior; executed c1 distribution unknown", "derived uniform K2",
    "unknown for empirical paired arm", "measured capture", "estimated capture transfer",
    "assumed occupancy; reuse/pad hygiene prevent extra stale routes",
    "estimated prior or measured expected-work table transfer; empirical arm delta separate",
    "signed capture residual; includes overlap and profiler distortion",
    "derived/assumed", "measured companions",
    "unknown; explicit capture state required before Model use",
    "estimated bytes/time; approximate prompt reference",
    "derived weight bytes / measured time; approximate prompt reference",
    "estimated, assumed expert union; interval containment only",
    "measured fit-counter prefix survival",
    "assumed structural prior only; current executed-pass distribution unknown",
    "estimated transfer of T=0 structured counter survival",
    "assumed structured survival; JSON counter capture missing",
    "estimated transfer of fitted thinking prefix survival", "assumed historical prior",
    "estimated rescaled historical profile; current per-kernel capture missing",
    "historical fixed K2 expected graph cost divided by two; current price unknown",
    "measured aggregate prior; all category attribution estimated",
    "measured aggregate prior; current fit survival already includes effect",
    "estimated transfer from measured capture",
    "estimated historical head fingerprint; weight bytes derived",
    "assumed heuristic; not a cache measurement",
}
CYCLE_METRICS = {"cycle_ms", "committed_per_cycle", "tok_s_per_stream", "aggregate_tok_s",
                 "measured", "predicted", "error_pct", "max_abs_pct", "median_abs_pct",
                 "n_anchors", "cycle_saving_ms", "empirical_cycle_delta_ms",
                 "measured_prose_gain_pct", "predicted_prose_gain_pct"}


def cyclesim(data):
    if isinstance(data, dict) and "schema" in data:
        require(type(data["schema"]) is int and data["schema"] == 1, "unsupported cyclesim native schema")
    rows = data if isinstance(data, list) else [data]
    require(1 <= len(rows) <= 4096 and all(isinstance(row, dict) for row in rows), "invalid cyclesim report")
    statuses, metrics = [], []
    stack = list(rows)
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "decision":
                    require(item in ("PASS", "FAIL", "HOLD", "KILL", "INDETERMINATE"), "unsupported cyclesim decision")
                    statuses.append({"field": "decision", "value": item})
                elif key == "status" or key.endswith("_status"):
                    # Category statuses can be a dictionary of fixed literal values.
                    values = item.values() if isinstance(item, dict) else [item]
                    for text in values:
                        require(isinstance(text, str) and text in CYCLE_STATUSES, "unsupported cyclesim status")
                        statuses.append({"field": "status" if key == "status" else "component_status", "value": text})
                elif key in CYCLE_METRICS and type(item) in (int, float):
                    metrics.append({"metric": key, "value": item})
                elif key in CYCLE_METRICS and isinstance(item, list):
                    require(len(item) <= 16 and all(type(x) in (int, float) for x in item), "invalid cyclesim metric interval")
                    metrics.append({"metric": key, "value": item})
                elif isinstance(item, (dict, list)):
                    stack.append(item)
        elif isinstance(value, list):
            stack.extend(value)
    require(statuses or metrics, "unrecognized cyclesim report")
    require(len(statuses) <= 8192 and len(metrics) <= 8192, "cyclesim report summary limit exceeded")
    return {"decision": data.get("decision") if isinstance(data, dict) else None,
            "status": data.get("status") if isinstance(data, dict) else None,
            "basis": "preserved", "original_statuses": statuses, "metrics": metrics}


def static_report(data, kind):
    require(type(data.get("schema")) is int and data["schema"] == 1,
            "unsupported tool report schema")
    require(data.get("status") in ("PASS", "FAIL"), "invalid tool report status")
    result = {"decision": None, "status": data["status"], "basis": "preserved"}
    if kind == "gb10check.report":
        require(data.get("mode") == "STATIC_NO_GPU", "unsupported GB10 report mode")
        result["mode"] = data["mode"]
        rows = data.get("resources", data.get("entries", []))
        require(isinstance(rows, list) and len(rows) <= 4096, "invalid GB10 report entries")
        result["entry_statuses"] = []
        for row in rows:
            require(isinstance(row, dict) and row.get("status") in ("PASS", "FAIL"), "invalid GB10 entry status")
            result["entry_statuses"].append(row["status"])
        result["original_statuses"] = []
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, val in item.items():
                    if key == "status":
                        require(val in ("PASS", "FAIL", "estimated", "unknown", "observed"), "unsupported GB10 component status")
                        result["original_statuses"].append({"field": "status", "value": val})
                    elif isinstance(val, (dict, list)):
                        stack.append(val)
            elif isinstance(item, list):
                stack.extend(item)
        require(len(result["original_statuses"]) <= 8192, "GB10 report summary limit exceeded")
        # Resource estimates remain estimates even when the static gate passes.
        result["evidence_scope"] = "static-only"
    else:
        if kind.startswith("exl3"):
            require(data.get("format") == "EXL3", "EXL3 report identity mismatch")
        elif kind.startswith("exlr8"):
            require(data.get("format") == "EXLR8", "EXLR8 report identity mismatch")
        else:
            require(data.get("runtime") == "TensorFold", "TensorFold report identity mismatch")
        result["evidence_scope"] = "supplementary-only"
    return result


def analyze(analysis, artifacts, documents):
    tool, inputs = analysis["tool"], analysis["inputs"]
    if tool == "abharness":
        expected = {"prereg": "abharness.prereg", "prereg_digest": "prereg.digest", "panels": "abharness.panels"}
    elif tool == "qpanel":
        expected = {"prereg": "qpanel.prereg", "prereg_digest": "prereg.digest",
                    "A": "qpanel.run", "A2": "qpanel.run", "B": "qpanel.run"}
    else:
        expected = {"report": {"cyclesim": "cyclesim.report", "gb10check": "gb10check.report",
                               "exl3": "exl3.report", "exlr8": "exlr8.report"}.get(tool)}
    require(set(inputs) == set(expected), "unexpected or missing adapter inputs")
    require(len(set(inputs.values())) == len(inputs), "duplicate adapter input reference")
    for role, ref in inputs.items():
        kind = artifacts[ref]["kind"]
        if tool == "tensorfold":
            require(kind in ("tensorfold.source", "tensorfold.probe"), "adapter artifact kind mismatch")
        else:
            require(kind == expected[role], "adapter artifact kind mismatch")
        if tool == "qpanel" and role in ("A", "A2", "B"):
            require(artifacts[ref].get("arm") == role, "manifest quality arm mismatch")
        if tool in ("abharness", "qpanel"):
            require(artifacts[ref]["origin"] in ("fixture", "recorded", "static"), "invalid panel evidence origin")
    try:
        if tool in ("abharness", "qpanel"):
            sha = artifacts[inputs["prereg"]]["sha256"]
            cfg = preregistration(documents[inputs["prereg"]], sha,
                                  documents[inputs["prereg_digest"]], tool)
            if tool == "abharness":
                result = abharness(documents[inputs["panels"]], sha)
            else:
                result = qpanel([documents[inputs[arm]] for arm in ("A", "A2", "B")], cfg, sha)
        elif tool == "cyclesim":
            result = cyclesim(documents[inputs["report"]])
        else:
            result = static_report(documents[inputs["report"]], artifacts[inputs["report"]]["kind"])
        return {"id": analysis["id"], "tool": tool, "artifacts": list(inputs.values()), **result}
    except InvalidExperiment:
        raise
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, OSError, OverflowError):
        raise InvalidExperiment("invalid or inconsistent adapter evidence") from None
