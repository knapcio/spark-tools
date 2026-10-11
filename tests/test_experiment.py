# SPDX-License-Identifier: Apache-2.0
"""Offline experiment contract, existing arithmetic, and adversarial I/O tests."""
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

import pytest

from common.lab_cli import main as lab_main
from lab.experiment import build_report
from lab.experiment.cli import main
from lab.experiment.manifest import InvalidExperiment, SCHEMAS, strict_json
from lab.experiment.report import markdown

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest():
    return {"schema": "spark-experiment", "version": 1, "identity": {"id": "toy"},
            "model": {"id": "toy-model", "revision": "unknown"},
            "runtime": {"id": "toy-runtime", "revision": "unknown"},
            "arms": [{"id": arm, "config_sha256": SHA} for arm in ("A", "Aprime", "A2", "B")],
            "artifacts": [], "analyses": [], "evidence": []}


def artifact(tmp_path, value, *, name="static", kind="gb10check.report", origin="static", arm=None):
    file = tmp_path / (name + (".sha256" if kind == "prereg.digest" else ".json"))
    if kind == "prereg.digest":
        file.write_text(value + "\n")
        sha = hashlib.sha256(file.read_bytes()).hexdigest()
    else:
        sha = write_json(file, value)
    item = {"id": name, "kind": kind, "path": file.name, "sha256": sha,
            "schema": SCHEMAS[kind], "origin": origin}
    if arm:
        item["arm"] = arm
    return item


def save(tmp_path, value):
    path = tmp_path / "experiment.json"
    write_json(path, value)
    return path


def static_manifest(tmp_path, status="PASS"):
    value = manifest()
    example = json.loads((ROOT / "examples/gb10check/expected-pass.json").read_text())
    example["status"] = status
    # These fields must never appear in the generated JSON or Markdown.
    example.update(prompt="raw-prompt-private", response="raw-response-private",
                   endpoint="https://private-host.invalid/v1", source_path="/absolute/private/location",
                   **{"sec" + "ret": "private-credential"})
    value["artifacts"] = [artifact(tmp_path, example)]
    value["analyses"] = [{"id": "kernel", "tool": "gb10check", "inputs": {"report": "static"}}]
    return value


def panel_manifest(tmp_path, tool):
    value = manifest()
    base = ROOT / "examples" / tool
    plan = json.loads((base / "prereg.json").read_text())
    p = artifact(tmp_path, plan, name="plan", kind=tool + ".prereg")
    sidecar = artifact(tmp_path, p["sha256"], name="digest", kind="prereg.digest")
    value["artifacts"] = [p, sidecar]
    inputs = {"prereg": "plan", "prereg_digest": "digest"}
    if tool == "abharness":
        panels = json.loads((base / "panels.json").read_text())
        panels["prereg_sha256"] = p["sha256"]
        panels["decision"] = "invented-PASS"
        value["artifacts"].append(artifact(tmp_path, panels, name="panels", kind="abharness.panels", origin="fixture"))
        inputs["panels"] = "panels"
    else:
        for arm in ("A", "A2", "B"):
            run = json.loads((base / (arm + ".json")).read_text())
            run.update(prereg_sha256=p["sha256"], decision="invented-PASS")
            value["artifacts"].append(artifact(tmp_path, run, name=arm, kind="qpanel.run", origin="fixture", arm=arm))
            inputs[arm] = arm
    value["analyses"] = [{"id": "panel", "tool": tool, "inputs": inputs}]
    return value


def replace_document(tmp_path, value, name, mutate):
    entry = next(x for x in value["artifacts"] if x["id"] == name)
    path = tmp_path / entry["path"]
    document = json.loads(path.read_text())
    mutate(document)
    entry["sha256"] = write_json(path, document)


def rebind_plan(tmp_path, value):
    sha = value["artifacts"][0]["sha256"]
    sidecar = value["artifacts"][1]
    path = tmp_path / sidecar["path"]
    path.write_text(sha + "\n")
    sidecar["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for entry in value["artifacts"][2:]:
        replace_document(tmp_path, value, entry["id"], lambda x: x.update(prereg_sha256=sha))


def test_static_integrity_and_original_decisions_separate(tmp_path):
    report = build_report(save(tmp_path, static_manifest(tmp_path, "FAIL")))
    assert report["integrity"]["status"] == "PASS"
    outcome = report["tool_decisions"][0]
    assert outcome["status"] == "FAIL" and outcome["decision"] is None
    assert outcome["basis"] == "preserved" and outcome["mode"] == "STATIC_NO_GPU"
    assert outcome["entry_statuses"] == ["PASS"]
    assert "decision" not in report and "status" not in report
    assert report["identity_basis"] == "operator-declared"


def test_redaction_and_no_implicit_io(tmp_path, monkeypatch):
    path = save(tmp_path, static_manifest(tmp_path))
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected discovery or external execution")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(os, "listdir", forbidden)
    monkeypatch.setattr(os, "scandir", forbidden)
    report = build_report(path)
    text = json.dumps(report) + markdown(report)
    for forbidden_text in ("raw-prompt-private", "raw-response-private", "private-host", "private-credential",
                           "/absolute/private/location", str(tmp_path), "static.json"):
        assert forbidden_text not in text


@pytest.mark.parametrize("tool", ["abharness", "qpanel"])
def test_original_fixtures_recomputed(tmp_path, tool):
    report = build_report(save(tmp_path, panel_manifest(tmp_path, tool)))
    actual = report["tool_decisions"][0]
    expected = json.loads((ROOT / "examples" / tool / "expected-decision.json").read_text())
    assert actual["basis"] == "recomputed" and actual["decision"] == expected["decision"]
    if tool == "abharness":
        assert actual["groups"] == expected["groups"] and actual["gates"] == expected["gates"]
    else:
        for key in ("A_pass", "A2_pass", "B_pass", "items", "threshold_items", "truncations"):
            assert actual[key] == expected[key]
    assert "invented-PASS" not in json.dumps(report)


@pytest.mark.parametrize("tool", ["abharness", "qpanel"])
@pytest.mark.parametrize("fault", ["source", "sidecar", "prereg", "kind", "duplicate-input"])
def test_preregistration_and_input_binding(tmp_path, tool, fault):
    value = panel_manifest(tmp_path, tool)
    if fault == "source":
        replace_document(tmp_path, value, "plan", lambda x: x["sources"].update({"logic.py": SHA}))
        rebind_plan(tmp_path, value)
    elif fault == "sidecar":
        entry = value["artifacts"][1]
        path = tmp_path / entry["path"]
        path.write_text(SHA + "\n")
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif fault == "prereg":
        name = "panels" if tool == "abharness" else "B"
        replace_document(tmp_path, value, name, lambda x: x.update(prereg_sha256=SHA))
    elif fault == "kind":
        entry = value["artifacts"][-1]
        entry.update(kind="gb10check.report", schema="gb10check-report-v1")
    else:
        value["analyses"][0]["inputs"]["prereg_digest"] = "plan"
    with pytest.raises(InvalidExperiment):
        build_report(save(tmp_path, value))


@pytest.mark.parametrize("fault", ["row-duplicate", "row-missing", "bool-index", "negative", "wrong-rule"])
def test_cycle_panel_contract(tmp_path, fault):
    value = panel_manifest(tmp_path, "abharness")
    def mutate(data):
        rows = data["B"][0]["rows"]
        if fault == "row-duplicate": rows[1] = dict(rows[0])
        elif fault == "row-missing": rows.pop()
        elif fault == "bool-index": rows[0]["i"] = False
        else: rows[0]["cycle_ms"] = -1
    if fault == "wrong-rule":
        replace_document(tmp_path, value, "plan", lambda x: x["configuration"].update(prose_saving_ms=0))
        rebind_plan(tmp_path, value)
    else:
        replace_document(tmp_path, value, "panels", mutate)
    with pytest.raises(InvalidExperiment):
        build_report(save(tmp_path, value))


@pytest.mark.parametrize("fault", ["duplicate", "arm", "count", "truncated", "score", "content", "manifest-arm", "status"])
def test_quality_panel_contract(tmp_path, fault):
    value = panel_manifest(tmp_path, "qpanel")
    def mutate(data):
        rows = data["results"]
        if fault == "duplicate": rows[1]["key"] = rows[0]["key"]
        elif fault == "arm": data["arm"] = "A"
        elif fault == "count": rows.pop()
        elif fault == "truncated": rows[0]["truncated"] = 0
        elif fault == "score": rows[0]["pass"] = 1
        elif fault == "content": rows[0]["output"]["content"] = None
        elif fault == "status": data["status"] = "PARTIAL"
    if fault == "manifest-arm": value["artifacts"][-1]["arm"] = "A"
    else: replace_document(tmp_path, value, "B", mutate)
    with pytest.raises(InvalidExperiment):
        build_report(save(tmp_path, value))


@pytest.mark.parametrize("tool", ["abharness", "qpanel"])
def test_supplied_pass_cannot_override_failure(tmp_path, tool):
    value = panel_manifest(tmp_path, tool)
    if tool == "abharness":
        def fail(data):
            for panel in data["B"]:
                for row in panel["rows"]: row["cycle_ms"] = 100
        replace_document(tmp_path, value, "panels", fail)
    else:
        def fail(data):
            for row in data["results"]: row["pass"] = False
        replace_document(tmp_path, value, "B", fail)
    assert build_report(save(tmp_path, value))["tool_decisions"][0]["decision"] in ("KILL", "FAIL")


@pytest.mark.parametrize("example", ["expected-prediction.json", "expected-validation.json", "expected-comparison.json"])
def test_cyclesim_existing_outputs_and_scopes(tmp_path, example):
    value = manifest()
    doc = json.loads((ROOT / "examples/cyclesim" / example).read_text())
    value["artifacts"] = [artifact(tmp_path, doc, kind="cyclesim.report", origin="simulation")]
    value["analyses"] = [{"id": "model", "tool": "cyclesim", "inputs": {"report": "static"}}]
    report = build_report(save(tmp_path, value))
    result = report["tool_decisions"][0]
    assert result["basis"] == "preserved" and result["decision"] is None
    assert report["artifacts"][0]["origin"] == "simulation"
    assert result["metrics"]
    if example == "expected-prediction.json":
        assert {x["value"] for x in result["original_statuses"]} >= {"estimated", "assumed prior; executed c1 distribution unknown"}


@pytest.mark.parametrize("path", ["/absolute.json", "../outside.json", "nested/../static.json", "./static.json",
                                  "nested//static.json", "C:/static.json", "nested\\static.json", "static.json\n"])
def test_artifact_path_restrictions(tmp_path, path):
    value = static_manifest(tmp_path)
    value["artifacts"][0]["path"] = path
    with pytest.raises(InvalidExperiment): build_report(save(tmp_path, value))


@pytest.mark.parametrize("target", ["file", "directory", "manifest", "hardlink", "fifo"])
def test_unsafe_files_refused_without_blocking(tmp_path, target):
    value = static_manifest(tmp_path)
    source = tmp_path / "static.json"
    if target == "file":
        source.rename(tmp_path / "original.json")
        source.symlink_to("original.json")
    elif target == "directory":
        (tmp_path / "actual").mkdir()
        source.rename(tmp_path / "actual/static.json")
        (tmp_path / "linked").symlink_to("actual", target_is_directory=True)
        value["artifacts"][0]["path"] = "linked/static.json"
    elif target == "hardlink":
        os.link(source, tmp_path / "alias.json")
        entry = copy.deepcopy(value["artifacts"][0]); entry.update(id="alias", path="alias.json")
        value["artifacts"].append(entry)
    elif target == "fifo":
        source.unlink(); os.mkfifo(source)
    path = save(tmp_path, value)
    if target == "manifest":
        path.rename(tmp_path / "original-manifest.json")
        path.symlink_to("original-manifest.json")
    with pytest.raises(InvalidExperiment): build_report(path)


@pytest.mark.parametrize("fault", ["digest", "version", "unknown", "duplicate-id", "duplicate-path", "duplicate-arm",
                                  "origin", "schema", "unknown-ref", "secret-identity", "analysis-tool"])
def test_manifest_validation(tmp_path, fault):
    value = static_manifest(tmp_path)
    if fault == "digest": value["artifacts"][0]["sha256"] = SHA
    elif fault == "version": value["version"] = True
    elif fault == "unknown": value["endpoint"] = "private"
    elif fault == "duplicate-id":
        entry = dict(value["artifacts"][0]); entry["path"] = "other.json"; value["artifacts"].append(entry)
    elif fault == "duplicate-path":
        entry = dict(value["artifacts"][0]); entry["id"] = "other"; value["artifacts"].append(entry)
    elif fault == "duplicate-arm": value["arms"].append(dict(value["arms"][0]))
    elif fault == "origin": value["artifacts"][0]["origin"] = "live"
    elif fault == "schema": value["artifacts"][0]["schema"] = "v999"
    elif fault == "unknown-ref": value["analyses"][0]["inputs"]["report"] = "missing"
    elif fault == "secret-identity": value["identity"]["id"] = "https://private.invalid"
    else: value["analyses"][0]["tool"] = []
    with pytest.raises(InvalidExperiment): build_report(save(tmp_path, value))


@pytest.mark.parametrize("data", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}',
                                  b'{"x":-Infinity}', b'"\xff"', b'[' * 40 + b'0' + b']' * 40])
def test_strict_json(data):
    with pytest.raises(InvalidExperiment): strict_json(data)


@pytest.mark.parametrize("limit", ["manifest", "artifact", "total", "count", "nodes"])
def test_resource_bounds(tmp_path, monkeypatch, limit):
    from lab.experiment import manifest as contract
    value = static_manifest(tmp_path)
    if limit == "count":
        monkeypatch.setattr(contract, "MAX_ARTIFACTS", 0)
    elif limit == "nodes": monkeypatch.setattr(contract, "MAX_NODES", 2)
    else: monkeypatch.setattr(contract, "MAX_" + limit.upper() + "_BYTES", 20)
    with pytest.raises(InvalidExperiment): build_report(save(tmp_path, value))


def test_file_change_during_read_refused(tmp_path, monkeypatch):
    from lab.experiment import manifest as contract
    path = save(tmp_path, static_manifest(tmp_path))
    original = os.read
    def changed(fd, size):
        data = original(fd, size)
        if data:
            stat = os.fstat(fd)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        return data
    monkeypatch.setattr(contract.os, "read", changed)
    with pytest.raises(InvalidExperiment, match="changed"): build_report(path)


def test_evidence_classes_and_measured_fixture_refusal(tmp_path):
    value = static_manifest(tmp_path)
    value["evidence"] = [{"id": status, "status": status} for status in ("assumed", "missing", "operator-declared")]
    value["evidence"].append({"id": "estimate", "status": "estimated", "artifact": "static", "value": 2.5, "unit": "ms"})
    report = build_report(save(tmp_path, value))
    assert set(report["evidence"]) == {"measured", "estimated", "assumed", "missing", "operator-declared"}
    value["evidence"].append({"id": "observed", "status": "measured", "artifact": "static", "value": 1})
    with pytest.raises(InvalidExperiment, match="recorded"): build_report(save(tmp_path, value))
    value["artifacts"][0]["origin"] = "recorded"
    assert build_report(save(tmp_path, value))["evidence"]["measured"][0]["basis"] == "manifest-declaration"


@pytest.mark.parametrize("kind,extra", [("exl3.report", {"format": "EXL3"}),
                                         ("exlr8.report", {"format": "EXLR8"}),
                                         ("tensorfold.source", {"runtime": "TensorFold"}),
                                         ("tensorfold.probe", {"runtime": "TensorFold"})])
def test_supplementary_reports_never_promote(tmp_path, kind, extra):
    value = manifest()
    value["artifacts"] = [artifact(tmp_path, {"schema": 1, "status": "PASS", **extra}, kind=kind)]
    tool = kind.split(".")[0]
    value["analyses"] = [{"id": "supplement", "tool": tool, "inputs": {"report": "static"}}]
    result = build_report(save(tmp_path, value))["tool_decisions"][0]
    assert result["status"] == "PASS" and result["evidence_scope"] == "supplementary-only"


def test_cli_delegation_and_create_only_outputs(tmp_path, capsys):
    path = save(tmp_path, static_manifest(tmp_path, "FAIL"))
    assert lab_main(["experiment", "check", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["integrity"]["status"] == "PASS"
    out, md = tmp_path / "report.json", tmp_path / "report.md"
    assert main(["report", str(path), "--out", str(out), "--markdown", str(md)]) == 0
    assert json.loads(out.read_text())["tool_decisions"][0]["status"] == "FAIL"
    assert "static.json" not in md.read_text()
    assert main(["report", str(path), "--out", str(out), "--markdown", str(md)]) == 0
    fresh = tmp_path / "fresh.json"
    assert main(["report", str(path), "--out", str(fresh), "--markdown", str(md)]) == 0
    assert fresh.read_bytes() == out.read_bytes()
    for args in (["--out", str(path)],):
        with pytest.raises(SystemExit) as error: main(["report", str(path), *args])
        assert error.value.code == 2


def test_report_crash_between_secondary_and_primary_is_retryable(tmp_path, monkeypatch):
    from lab.experiment import cli
    path = save(tmp_path, static_manifest(tmp_path, "FAIL"))
    out, md = tmp_path / "report.json", tmp_path / "report.md"
    real = cli.publish_bytes
    calls = 0

    def stop_before_primary(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic interruption")
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "publish_bytes", stop_before_primary)
    with pytest.raises(SystemExit) as error:
        cli.main(["report", str(path), "--out", str(out), "--markdown", str(md)])
    assert error.value.code == 2
    assert md.is_file() and not out.exists()

    monkeypatch.setattr(cli, "publish_bytes", real)
    assert cli.main(["report", str(path), "--out", str(out), "--markdown", str(md)]) == 0
    assert json.loads(out.read_text())["integrity"]["status"] == "PASS"


def test_report_conflict_never_unlinks_existing_canonical(tmp_path):
    path = save(tmp_path, static_manifest(tmp_path, "FAIL"))
    out, md = tmp_path / "report.json", tmp_path / "report.md"
    out.write_text("existing primary")
    md.write_text("existing secondary")
    with pytest.raises(SystemExit) as error:
        main(["report", str(path), "--out", str(out), "--markdown", str(md)])
    assert error.value.code == 2
    assert out.read_text() == "existing primary"
    assert md.read_text() == "existing secondary"


def test_report_rejects_shared_primary_and_secondary_destination(tmp_path):
    path = save(tmp_path, static_manifest(tmp_path, "FAIL"))
    shared = tmp_path / "report"
    with pytest.raises(SystemExit) as error:
        main(["report", str(path), "--out", str(shared), "--markdown", str(shared)])
    assert error.value.code == 2
    assert not shared.exists()


def test_cli_diagnostic_does_not_echo_private_artifact_path(tmp_path, capsys):
    value = static_manifest(tmp_path)
    value["artifacts"][0]["path"] = "private-credential-location.json"
    with pytest.raises(SystemExit): main(["check", str(save(tmp_path, value))])
    assert "private-credential-location" not in capsys.readouterr().err


@pytest.mark.parametrize("tool", ["abharness", "qpanel"])
def test_panel_adapters_never_call_transport_or_discovery(tmp_path, monkeypatch, tool):
    value = panel_manifest(tmp_path, tool)
    if tool == "qpanel":
        replace_document(tmp_path, value, "plan", lambda x: x["configuration"].update(
            endpoints={"A": "https://private-target.invalid/v1"}))
        rebind_plan(tmp_path, value)
    def forbidden(*args, **kwargs): raise AssertionError("external execution attempted")
    import common.endpoint
    monkeypatch.setattr(common.endpoint, "HTTP", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "scandir", forbidden)
    report = build_report(save(tmp_path, value))
    assert "private-target" not in json.dumps(report)


def test_checked_example(tmp_path):
    example = ROOT / "examples/experiment/manifest.json"
    report = build_report(example)
    assert report["integrity"]["artifact_count"] == 2
    assert {a["origin"] for a in report["artifacts"]} == {"fixture", "simulation"}
    assert "assumed prior; executed c1 distribution unknown" in markdown(report)
    out, md = tmp_path / "report.json", tmp_path / "report.md"
    assert lab_main(["experiment", "report", str(example), "--out", str(out), "--markdown", str(md)]) == 0


@pytest.mark.parametrize("identifier", ["192." + "168.1.1", "sk-" + "x" * 20, "hf_" + "x" * 20])
def test_export_identifiers_refuse_endpoint_and_credential_patterns(tmp_path, identifier):
    value = static_manifest(tmp_path)
    value["identity"]["id"] = identifier
    with pytest.raises(InvalidExperiment): build_report(save(tmp_path, value))


def test_unsupported_cyclesim_native_schema(tmp_path):
    value = manifest()
    value["artifacts"] = [artifact(tmp_path, {"schema": 2, "status": "estimated"}, kind="cyclesim.report")]
    value["analyses"] = [{"id": "estimate", "tool": "cyclesim", "inputs": {"report": "static"}}]
    with pytest.raises(InvalidExperiment): build_report(save(tmp_path, value))
