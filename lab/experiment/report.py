# SPDX-License-Identifier: Apache-2.0
"""A hash integrity result is separate from every original tool decision."""
import json
from .adapters import analyze
from .manifest import STATUSES, load_manifest, require


def build_report(path):
    manifest, documents, sha, size = load_manifest(path)
    artifacts = {item["id"]: item for item in manifest["artifacts"]}
    decisions = [analyze(item, artifacts, documents) for item in manifest["analyses"]]
    evidence = {name: [] for name in sorted(STATUSES)}
    for item in manifest.get("evidence", []):
        if item["status"] == "measured":
            # Origin and epistemic status are independent. A fixture/simulation
            # cannot become a measured hardware observation by declaration.
            require(artifacts[item["artifact"]]["origin"] == "recorded",
                    "measured evidence requires recorded origin")
        evidence[item["status"]].append({**item, "basis": "manifest-declaration"})
    result = {
        "schema": "spark-experiment-report", "version": 1,
        "identity": manifest["identity"], "model": manifest["model"], "runtime": manifest["runtime"],
        "arms": manifest["arms"], "identity_basis": "operator-declared",
        "integrity": {"status": "PASS", "manifest_sha256": sha,
                      "artifact_count": len(artifacts), "artifact_bytes": size},
        "artifacts": [{k: v for k, v in item.items() if k != "path"} for item in artifacts.values()],
        "tool_decisions": decisions, "evidence": evidence,
        "limitations": [
            "Integrity verifies explicit local bytes and adapter contracts, not authenticity or hardware execution.",
            "Manifest identity, configuration hashes and evidence classifications are operator declarations.",
            "Fixture, simulation and static origins remain visible; static PASS does not qualify GPU execution.",
            "Each tool retains its own rule and scope. No experiment-wide promotion verdict is produced.",
        ],
    }
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, OverflowError):
        require(False, "nonfinite or excessive derived result")
    return result


def markdown(report):
    lines = ["# Experiment " + report["identity"]["id"], "",
             "Integrity: " + report["integrity"]["status"], "",
             "| Analysis | Tool | Decision | Status | Basis |", "| --- | --- | --- | --- | --- |"]
    for item in report["tool_decisions"]:
        lines.append("| " + " | ".join(str(item.get(key) or "missing") for key in
                                           ("id", "tool", "decision", "status", "basis")) + " |")
    lines.extend(["", "| Artifact | Kind | Origin | SHA-256 |", "| --- | --- | --- | --- |"])
    for item in report["artifacts"]:
        lines.append("| " + " | ".join(item[key] for key in ("id", "kind", "origin", "sha256")) + " |")
    for item in report["tool_decisions"]:
        statuses = sorted({entry["value"] for entry in item.get("original_statuses", [])})
        if statuses:
            lines.extend(["", item["id"] + " original tool statuses:", "", *["- " + text for text in statuses]])
    lines.append("")
    for status, entries in report["evidence"].items():
        lines.extend([status + ": " + str(len(entries)) + " manifest declarations.", ""])
        for entry in entries:
            text = entry["id"]
            if "value" in entry:
                text += " = " + str(entry["value"]) + (" " + entry["unit"] if "unit" in entry else "")
            if "artifact" in entry:
                text += " (artifact " + entry["artifact"] + ")"
            lines.append("- " + text)
        if entries:
            lines.append("")
    lines.extend(["", *report["limitations"], ""])
    return "\n".join(lines)
