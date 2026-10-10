# SPDX-License-Identifier: Apache-2.0
"""The spark-experiment v1 contract. Paths are relative to the manifest directory."""
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

MAX_MANIFEST_BYTES = 256 * 1024
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_ARTIFACTS = 64
MAX_NODES = 200_000
ORIGINS = {"fixture", "recorded", "simulation", "static"}
STATUSES = {"measured", "estimated", "assumed", "missing", "operator-declared"}
SCHEMAS = {
    "abharness.prereg": "abharness-prereg-v1",
    "abharness.panels": "abharness-panels-v1",
    "qpanel.prereg": "qpanel-prereg-v1",
    "qpanel.run": "qpanel-run-v1",
    "prereg.digest": "sha256-text-v1",
    "cyclesim.report": "cyclesim-report-v1",
    "gb10check.report": "gb10check-report-v1",
    "exl3.report": "exl3-report-v1",
    "exlr8.report": "exlr8-report-v1",
    "tensorfold.source": "tensorfold-source-v1",
    "tensorfold.probe": "tensorfold-probe-v1",
}
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
SECRET_PREFIXES = ("sk-", "hf_", "ghp_", "gho_", "ghu_", "ghs_", "github_pat_", "AIza", "AKIA", "ASIA")


class InvalidExperiment(ValueError):
    """Messages deliberately contain no user-controlled paths or document text."""


def require(condition, message):
    if not condition:
        raise InvalidExperiment(message)


def token(value):
    require(isinstance(value, str) and TOKEN.fullmatch(value), "invalid opaque identifier")
    require(not value.startswith(SECRET_PREFIXES) and not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", value),
            "credential-like or endpoint identifier refused")
    return value


def digest(value):
    require(isinstance(value, str) and SHA.fullmatch(value), "invalid SHA-256 digest")
    return value


def fields(value, required, optional=()):
    require(isinstance(value, dict), "object required")
    require(set(required) <= value.keys() <= set(required) | set(optional), "unexpected or missing fields")


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON member")
            result[key] = value
        return result

    def constant(_):
        raise InvalidExperiment("nonfinite JSON number")

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
        stack = [(value, 0)]
        count = 0
        while stack:
            item, depth = stack.pop()
            count += 1
            require(depth <= 32 and count <= MAX_NODES, "JSON structure limit exceeded")
            if isinstance(item, float):
                require(math.isfinite(item), "nonfinite JSON number")
            if isinstance(item, dict):
                stack.extend((v, depth + 1) for v in item.values())
            elif isinstance(item, list):
                stack.extend((v, depth + 1) for v in item)
        return value
    except InvalidExperiment:
        raise
    except (UnicodeError, ValueError, RecursionError, OverflowError):
        raise InvalidExperiment("invalid or excessive JSON") from None


def read_fd(fd, limit):
    before = os.fstat(fd)
    require(stat.S_ISREG(before.st_mode), "regular file required")
    require(before.st_size <= limit, "file size limit exceeded")
    chunks, size = [], 0
    while True:
        chunk = os.read(fd, min(65536, limit + 1 - size))
        if not chunk:
            break
        size += len(chunk)
        require(size <= limit, "file size limit exceeded")
        chunks.append(chunk)
    after = os.fstat(fd)
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "file changed during read")
    return b"".join(chunks)


def path_parts(value):
    require(isinstance(value, str) and 0 < len(value) <= 240, "invalid artifact path")
    require("\\" not in value and ":" not in value and "\x00" not in value,
            "invalid artifact path")
    parts = value.split("/")
    require(all(part not in ("", ".", "..") for part in parts), "artifact traversal refused")
    require(all(not any(ord(c) < 32 for c in p) for p in parts), "invalid artifact path")
    return parts


def read_relative(root_fd, parts, limit, seen=None):
    fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            if seen is not None:
                info = os.fstat(file_fd)
                identity = (info.st_dev, info.st_ino)
                require(identity not in seen, "duplicate artifact file identity")
                seen.add(identity)
            return read_fd(file_fd, limit)
        finally:
            os.close(file_fd)
    finally:
        os.close(fd)


def validate_manifest(value):
    fields(value, {"schema", "version", "identity", "model", "runtime", "arms", "artifacts", "analyses"}, {"evidence"})
    require(value["schema"] == "spark-experiment" and type(value["version"]) is int
            and value["version"] == 1, "unsupported experiment schema")
    fields(value["identity"], {"id"})
    token(value["identity"]["id"])
    for name in ("model", "runtime"):
        fields(value[name], {"id", "revision"}, {"sha256"})
        for key in ("id", "revision"):
            token(value[name][key])
        if "sha256" in value[name]:
            digest(value[name]["sha256"])
    require(isinstance(value["arms"], list) and 1 <= len(value["arms"]) <= 16, "invalid arms")
    arm_ids = set()
    for arm in value["arms"]:
        fields(arm, {"id", "config_sha256"})
        require(token(arm["id"]) not in arm_ids, "duplicate arm")
        arm_ids.add(arm["id"])
        digest(arm["config_sha256"])
    artifacts = value["artifacts"]
    require(isinstance(artifacts, list) and 1 <= len(artifacts) <= MAX_ARTIFACTS, "invalid artifact count")
    ids, paths = set(), set()
    for artifact in artifacts:
        fields(artifact, {"id", "kind", "path", "sha256", "schema", "origin"}, {"arm"})
        require(token(artifact["id"]) not in ids, "duplicate artifact ID")
        ids.add(artifact["id"])
        path_parts(artifact["path"])
        require(artifact["path"] not in paths, "duplicate artifact path")
        paths.add(artifact["path"])
        digest(artifact["sha256"])
        require(isinstance(artifact["kind"], str) and artifact["kind"] in SCHEMAS,
                "unsupported artifact kind")
        require(artifact["schema"] == SCHEMAS[artifact["kind"]], "unsupported artifact schema")
        require(isinstance(artifact["origin"], str) and artifact["origin"] in ORIGINS, "invalid evidence origin")
        if "arm" in artifact:
            require(isinstance(artifact["arm"], str) and artifact["arm"] in arm_ids, "undeclared artifact arm")
    analyses = value["analyses"]
    require(isinstance(analyses, list) and len(analyses) <= 32, "invalid analysis count")
    analysis_ids = set()
    for analysis in analyses:
        fields(analysis, {"id", "tool", "inputs"})
        require(token(analysis["id"]) not in analysis_ids, "duplicate analysis ID")
        analysis_ids.add(analysis["id"])
        require(isinstance(analysis["tool"], str) and analysis["tool"] in {"abharness", "qpanel", "cyclesim", "gb10check", "exl3", "exlr8", "tensorfold"},
                "unsupported analysis tool")
        require(isinstance(analysis["inputs"], dict) and 1 <= len(analysis["inputs"]) <= 5, "invalid analysis inputs")
        for key, ref in analysis["inputs"].items():
            token(key)
            require(isinstance(ref, str) and ref in ids, "unknown artifact reference")
    evidence = value.get("evidence", [])
    require(isinstance(evidence, list) and len(evidence) <= 128, "invalid evidence count")
    evidence_ids = set()
    for item in evidence:
        fields(item, {"id", "status"}, {"artifact", "metric", "value", "unit"})
        require(token(item["id"]) not in evidence_ids, "duplicate evidence ID")
        evidence_ids.add(item["id"])
        require(isinstance(item["status"], str) and item["status"] in STATUSES, "invalid evidence status")
        if "artifact" in item:
            require(isinstance(item["artifact"], str) and item["artifact"] in ids, "unknown evidence artifact")
        if item["status"] in {"measured", "estimated"}:
            require("artifact" in item, "measured/estimated evidence needs an artifact")
        if "value" in item:
            require(item["status"] != "missing" and type(item["value"]) in (int, float, bool), "invalid evidence value")
        for key in ("metric", "unit"):
            if key in item:
                token(item[key])
    return value


def load_manifest(path):
    """Read only explicit references, with bounded, no-follow directory traversal."""
    path = Path(path)
    root_fd = None
    try:
        root_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        seen = set()
        manifest_data = read_relative(root_fd, [path.name], MAX_MANIFEST_BYTES, seen)
        manifest = validate_manifest(strict_json(manifest_data))
        documents, total = {}, 0
        for artifact in manifest["artifacts"]:
            parts = path_parts(artifact["path"])
            data = read_relative(root_fd, parts, min(MAX_ARTIFACT_BYTES, MAX_TOTAL_BYTES - total), seen)
            total += len(data)
            require(hashlib.sha256(data).hexdigest() == artifact["sha256"], "artifact digest mismatch")
            if artifact["kind"] == "prereg.digest":
                try:
                    document = data.decode("ascii").strip()
                except UnicodeError:
                    raise InvalidExperiment("invalid preregistration digest file") from None
                digest(document)
            else:
                document = strict_json(data)
                require(isinstance(document, dict) or
                        (artifact["kind"] == "cyclesim.report" and isinstance(document, list)),
                        "artifact JSON object required")
            documents[artifact["id"]] = document
        return manifest, documents, hashlib.sha256(manifest_data).hexdigest(), total
    except OSError:
        raise InvalidExperiment("local file unavailable, unsafe, or unreadable") from None
    finally:
        if root_fd is not None:
            os.close(root_fd)
