# SPDX-License-Identifier: Apache-2.0
"""Compare two clean GPU-free CUDA builds without artifact normalization."""
from pathlib import Path
import shutil
import stat
import subprocess
import time
from common.config import dump, finite, load
from gb10check.provenance import file_hash


def snapshot_inputs(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    files, total = [], 0
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError("repeat build input directory must contain only regular files and directories")
        if stat.S_ISREG(mode):
            files.append(path)
            total += path.stat().st_size
            if len(files) > 512 or total > 32 * 1024 * 1024:
                raise ValueError("repeat build snapshot is limited to 512 files and 32 MiB")
    if not 1 <= len(files) <= 512 or total > 32 * 1024 * 1024:
        raise ValueError("repeat build snapshot is limited to 512 files and 32 MiB")
    output.mkdir()
    hashes = {}
    for path in files:
        name = path.relative_to(root).as_posix()
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        before = file_hash(path)
        shutil.copyfile(path, target)
        if file_hash(path) != before or file_hash(target) != before:
            raise ValueError("repeat build input changed during snapshot")
        target.chmod(0o444)
        hashes[name] = before
    return hashes


def artifact_hashes(output):
    return {path.relative_to(output).as_posix(): file_hash(path)
            for path in sorted(Path(output).rglob("*")) if path.is_file() and (
                path.suffix in {".cubin", ".ptx", ".sass"} or path.name == "occupancy")}


def build_fingerprint(report, output):
    p = report.get("build_provenance", {})
    fields = ("config_digest", "config_file_sha256", "declared_inputs_sha256", "include_closure_sha256",
              "tool_sources_sha256", "toolkit_versions", "occupancy_header", "occupancy_source_sha256",
              "target", "platform", "image")
    result = {name: p.get(name) for name in fields}
    result["commands"] = [{"argv": command["argv"], "returncode": command["returncode"]}
                          for command in p.get("commands", [])]
    result["artifacts_sha256"] = artifact_hashes(output)
    result["complete"] = bool(report.get("entries")) and "error" not in report and bool(
        result["include_closure_sha256"]) and any(name.endswith(".cubin") for name in result["artifacts_sha256"]) and any(
        name.endswith(".sass") for name in result["artifacts_sha256"])
    return result


def compare_builds(first, second):
    mismatches = [{"field": name, "first": first.get(name), "second": second.get(name)}
                  for name in sorted(set(first) | set(second)) if first.get(name) != second.get(name)]
    return {"status": "PASS" if first.get("complete") and second.get("complete") and not mismatches else "FAIL",
            "mismatches": mismatches, "complete": [first.get("complete", False), second.get("complete", False)]}


def repeat_build(config_path, image, output, runtime="docker", platform="linux/arm64", timeout=900):
    from gb10check.cli import container_command, image_identity
    finite(timeout, "timeout", 1, 3600)
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    root = Path(__file__).resolve().parents[1]
    if output == root or root in output.parents or output == config_path.parent or config_path.parent in output.parents:
        raise ValueError("repeat output must be outside source and input directories")
    config = load(config_path)
    if config.get("target") not in {"sm_121", "sm_121a"} or not config.get("entries") or any(
        entry.get("kind") != "cuda" for entry in config["entries"]):
        raise ValueError("repeat build requires nonempty CUDA entries and a GB10 target")
    identity = image_identity(image, runtime)
    if identity.get("status") != "observed":
        raise ValueError("repeat build requires an observed immutable local image ID")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "mode": "STATIC_NO_GPU", "kind": "two clean build comparison", "status": "FAIL",
              "limitation": "A two-build comparison, not general reproducible-build proof. Raw artifact hashes are never normalized.",
              "image": identity, "builds": []}
    deadline = time.monotonic() + timeout
    try:
        snapshot = output / "inputs"
        hashes = snapshot_inputs(config_path.parent, snapshot)
        report["snapshot_inputs_sha256"] = hashes
        for index in (1, 2):
            if {name: file_hash(snapshot / name) for name in hashes} != hashes:
                raise ValueError("repeat build immutable snapshot changed")
            dest = output / ("build-" + str(index))
            dest.mkdir()
            command = container_command(snapshot / config_path.name, image, dest, runtime, platform, identity)
            command.append("--capture-include-closure")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            proc = subprocess.run(command, timeout=remaining, check=False)
            build = load(dest / "report.json")
            fingerprint = build_fingerprint(build, dest)
            report["builds"].append({"returncode": proc.returncode, "fingerprint": fingerprint})
        report.update(compare_builds(*(build["fingerprint"] for build in report["builds"])))
        if any(row["returncode"] not in (0, 1) for row in report["builds"]):
            report["status"] = "FAIL"
        if {name: file_hash(snapshot / name) for name in hashes} != hashes:
            raise ValueError("repeat build immutable snapshot changed")
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        report["status"] = "FAIL"
        report["error"] = str(exc)
    dump(report, output / "repeat-report.json")
    return int(report["status"] != "PASS")
