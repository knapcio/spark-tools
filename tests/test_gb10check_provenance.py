# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from common.config import digest, load
from gb10check import backend, cli
from gb10check.provenance import build_provenance, declared_inputs, file_hash

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/gb10check"


def test_explicit_sources_baselines_and_inputs_are_hashed(tmp_path):
    for name in ("kernel.cu", "before.ptx", "include.h", "other.h"):
        (tmp_path / name).write_text(name)
    config = {"target": "sm_121a", "inputs": ["include.h"], "entries": [
        {"source": "kernel.cu", "baseline_ptx": "before.ptx", "inputs": ["other.h", "include.h"]}]}
    report = build_provenance(config, tmp_path, backend.source_path)
    assert report["config_digest"] == digest(config)
    assert report["declared_inputs_sha256"] == {name: file_hash(tmp_path / name)
        for name in ("kernel.cu", "before.ptx", "include.h", "other.h")}
    assert report["target"] == "sm_121a"
    assert report["image"]["status"] == "unknown"
    assert "Not reproducible-build proof" in report["limitation"]
    (tmp_path / "include.h").write_text("changed")
    changed = build_provenance(config, tmp_path, backend.source_path)
    assert changed["config_digest"] == report["config_digest"]
    assert changed["declared_inputs_sha256"]["include.h"] != report["declared_inputs_sha256"]["include.h"]


@pytest.mark.parametrize("names", ["kernel.cu", [123], ["../outside.cu"], ["missing.cu"]])
def test_declared_inputs_fail_closed(names, tmp_path):
    with pytest.raises(ValueError):
        declared_inputs({"inputs": names}, tmp_path, backend.source_path)


def test_executed_and_failed_argv_retained(monkeypatch, tmp_path):
    command = ["nvcc", "-arch=sm_121a", "--cubin", "kernel.cu"]
    calls = []
    def execute(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=3, stdout="", stderr="compiler failed")
    monkeypatch.setattr(backend.subprocess, "run", execute)
    provenance = {"commands": []}
    with pytest.raises(ValueError, match="compilation failed"):
        backend.run(command, tmp_path / "build.log", provenance)
    assert calls == [command]
    assert provenance["commands"] == [{"argv": command, "log": "build.log", "returncode": 3}]
    assert (tmp_path / "build.log").read_text() == "compiler failed"


def test_timeout_retains_attempted_argv(monkeypatch, tmp_path):
    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 600)
    monkeypatch.setattr(backend.subprocess, "run", timeout)
    provenance = {"commands": []}
    with pytest.raises(subprocess.TimeoutExpired):
        backend.run(["ptxas", "kernel.ptx"], tmp_path / "build.log", provenance)
    assert provenance["commands"][0]["returncode"] == "unknown"


def test_local_image_identity_is_filtered_and_used(monkeypatch, tmp_path):
    identifier = "sha256:" + "a" * 64
    calls = []
    def execute(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"Id": identifier,
            "RepoDigests": ["example/toolkit@sha256:" + "b" * 64], "Os": "linux",
            "Architecture": "arm64", "Config": {"Env": ["DO_NOT_RETAIN"]}}))
    monkeypatch.setattr(cli.subprocess, "run", execute)
    identity = cli.image_identity("example/toolkit:13")
    command = cli.container_command(EXAMPLE / "compile-pass.json", "example/toolkit:13",
                                    tmp_path, identity=identity)
    assert calls == [["docker", "image", "inspect", "--format", "{{json .}}", "example/toolkit:13"]]
    assert command[command.index("--entrypoint") + 2] == identifier
    assert json.loads(command[command.index("--image-identity") + 1]) == identity
    assert "DO_NOT_RETAIN" not in json.dumps(identity)
    assert identity["platform"] == {"os": "linux", "architecture": "arm64"}


def test_unobserved_digest_is_only_declared(monkeypatch):
    reference = "example/toolkit@sha256:" + "b" * 64
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    assert cli.image_identity(reference) == {"reference": reference, "status": "unknown",
                                            "declared_digest": "sha256:" + "b" * 64}


def test_failed_backend_preserves_attempted_command_and_known_versions(monkeypatch, tmp_path):
    def execute(argv, **kwargs):
        return SimpleNamespace(returncode=int(argv[0] == "ptxas"), stdout="fixture version", stderr="")
    monkeypatch.setattr(backend.subprocess, "run", execute)
    monkeypatch.setattr(backend.sys, "platform", "linux")
    assert backend.main(["--config", str(EXAMPLE / "compile-pass.json"), "--out", str(tmp_path)]) == 1
    report = load(tmp_path / "report.json")
    assert report["mode"] == "STATIC_NO_GPU"
    assert report["toolchain"] == {"nvcc": "fixture version"}
    assert report["build_provenance"]["toolkit_versions"] == report["toolchain"]
    assert report["build_provenance"]["commands"][-1]["argv"] == ["ptxas", "--version"]
    assert report["build_provenance"]["commands"][-1]["returncode"] == 1


def test_complete_backend_adds_provenance_without_gpu(monkeypatch, tmp_path):
    executed = []
    def execute(argv, **kwargs):
        executed.append(argv)
        text = ""
        if "--version" in argv:
            text = argv[0] + " fixture version"
        elif argv[0] == "nvcc":
            Path(argv[argv.index("-o") + 1]).write_bytes(b"fixture")
            if "--cubin" in argv:
                text = (EXAMPLE / "pass.ptxas").read_text()
        elif "--dump-resource-usage" in argv:
            text = (EXAMPLE / "pass.resources").read_text()
        elif argv[0].endswith("/occupancy"):
            text = json.dumps({"active_blocks_per_sm": 12, "warp_occupancy_pct": 100})
        return SimpleNamespace(returncode=0, stdout=text, stderr="")
    actual_hash = backend.file_hash
    monkeypatch.setattr(backend, "file_hash", lambda path: "c" * 64 if
        str(path).endswith("cuda_occupancy.h") else actual_hash(path))
    monkeypatch.setattr(backend.subprocess, "run", execute)
    monkeypatch.setattr(backend.sys, "platform", "linux")
    output = tmp_path / "report"
    identity = {"status": "observed", "id": "sha256:" + "a" * 64}
    code = backend.main(["--config", str(EXAMPLE / "compile-pass.json"), "--out", str(output),
                         "--image-identity", json.dumps(identity), "--requested-platform", "linux/arm64"])
    report = load(output / "report.json")
    assert code == 0
    assert report["mode"] == "STATIC_NO_GPU"
    assert report["status"] == "PASS"
    provenance = report["build_provenance"]
    assert [row["argv"] for row in provenance["commands"]] == executed
    assert provenance["config_file_sha256"] == file_hash(EXAMPLE / "compile-pass.json")
    assert provenance["occupancy_header"]["sha256"] == "c" * 64
    assert provenance["occupancy_source_sha256"] == file_hash(EXAMPLE.parents[1] / "gb10check/occupancy.cpp")
    assert provenance["toolkit_versions"]["nvcc"] == report["toolchain"]["nvcc"]
    assert provenance["image"] == identity
    assert provenance["platform"]["requested"] == "linux/arm64"
    assert report["entries"][0]["source_sha256"] == file_hash(EXAMPLE / "copy.cu")
