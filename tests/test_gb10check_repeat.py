# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from common.config import dump, load
from gb10check import cli, repeat
from gb10check.provenance import dependency_hashes, file_hash


def test_dependency_closure_preserves_paths_and_hashes_escaped_spaces(tmp_path):
    one, two = tmp_path / "one.h", tmp_path / "two words.h"
    one.write_text("one")
    two.write_text("two")
    text = "kernel: " + str(one) + " \\\n " + str(two).replace(" ", "\\ ") + "\n"
    assert dependency_hashes(text) == {str(one): file_hash(one), str(two): file_hash(two)}


@pytest.mark.parametrize("text", ["", "no target", "target:", "target: relative.h", "target: /missing/header.h"])
def test_dependency_closure_rejects_unknown_or_incomplete_files(text):
    with pytest.raises(ValueError):
        dependency_hashes(text)


def test_input_snapshot_is_bounded_and_hash_checked(tmp_path):
    inputs, output = tmp_path / "source", tmp_path / "snapshot"
    inputs.mkdir()
    (inputs / "kernel.cu").write_text("fixture")
    hashes = repeat.snapshot_inputs(inputs, output)
    assert hashes == {"kernel.cu": file_hash(output / "kernel.cu")}
    assert (output / "kernel.cu").stat().st_mode & 0o222 == 0
    (inputs / "reference.h").symlink_to(inputs / "kernel.cu")
    with pytest.raises(ValueError, match="regular"):
        repeat.snapshot_inputs(inputs, tmp_path / "other")


def test_input_snapshot_rejects_oversize_before_copying(tmp_path):
    inputs = tmp_path / "source"
    inputs.mkdir()
    with (inputs / "large.bin").open("wb") as stream:
        stream.truncate(32 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="32 MiB"):
        repeat.snapshot_inputs(inputs, tmp_path / "snapshot")


@pytest.mark.parametrize("field", ["commands", "config_digest", "declared_inputs_sha256", "include_closure_sha256",
    "toolkit_versions", "occupancy_header", "target", "platform", "image", "artifacts_sha256"])
def test_repeat_comparison_reports_every_difference_without_normalization(field):
    first = {"complete": True, field: {"value": "first"}}
    second = {"complete": True, field: {"value": "second"}}
    result = repeat.compare_builds(first, second)
    assert result["status"] == "FAIL"
    assert result["mismatches"] == [{"field": field, "first": first[field], "second": second[field]}]
    assert repeat.compare_builds(first, copy.deepcopy(first))["status"] == "PASS"
    assert repeat.compare_builds({"complete": False}, {"complete": False})["status"] == "FAIL"


def test_artifact_hashes_include_exact_sass_and_cubin_bytes(tmp_path):
    (tmp_path / "kernel.cubin").write_bytes(b"\x00\x01")
    (tmp_path / "kernel.sass").write_text("SASS 1\n")
    before = repeat.artifact_hashes(tmp_path)
    (tmp_path / "kernel.sass").write_text("SASS 1 \n")
    assert repeat.artifact_hashes(tmp_path)["kernel.sass"] != before["kernel.sass"]


def test_repeat_runs_two_clean_pinned_containers_and_compares(monkeypatch, tmp_path):
    inputs = tmp_path / "source"
    inputs.mkdir()
    config = inputs / "compile.json"
    config.write_text(json.dumps({"target": "sm_121", "entries": [{"kind": "cuda", "name": "fixture", "source": "kernel.cu"}]}))
    (inputs / "kernel.cu").write_text("fixture")
    identity = {"reference": "example/toolkit:13", "id": "sha256:" + "a" * 64, "status": "observed"}
    monkeypatch.setattr(cli, "image_identity", lambda *args: identity)
    commands = []
    def execute(command, timeout, check):
        commands.append(command)
        mount = next(value for value in command if value.endswith(",target=/out"))
        dest = Path(mount.split("source=", 1)[1].split(",target=", 1)[0])
        (dest / "fixture").mkdir()
        (dest / "fixture/kernel.cubin").write_bytes(b"cubin fixture")
        (dest / "fixture/kernel.sass").write_text("SASS fixture")
        p = {"config_digest": "a" * 64, "config_file_sha256": "b" * 64, "declared_inputs_sha256": {"kernel.cu": "c" * 64},
             "include_closure_sha256": {"fixture": {"/inputs/kernel.cu": "c" * 64}}, "tool_sources_sha256": {},
             "toolkit_versions": {"nvcc": "fixture"}, "occupancy_header": {"sha256": "d" * 64},
             "occupancy_source_sha256": "e" * 64, "target": "sm_121", "platform": {"requested": "linux/arm64"},
             "image": identity, "commands": [{"argv": ["nvcc", "/inputs/kernel.cu", "-o", "/out/fixture/kernel.cubin"], "returncode": 0}]}
        dump({"build_provenance": p, "entries": [{"status": "PASS"}]}, dest / "report.json")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(repeat.subprocess, "run", execute)
    output = tmp_path / "result"
    assert repeat.repeat_build(config, identity["reference"], output) == 0
    report = load(output / "repeat-report.json")
    assert report["mode"] == "STATIC_NO_GPU" and report["status"] == "PASS"
    assert report["snapshot_inputs_sha256"]
    assert len(commands) == 2
    for command in commands:
        assert command[-1] == "--capture-include-closure"
        assert command[command.index("--entrypoint") + 2] == identity["id"]
        assert "--gpus" not in command
        assert command[command.index("--network") + 1] == "none"
    assert commands[0] != commands[1]  # Separate host output mounts, identical in-container paths.


def test_repeat_refuses_unknown_image_and_unsupported_compiler_before_building(monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"target": "sm_121", "entries": [{"kind": "triton"}]}))
    with pytest.raises(ValueError, match="CUDA"):
        repeat.repeat_build(config, "example/toolkit:13", tmp_path.parent / (tmp_path.name + "-out"))
    config.write_text(json.dumps({"target": "sm_121", "entries": [{"kind": "cuda"}]}))
    monkeypatch.setattr(cli, "image_identity", lambda *args: {"status": "unknown"})
    with pytest.raises(ValueError, match="immutable"):
        repeat.repeat_build(config, "example/toolkit:13", tmp_path.parent / (tmp_path.name + "-out"))
