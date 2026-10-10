# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path

import pytest
from common.config import load
from gb10check import cli, device

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/gb10check/device-cases.json"


def config(case=None):
    return {"schema": 1, "target": "sm_121", "cases": [case or {"id": "baseline", "threads": 128}]}


def observation():
    return {"device": {"major": 12, "minor": 1, "multiProcessorCount": 48, "warpSize": 32,
        "regsPerMultiprocessor": 65536, "maxThreadsPerMultiProcessor": 1536, "maxBlocksPerMultiProcessor": 24,
        "sharedMemPerMultiprocessor": 102400, "sharedMemPerBlockOptin": 101376, "reservedSharedMemPerBlock": 1024,
        "maxThreadsPerBlock": 1024},
        "attributes": {"numRegs": 32, "sharedSizeBytes": 0, "maxThreadsPerBlock": 1024,
                       "maxDynamicSharedSizeBytes": 49152, "preferredShmemCarveout": -1},
        "runtime_error": 0, "runtime_blocks_per_sm": 12, "header_error": 0, "header_blocks_per_sm": 12,
        "set_dynamic_error": 0, "set_carveout_error": 0, "attributes_error": 0}


def test_bounded_fixture_covers_every_block_size_and_resource_boundaries():
    cases = device.validate_cases(load(FIXTURE))
    assert len(cases) == 92
    assert {case["threads"] for case in cases if case["id"].startswith("block-")} == set(range(32, 1025, 32))
    assert {case["register_pressure"] for case in cases} >= {31, 32, 33, 63, 64, 65, 127, 128, 129, 223}
    assert {case["static_shared_bytes"] for case in cases} >= {127, 128, 129, 49151, 49152}
    assert {case["dynamic_shared_bytes"] for case in cases} >= {49151, 49152, 49153, 101375, 101376, 101377}
    assert {case["flags"] for case in cases} == {0, 1}
    assert {"valid", "invalid"} == {case["expect"] for case in cases}


@pytest.mark.parametrize("field,value", [
    ("threads", True), ("threads", 1.5), ("threads", -1), ("threads", 2049),
    ("register_pressure", 225), ("static_shared_bytes", 49153), ("dynamic_shared_bytes", 131073),
    ("launch_bound", 31), ("launch_bound", 1025), ("optin_dynamic_bytes", -2),
    ("carveout", -2), ("carveout", 101), ("flags", 2), ("expect", "unknown"),
    ("id", "../escape"), ("id", "bad/name"), ("id", ""), ("other", 1)])
def test_case_contract_refuses_unsafe_or_unbounded_inputs(field, value):
    case = {"id": "baseline", "threads": 128, field: value}
    with pytest.raises(ValueError):
        device.validate_cases(config(case))


@pytest.mark.parametrize("change", ["extra", "schema", "target", "empty", "many", "duplicate"])
def test_case_root_contract(change):
    value = config()
    if change == "extra": value["extra"] = 1
    if change == "schema": value["schema"] = True
    if change == "target": value["target"] = "sm_120"
    if change == "empty": value["cases"] = []
    if change == "many": value["cases"] *= 129
    if change == "duplicate": value["cases"] *= 2
    with pytest.raises(ValueError):
        device.validate_cases(value)


def test_generated_sources_are_deterministic_and_no_launch_api_exists():
    cases = device.validate_cases(load(FIXTURE))
    source, variants = device.kernel_sources(cases)
    assert (source, variants) == device.kernel_sources(cases)
    assert len(variants) == 33
    assert "__syncthreads();" in source and "__launch_bounds__(128)" in source
    helper = (ROOT / "gb10check/device_probe.cpp").read_text()
    for forbidden in ("<<<", "cudaMalloc", "cudaLaunchKernel", "cudaGetDeviceCount"):
        assert forbidden not in helper + source
    assert "cudaFuncGetAttributes" in helper
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor(" in helper
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(" in helper
    assert "cudaOccDeviceProp header_device(p)" in helper
    assert "a.preferredShmemCarveout" in helper


@pytest.mark.parametrize("threads", range(32, 1025, 32))
def test_comparison_retains_cpu_estimate_and_signed_error(threads):
    case = device.validate_cases(config({"id": "baseline", "threads": threads}))[0]
    raw = observation()
    raw["runtime_blocks_per_sm"] = raw["header_blocks_per_sm"] = 1
    result = device.compare_case(case, raw, 1, "sm_121")
    assert result["status"] == "PASS"
    assert result["observations"] == raw
    assert result["cpu_approximation"]["status"] == "estimated"
    assert result["cpu_blocks_per_sm_error"] == result["cpu_approximation"]["active_blocks_per_sm"] - 1
    assert "not measured occupancy or speed" in result["basis"]


@pytest.mark.parametrize("field,value", [("runtime_blocks_per_sm", 11), ("header_error", 1),
    ("runtime_error", 1), ("set_dynamic_error", 1), ("set_carveout_error", 1), ("attributes_error", 1)])
def test_valid_case_never_hides_mismatches_or_errors(field, value):
    raw = observation()
    raw[field] = value
    result = device.compare_case(device.validate_cases(config())[0], raw, 1, "sm_121")
    assert result["status"] == "FAIL"


@pytest.mark.parametrize("setting", ["runtime", "setter", "fatal", "accepted", "disagree"])
def test_invalid_case_distinguishes_rejection_from_environment_failure(setting):
    raw = observation()
    case = device.validate_cases(config({"id": "invalid", "threads": 128,
        "dynamic_shared_bytes": 49153, "expect": "invalid"}))[0]
    if setting == "runtime": raw.update(runtime_error=1, header_error=1)
    if setting == "setter": raw["set_dynamic_error"] = 1
    if setting == "fatal": raw["runtime_error"] = 999
    if setting == "disagree": raw["runtime_blocks_per_sm"] = 0
    result = device.compare_case(case, raw, 1, "sm_121")
    assert result["status"] == ("FAIL" if setting == "fatal" else "PASS")


def test_explicit_launch_guard_rejects_bound_exceeded_even_when_occupancy_apis_return_estimates():
    raw = observation()
    raw["attributes"]["maxThreadsPerBlock"] = 128
    raw["runtime_blocks_per_sm"] = raw["header_blocks_per_sm"] = 9
    case = device.validate_cases(config({"id": "bound-exceeded", "threads": 129,
        "launch_bound": 128, "expect": "invalid"}))[0]
    result = device.compare_case(case, raw, 1, "sm_121")
    assert result["status"] == "PASS"
    assert result["launch_validity"]["status"] == "invalid"
    assert result["launch_validity"]["checks"]["within_function_max_threads"] is False
    assert result["comparison"] == "invalid request rejected by explicit launch-validity guard"
    raw["runtime_error"] = 999
    assert device.compare_case(case, raw, 1, "sm_121")["status"] == "FAIL"


def test_explicit_launch_guard_rejects_dynamic_shared_limit_exceeded_even_when_apis_accept():
    raw = observation()
    raw["runtime_blocks_per_sm"] = raw["header_blocks_per_sm"] = 1
    case = device.validate_cases(config({"id": "dynamic-shared-exceeded", "threads": 128,
        "dynamic_shared_bytes": 49153, "expect": "invalid"}))[0]
    result = device.compare_case(case, raw, 1, "sm_121")
    assert result["status"] == "PASS"
    assert result["launch_validity"]["status"] == "invalid"
    assert result["launch_validity"]["checks"]["within_function_dynamic_shared"] is False
    assert result["comparison"] == "invalid request rejected by explicit launch-validity guard"


def fake_compiler(monkeypatch, output, raw=None):
    actual_hash = device.file_hash
    monkeypatch.setattr(device.sys, "platform", "linux")
    monkeypatch.setattr(device, "file_hash", lambda path: "a" * 64 if str(path).endswith(
        "cuda_occupancy.h") else actual_hash(path))
    calls = []
    def execute(command, logfile, provenance, timeout):
        calls.append(command)
        provenance["commands"].append({"argv": command, "returncode": 0})
        if "--version" in command:
            text = "fixture version"
        elif "-M" in command:
            (output / "cuda_occupancy.h").write_text("header fixture")
            text = "probe: " + str(output / "kernels.inc") + " " + str(output / "cuda_occupancy.h") + "\n"
        elif command[0] == "nvcc":
            (output / "probe").write_bytes(b"executable fixture")
            text = (ROOT / "examples/gb10check/pass.ptxas").read_text().replace("sample_copy", "fixture_0")
        else:
            text = json.dumps(raw or observation())
        logfile.write_text(text)
        return text
    monkeypatch.setattr(device, "run", execute)
    return calls


def test_native_command_builds_queries_and_records_explicit_device(monkeypatch, tmp_path):
    cases_path, output = tmp_path / "cases.json", tmp_path.parent / (tmp_path.name + "-result")
    cases_path.write_text(json.dumps(config()))
    calls = fake_compiler(monkeypatch, output)
    assert device.validate_device(cases_path, output, 3) == 0
    report = load(output / "report.json")
    assert report["mode"] == "DEVICE_RESIDENCY_API"
    assert report["device_ordinal"] == 3
    assert report["cases"][0]["status"] == "PASS"
    assert report["build_provenance"]["include_closure_sha256"]
    assert calls[-1][1] == "3"
    with pytest.raises(FileExistsError):
        device.validate_device(cases_path, output, 3)


def test_build_only_never_calls_probe(monkeypatch, tmp_path):
    cases_path, output = tmp_path / "cases.json", tmp_path.parent / (tmp_path.name + "-result")
    cases_path.write_text(json.dumps(config()))
    calls = fake_compiler(monkeypatch, output)
    assert device.validate_device(cases_path, output, 0, build_only=True) == 0
    assert load(output / "report.json")["mode"] == "BUILD_ONLY_NO_GPU"
    assert not any(command[0].endswith("/probe") for command in calls)


def test_device_failure_stops_without_trying_another_ordinal(monkeypatch, tmp_path):
    cases_path, output = tmp_path / "cases.json", tmp_path.parent / (tmp_path.name + "-result")
    cases_path.write_text(json.dumps(config()))
    calls = fake_compiler(monkeypatch, output, {"setup_error": 100})
    assert device.validate_device(cases_path, output, 0) == 1
    assert len([command for command in calls if command[0].endswith("/probe")]) == 1


def test_cli_requires_device_and_never_schedules_gpu_work_implicitly(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["validate-device", "--cases", str(FIXTURE), "--out", str(tmp_path / "report")])
    monkeypatch.setattr(device.sys, "platform", "darwin")
    with pytest.raises(ValueError, match="Linux"):
        device.validate_device(FIXTURE, tmp_path / "report", 0)
