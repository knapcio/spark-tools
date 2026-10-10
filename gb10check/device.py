# SPDX-License-Identifier: Apache-2.0
"""Explicit GB10 residency API comparison; never execute a kernel."""
import argparse
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
import time
from common.config import digest, dump, finite, integer, load
from gb10check.backend import run
from gb10check.provenance import dependency_hashes, file_hash
from gb10check.resources import Resource, occupancy, parse_ptxas


FIELDS = {"id", "threads", "register_pressure", "static_shared_bytes", "dynamic_shared_bytes",
          "launch_bound", "optin_dynamic_bytes", "carveout", "flags", "expect"}


def validate_cases(config):
    if set(config) != {"schema", "target", "cases"} or type(config["schema"]) is not int or config["schema"] != 1:
        raise ValueError("device cases require schema 1, target and cases")
    if config["target"] not in {"sm_121", "sm_121a"}:
        raise ValueError("device validation requires a GB10 target")
    cases = config["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 128:
        raise ValueError("device validation requires 1..128 cases")
    names, result = set(), []
    for source in cases:
        if not isinstance(source, dict) or set(source) - FIELDS or not {"id", "threads"} <= set(source):
            raise ValueError("invalid device case fields")
        case = dict(register_pressure=0, static_shared_bytes=0, dynamic_shared_bytes=0,
                    launch_bound=1024, optin_dynamic_bytes=-1, carveout=-1, flags=0, expect="valid")
        case.update(source)
        name = case["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name) or name in names:
            raise ValueError("device case IDs must be unique safe names")
        names.add(name)
        for field, low, high in (("threads", 0, 2048), ("register_pressure", 0, 224),
                                 ("static_shared_bytes", 0, 49152), ("dynamic_shared_bytes", 0, 131072),
                                 ("launch_bound", 32, 1024), ("optin_dynamic_bytes", -1, 131072),
                                 ("carveout", -1, 101), ("flags", 0, 1)):
            integer(case[field], field, low, high)
        if case["expect"] not in {"valid", "invalid"}:
            raise ValueError("device case expect must be valid or invalid")
        if case["expect"] == "valid" and (not 32 <= case["threads"] <= 1024 or case["carveout"] > 100):
            raise ValueError("valid cases require 32..1024 threads and a valid carveout")
        result.append(case)
    return result


def kernel_sources(cases):
    variants = list(dict.fromkeys((case["register_pressure"], case["static_shared_bytes"], case["launch_bound"])
                                  for case in cases))
    text = ["// SPDX-License-Identifier: Apache-2.0"]
    for index, (pressure, shared, bound) in enumerate(variants):
        text.append(f'extern "C" __global__ __launch_bounds__({bound}) void fixture_{index}(unsigned* output) {{')
        text.append("unsigned value=threadIdx.x;")
        for register in range(pressure):
            text.append(f'unsigned r{register}; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(r{register}) : "l"(output + {register} * 1024 + threadIdx.x));')
        if shared:
            text.append(f"__shared__ volatile unsigned char tile[{shared}]; tile[threadIdx.x % {shared}]=value;")
        text.append("__syncthreads();")
        if pressure:
            text.append('unsigned selector; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(selector) : "l"(output));')
            text.append(f"switch(selector % {pressure}) {{")
            text.extend(f"case {register}: value += r{register}; break;" for register in range(pressure))
            text.append("}")
        if shared:
            text.append(f"value += tile[(threadIdx.x + 1) % {shared}];")
        text.append("output[threadIdx.x]=value; }")
    text.append("static const void* fixture_functions[]={" + ",".join(f"(const void*)fixture_{i}" for i in range(len(variants))) + "};")
    return "\n".join(text) + "\n", variants


def compare_case(case, raw, barriers, target):
    report = {"id": case["id"], "request": case, "observations": raw, "status": "FAIL",
              "basis": "runtime API residency estimate; not measured occupancy or speed"}
    if "device" not in raw or raw.get("setup_error") or raw.get("unsupported_device"):
        report["comparison"] = "device query failed"
        return report
    errors = any(raw[name] != 0 for name in ("set_dynamic_error", "set_carveout_error", "attributes_error"))
    runtime_valid = raw["runtime_error"] == 0 and raw["runtime_blocks_per_sm"] > 0
    header_valid = raw["header_error"] == 0 and raw["header_blocks_per_sm"] > 0
    launch_checks = {
        "positive_threads": case["threads"] > 0,
        "within_device_max_threads": case["threads"] <= raw["device"]["maxThreadsPerBlock"],
        "within_function_max_threads": case["threads"] <= raw["attributes"]["maxThreadsPerBlock"],
        "within_function_dynamic_shared": (
            case["dynamic_shared_bytes"] <= raw["attributes"]["maxDynamicSharedSizeBytes"]),
    }
    report["launch_validity"] = {
        "status": "valid" if all(launch_checks.values()) else "invalid",
        "checks": launch_checks,
        "basis": "explicit device and cudaFuncAttributes launch limits; occupancy APIs are not launch validators",
    }
    report["blocks_per_sm_error"] = (raw["header_blocks_per_sm"] - raw["runtime_blocks_per_sm"]
        if raw["runtime_error"] == raw["header_error"] == 0 else None)
    if case["expect"] == "invalid":
        expected_errors = all(raw[name] in (0, 1) for name in
            ("set_dynamic_error", "set_carveout_error", "runtime_error", "header_error"))
        launch_invalid = not all(launch_checks.values())
        agreed = raw["attributes_error"] == 0 and expected_errors and (
            launch_invalid or errors or (not runtime_valid and not header_valid))
        if agreed and launch_invalid:
            report["comparison"] = "invalid request rejected by explicit launch-validity guard"
        else:
            report["comparison"] = "invalid request rejected" if agreed else "invalid request accepted or calculators disagree"
    else:
        agreed = all(launch_checks.values()) and not errors and runtime_valid and header_valid and report["blocks_per_sm_error"] == 0
        report["comparison"] = "matched" if agreed else "mismatch"
    report["status"] = "PASS" if agreed else "FAIL"
    p, a = raw["device"], raw["attributes"]
    try:
        approximation = occupancy(Resource("fixture", target, a["numRegs"], 0, 0, 0, a["sharedSizeBytes"], barriers),
            case["threads"], case["dynamic_shared_bytes"],
            {"sms": p["multiProcessorCount"], "registers_per_sm": p["regsPerMultiprocessor"],
             "warps_per_sm": p["maxThreadsPerMultiProcessor"] // p["warpSize"],
             "blocks_per_sm": p["maxBlocksPerMultiProcessor"], "shared_per_sm": p["sharedMemPerMultiprocessor"],
             "shared_per_block": p["sharedMemPerBlockOptin"], "shared_reserved": p["reservedSharedMemPerBlock"]})
    except ValueError:
        approximation = {"status": "unknown", "reason": "outside CPU approximation input limits"}
    report["cpu_approximation"] = approximation
    report["cpu_blocks_per_sm_error"] = (approximation["active_blocks_per_sm"] - raw["runtime_blocks_per_sm"]
        if "active_blocks_per_sm" in approximation and raw["runtime_error"] == 0 else None)
    return report


def validate_device(cases_path, output, device, timeout=600, build_only=False):
    if sys.platform != "linux":
        raise ValueError("device validation requires explicit execution on Linux")
    integer(device, "device ordinal", 0, 31)
    finite(timeout, "timeout", 1, 3600)
    deadline = time.monotonic() + timeout
    config = load(cases_path)
    cases = validate_cases(config)
    output = Path(output).resolve()
    root = Path(__file__).resolve().parents[1]
    inputs = Path(cases_path).resolve().parent
    if output == root or root in output.parents or output == inputs or inputs in output.parents:
        raise ValueError("device validation output must be outside source and input directories")
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"config_digest": digest(config), "config_file_sha256": file_hash(cases_path),
                  "commands": [], "toolkit_versions": {}, "helper_source_sha256": file_hash(Path(__file__).with_name("device_probe.cpp")),
                  "target": config["target"], "platform": {"system": platform.system(), "machine": platform.machine()},
                  "image": {"status": "unknown", "basis": "native execution"}}
    report = {"schema": 1, "mode": "BUILD_ONLY_NO_GPU" if build_only else "DEVICE_RESIDENCY_API",
              "status": "FAIL", "device_ordinal": device, "cases": [], "build_provenance": provenance,
              "limitation": "Runtime API residency estimates and calculator agreement; no kernel execution, measured occupancy or speed."}
    try:
        def execute(command, logfile):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            return run(command, logfile, provenance, timeout=remaining)
        source, variants = kernel_sources(cases)
        (output / "kernels.inc").write_text(source)
        provenance["kernels_sha256"] = file_hash(output / "kernels.inc")
        for tool in ("nvcc", "ptxas", "c++"):
            provenance["toolkit_versions"][tool] = execute([tool, "--version"], output / (tool + "-version.txt"))
        command = ["nvcc", "-x", "cu", "-O3", "-std=c++17", "-arch=" + config["target"], "-Xptxas=-v",
                   "-I" + str(output), str(Path(__file__).with_name("device_probe.cpp")), "-o", str(output / "probe")]
        dependencies = execute([*command[:-3], "-M", "-MT", "probe", str(Path(__file__).with_name("device_probe.cpp"))],
                               output / "dependencies.txt")
        provenance["include_closure_sha256"] = dependency_hashes(dependencies)
        headers = [name for name in provenance["include_closure_sha256"] if Path(name).name == "cuda_occupancy.h"]
        if len(headers) != 1:
            raise ValueError("device compiler closure must identify exactly one occupancy header")
        provenance["occupancy_header"] = {"path": headers[0], "sha256": provenance["include_closure_sha256"][headers[0]]}
        provenance["occupancy_header_sha256"] = provenance["occupancy_header"]["sha256"]
        execute(command, output / "compile.log")
        provenance["executable_sha256"] = file_hash(output / "probe")
        resources = {row.name: row for row in parse_ptxas((output / "compile.log").read_text())}
        report["compiled_resources"] = {name: {"registers": row.registers, "static_shared_bytes": row.static_shared_bytes,
            "barriers": row.barriers, "spill_store_bytes": row.spill_store_bytes, "spill_load_bytes": row.spill_load_bytes}
            for name, row in resources.items()}
        if file_hash(cases_path) != provenance["config_file_sha256"]:
            raise ValueError("device case file changed during compilation")
        if build_only:
            report["status"] = "PASS"
        else:
            for case in cases:
                variant = variants.index((case["register_pressure"], case["static_shared_bytes"], case["launch_bound"]))
                resource = resources["fixture_" + str(variant)]
                arguments = [device, variant, case["threads"], case["dynamic_shared_bytes"],
                             case["optin_dynamic_bytes"], case["carveout"], case["flags"], resource.barriers]
                raw = json.loads(execute([str(output / "probe"), *map(str, arguments)], output / (case["id"] + ".json")))
                report["cases"].append(compare_case(case, raw, resource.barriers, config["target"]))
                if "device" not in raw:
                    break
            report["status"] = "PASS" if len(report["cases"]) == len(cases) and all(row["status"] == "PASS" for row in report["cases"]) else "FAIL"
    except (ValueError, OSError, KeyError, subprocess.TimeoutExpired) as exc:
        report["error"] = str(exc)
    finally:
        dump(report, output / "report.json")
    return int(report["status"] != "PASS")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args(argv)
    return validate_device(args.cases, args.out, args.device, build_only=args.build_only)


if __name__ == "__main__":
    raise SystemExit(main())
