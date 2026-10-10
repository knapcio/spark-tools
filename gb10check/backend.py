# SPDX-License-Identifier: MIT
# Copyright (c) 2026 knapcio
"""Container-only compiler adapter: never load a CUDA module or extension."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from common.config import dump, integer, load
from gb10check.resources import check_resources, parse_cuobjdump, parse_ptxas
from gb10check.provenance import build_provenance, dependency_hashes, file_hash


def source_path(root, value):
    path = (root / value).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError("source must be a file within the input directory")
    return path


def run(command, logfile, provenance=None, timeout=600):
    record = {"argv": list(map(str, command)), "log": logfile.name, "returncode": "unknown"}
    if provenance is not None:
        provenance["commands"].append(record)
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    record["returncode"] = proc.returncode
    logfile.write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise ValueError("static compilation failed; inspect " + logfile.name)
    return proc.stdout + proc.stderr


def compile_entry(entry, target, inputs, output, provenance=None, capture_include_closure=False):
    name = entry["name"]
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
        raise ValueError("entry name must be a safe filename")
    source = source_path(inputs, entry["source"])
    dest = output / name
    dest.mkdir(exist_ok=False)
    binary, log = dest / "kernel.cubin", dest / "ptxas.log"
    kind = entry["kind"]
    metadata = {}
    if kind == "cuda":
        flags = entry.get("flags", [])
        if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
            raise ValueError("compiler flags must be strings")
        base = ["nvcc", "-O3", "-std=c++17", "-arch=" + target, *flags]
        if capture_include_closure:
            dependencies = run([*base, "-M", "-MT", name, str(source)], dest / "dependencies.txt", provenance)
            provenance.setdefault("include_closure_sha256", {})[name] = dependency_hashes(dependencies)
        run([*base, "--cubin", "-Xptxas=-v,--warn-on-spills", str(source), "-o", str(binary)], log, provenance)
        run([*base, "--ptx", str(source), "-o", str(dest / "kernel.ptx")], dest / "ptx.log", provenance)
    elif kind == "ptx":
        (dest / "kernel.ptx").write_bytes(source.read_bytes())
        run(["ptxas", "-arch=" + target, "-v", "--warn-on-spills", str(source), "-o", str(binary)], log, provenance)
    elif kind == "triton":
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("static_input", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, entry["function"])
        attributes = {(function.arg_names.index(key),): value for key, value in entry.get("attrs", {}).items()}
        ast = ASTSource(function, signature=entry["signature"], constexprs=entry.get("constants", {}), attrs=attributes)
        def freeze(value):
            return tuple(freeze(item) for item in value) if isinstance(value, list) else value
        options = {key: freeze(value) for key, value in entry.get("options", {}).items()}
        if provenance is not None:
            provenance["toolkit_versions"]["triton"] = triton.__version__
            provenance.setdefault("compiler_api", []).append({"entry": name, "api": "triton.compile",
                "target": {"backend": "cuda", "arch": 121, "warp_size": 32},
                "signature": entry["signature"], "constants": entry.get("constants", {}),
                "attrs": entry.get("attrs", {}), "options": entry.get("options", {})})
        kernel = triton.compile(ast, target=GPUTarget("cuda", 121, 32), options=options)
        # Reading asm and metadata is static. n_regs/n_spills properties can
        # initialize a driver handle, so they are deliberately never accessed.
        binary.write_bytes(kernel.asm["cubin"])
        (dest / "kernel.ptx").write_text(kernel.asm["ptx"])
        metadata = kernel.metadata._asdict()
        dump(metadata, dest / "metadata.json")
        from triton.backends.nvidia.compiler import get_ptxas
        ptxas = get_ptxas(121).path
        if provenance is not None:
            provenance["toolkit_versions"]["triton_ptxas"] = run(
                [str(ptxas), "--version"], dest / "triton-ptxas-version.txt", provenance)
        run([str(ptxas), "-arch=" + target, "-v", "--warn-on-spills",
             str(dest / "kernel.ptx"), "-o", str(dest / "reassembled.cubin")], log, provenance)
    else:
        raise ValueError("entry kind must be cuda, ptx or triton")
    run(["cuobjdump", "--dump-sass", str(binary)], dest / "kernel.sass", provenance)
    binary_text = run(["cuobjdump", "--dump-resource-usage", str(binary)], dest / "resources.txt", provenance)
    resources = parse_ptxas(log.read_text())
    launch = entry["kernels"]
    if kind == "triton":
        expected_threads = integer(metadata["num_warps"], "warp count", 1) * 32
        for value in launch.values():
            if value.get("threads") != expected_threads or value.get("dynamic_shared_bytes") != metadata["shared"]:
                raise ValueError("Triton launch config disagrees with static metadata")
    report = check_resources(resources, {"target": target, "kernels": launch,
                              "thresholds": entry.get("thresholds", {})}, parse_cuobjdump(binary_text))
    # The installed toolkit header provides more exact allocation rules than
    # the Mac fallback. The calculator is a CPU binary without CUDA linkage.
    for resource, row in zip(resources, report["resources"]):
        geometry = launch[resource.name]
        try:
            row["toolkit_occupancy"] = json.loads(run([str(output / "occupancy"), str(resource.registers),
                str(geometry["threads"]), str(resource.static_shared_bytes),
                str(geometry["dynamic_shared_bytes"]), str(max(1, resource.barriers))],
                dest / (resource.name + "-occupancy.log"), provenance))
            row["checks"]["launch_geometry"] = row["toolkit_occupancy"]["active_blocks_per_sm"] > 0
            row["checks"]["occupancy"] = row["toolkit_occupancy"]["warp_occupancy_pct"] >= entry.get("thresholds", {}).get("min_occupancy_pct", 0)
        except ValueError:
            row["toolkit_occupancy"] = {"status": "unknown", "reason": "toolkit rejected launch resources"}
            row["checks"]["launch_geometry"] = False
        row["status"] = "PASS" if all(row["checks"].values()) else "FAIL"
    report["status"] = "PASS" if all(row["status"] == "PASS" for row in report["resources"]) else "FAIL"
    from gb10check.ordering import barrier_cut, instruction_checks
    checks = {}
    for assembly in ("ptx", "sass"):
        if entry.get("ordering", {}).get(assembly):
            baseline = source_path(inputs, entry["baseline_" + assembly]).read_text() if entry.get("baseline_" + assembly) else None
            checks[assembly] = instruction_checks((dest / ("kernel." + assembly)).read_text(), entry["ordering"][assembly], baseline)
    if entry.get("barrier_cut"):
        checks["barrier_cut"] = barrier_cut((dest / "kernel.ptx").read_text(), **entry["barrier_cut"])["checks"]
    report["ordering"] = checks
    if any(not passed for group in checks.values() for passed in group.values()):
        report["status"] = "FAIL"
    report["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    report["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-identity", type=json.loads)
    parser.add_argument("--requested-platform")
    parser.add_argument("--capture-include-closure", action="store_true")
    args = parser.parse_args(argv)
    if sys.platform != "linux" or list(Path("/dev").glob("nvidia*")):
        parser.error("compiler adapter requires GPU-free Linux")
    config = load(args.config)
    if config.get("target") not in {"sm_121", "sm_121a"} or not config.get("entries"):
        parser.error("GB10 target and nonempty entry list required")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"schema": 1, "mode": "STATIC_NO_GPU", "status": "FAIL", "entries": []}
    try:
        provenance = build_provenance(config, args.config.parent, source_path,
                                      args.image_identity, args.requested_platform)
        report["build_provenance"] = provenance
        provenance["config_file_sha256"] = file_hash(args.config)
        if args.capture_include_closure:
            if any(entry.get("kind") != "cuda" for entry in config["entries"]):
                raise ValueError("include closure capture supports CUDA entries only")
            provenance["tool_sources_sha256"] = {path.name: file_hash(path) for path in sorted(
                Path(__file__).parent.iterdir()) if path.is_file() and path.suffix in {".py", ".cpp"}}
            provenance["tool_sources_sha256"].update({"common/" + path.name: file_hash(path) for path in
                sorted(Path(__file__).parents[1].joinpath("common").glob("*.py"))})
        versions = {}
        report["toolchain"] = versions
        for name in ("nvcc", "ptxas", "cuobjdump"):
            versions[name] = run([name, "--version"], args.out / (name + "-version.txt"), provenance)
            provenance["toolkit_versions"][name] = versions[name]
        provenance["toolkit_versions"]["c++"] = run(["c++", "--version"], args.out / "c++-version.txt", provenance)
        provenance["occupancy_header"]["sha256"] = file_hash(provenance["occupancy_header"]["path"])
        if args.capture_include_closure:
            dependencies = run(["c++", "-std=c++17", "-I/usr/local/cuda/include", "-M", "-MT", "occupancy",
                str(Path(__file__).with_name("occupancy.cpp"))], args.out / "occupancy-dependencies.txt", provenance)
            provenance.setdefault("include_closure_sha256", {})["occupancy"] = dependency_hashes(dependencies)
        run(["c++", "-O2", "-std=c++17", "-I/usr/local/cuda/include",
             str(Path(__file__).with_name("occupancy.cpp")), "-o", str(args.out / "occupancy")], args.out / "occupancy-build.log", provenance)
        for entry in config["entries"]:
            report["entries"].append(compile_entry(entry, config["target"], args.config.parent, args.out,
                                                   provenance, args.capture_include_closure))
        report["status"] = "PASS" if all(row["status"] == "PASS" for row in report["entries"]) else "FAIL"
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        report["error"] = str(exc)
    dump(report, args.out / "report.json")
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
