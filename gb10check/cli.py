# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path
import re
import subprocess
from common.config import dump, load
from gb10check.resources import check_resources, parse_cuobjdump, parse_ptxas
from gb10check.ordering import barrier_cut, instruction_checks


def inspect(config, directory):
    directory = Path(directory)
    resources = parse_ptxas((directory / config["ptxas"]).read_text())
    binary = parse_cuobjdump((directory / config["binary_resources"]).read_text()) if config.get("binary_resources") else None
    report = check_resources(resources, config, binary)
    checks = {}
    for kind in ("ptx", "sass"):
        spec = config.get("ordering", {}).get(kind)
        if spec:
            text = (directory / config[kind]).read_text()
            baseline = (directory / config["baseline_" + kind]).read_text() if config.get("baseline_" + kind) else None
            checks[kind] = instruction_checks(text, spec, baseline)
    if config.get("barrier_cut"):
        cut = barrier_cut((directory / config["ptx"]).read_text(), **config["barrier_cut"])
        checks["barrier_cut"] = cut["checks"]
        report["barrier_cut"] = cut
    report["ordering"] = checks
    if any(not passed for group in checks.values() for passed in group.values()):
        report["status"] = "FAIL"
    return report


def image_identity(image, runtime="docker"):
    """Inspect only the local image; never pull or copy runtime configuration."""
    if not image or image.startswith("-"):
        raise ValueError("a toolkit image is required")
    identity = {"reference": image, "status": "unknown"}
    if "@sha256:" in image:
        identity["declared_digest"] = image.rsplit("@", 1)[1]
    try:
        proc = subprocess.run([runtime, "image", "inspect", "--format", "{{json .}}", image],
                              capture_output=True, text=True, timeout=15, check=False)
        if proc.returncode:
            return identity
        result = json.loads(proc.stdout)
        if not isinstance(result, dict):
            return identity
        identifier = result.get("Id", result.get("ID", ""))
        if not isinstance(identifier, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", identifier):
            return identity
        identity.update(status="observed", id=identifier,
                        repo_digests=result.get("RepoDigests") or [],
                        platform={"os": result.get("Os", "unknown"),
                                  "architecture": result.get("Architecture", "unknown")})
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return identity


def container_command(config, image, output, runtime="docker", platform="linux/arm64", identity=None):
    config, output = Path(config).resolve(), Path(output).resolve()
    root = Path(__file__).resolve().parent.parent
    if output == root or root in output.parents or config.parent == output or config.parent in output.parents:
        raise ValueError("compiler output must be outside the checkout and inputs")
    if not image or image.startswith("-"):
        raise ValueError("a toolkit image is required")
    # Mount only the Python packages and inputs, so an installed wheel also
    # works without exposing host-specific dependencies to the Linux image.
    # No device or network access is requested.
    selected = identity.get("id", image) if identity else image
    return [runtime, "run", "--rm", "--pull", "never", "--platform", platform, "--network", "none",
            "--mount", f"type=bind,source={root / 'gb10check'},target=/tool/gb10check,readonly",
            "--mount", f"type=bind,source={root / 'common'},target=/tool/common,readonly",
            "--mount", f"type=bind,source={config.parent},target=/inputs,readonly",
            "--mount", f"type=bind,source={output},target=/out",
            "--env", "PYTHONPATH=/tool", "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--entrypoint", "python3", selected, "-m", "gb10check.backend",
            "--config", "/inputs/" + config.name, "--out", "/out",
            "--requested-platform", platform, "--image-identity",
            json.dumps(identity or {"reference": image, "status": "unknown"}, sort_keys=True)]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Static GB10 resource and instruction checks")
    commands = parser.add_subparsers(dest="command", required=True)
    parse = commands.add_parser("parse", help="Saved compiler output inspection")
    parse.add_argument("--config", type=Path, required=True)
    parse.add_argument("--out", type=Path)
    compile_parser = commands.add_parser("compile", help="GPU-free Linux container compilation")
    compile_parser.add_argument("--config", type=Path, required=True)
    compile_parser.add_argument("--image", required=True)
    compile_parser.add_argument("--platform", choices=["linux/arm64", "linux/amd64"], default="linux/arm64")
    compile_parser.add_argument("--runtime", default="docker", choices=["docker", "podman"])
    compile_parser.add_argument("--out", type=Path, required=True)
    compile_parser.add_argument("--timeout", type=float, default=900)
    device_parser = commands.add_parser("validate-device", help="Explicit GB10 runtime residency API comparison; no kernel execution")
    device_parser.add_argument("--cases", type=Path, required=True)
    device_parser.add_argument("--device", type=int, required=True)
    device_parser.add_argument("--out", type=Path, required=True)
    device_parser.add_argument("--timeout", type=float, default=600)
    device_parser.add_argument("--build-only", action="store_true", help="Compile the helper without querying a device")
    repeat_parser = commands.add_parser("repeat-build", help="Compare two clean GPU-free CUDA builds with include closure hashes")
    repeat_parser.add_argument("--config", type=Path, required=True)
    repeat_parser.add_argument("--image", required=True)
    repeat_parser.add_argument("--platform", choices=["linux/arm64", "linux/amd64"], default="linux/arm64")
    repeat_parser.add_argument("--runtime", choices=["docker", "podman"], default="docker")
    repeat_parser.add_argument("--out", type=Path, required=True)
    repeat_parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args(argv)
    try:
        if args.command == "parse":
            report = inspect(load(args.config), args.config.parent)
            dump(report, args.out)
            return int(report["status"] != "PASS")
        if args.command == "validate-device":
            from gb10check.device import validate_device
            return validate_device(args.cases, args.out, args.device, args.timeout, args.build_only)
        if args.command == "repeat-build":
            from gb10check.repeat import repeat_build
            return repeat_build(args.config, args.image, args.out, args.runtime, args.platform, args.timeout)
        if args.timeout <= 0:
            raise ValueError("positive timeout required")
        command = container_command(args.config, args.image, args.out, args.runtime, args.platform,
                                    image_identity(args.image, args.runtime))
        args.out.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(command, timeout=args.timeout, check=False)
        return proc.returncode
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        parser.error(str(exc))
